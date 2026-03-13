"""Supervisor: the outer optimization loop.

Ties together: proposer -> rollout -> evaluator -> selection.
"""

import concurrent.futures
import json
import logging
import random
import shutil
import time
from pathlib import Path

from anneal.alphaevolve.fitness import compute_fitness
from anneal.alphaevolve.island import Island, migrate
from anneal.alphaevolve.proposer import propose
from anneal.alphaevolve.rollout import _run_single, _seed_memory
from anneal.alphaevolve.types import IslandConfig, PromptPair, TaskConfig
from anneal.evaluator import evaluate, save_eval

logger = logging.getLogger("anneal.alphaevolve.supervisor")


DEFAULT_ISLANDS = [
    IslandConfig(
        name="first_try",
        objective=(
            "Optimize for FIRST-TRY SUCCESS: the coder should produce a patch "
            "that the reviewer approves on round 1, with no revisions needed."
        ),
        weights={
            "converged": 2.0,
            "rounds": -2.0,
            "coder_score": 1.5,
            "reviewer_score": 0.5,
            "cost": -0.1,
            "steps": -0.01,
        },
    ),
    IslandConfig(
        name="efficiency",
        objective=(
            "Optimize for COST EFFICIENCY: minimize total cost and steps while "
            "still producing correct fixes. The cheapest path to a good result."
        ),
        weights={
            "converged": 2.0,
            "rounds": -0.5,
            "coder_score": 0.5,
            "reviewer_score": 0.5,
            "cost": -3.0,
            "steps": -0.05,
        },
    ),
    IslandConfig(
        name="feedback_quality",
        objective=(
            "Optimize for FEEDBACK QUALITY: the reviewer should give precise, "
            "actionable feedback that leads to meaningful improvement between rounds. "
            "The coder should address feedback thoroughly."
        ),
        weights={
            "converged": 1.0,
            "rounds": 0.0,
            "coder_score": 1.0,
            "reviewer_score": 2.5,
            "cost": -0.1,
            "steps": -0.01,
        },
    ),
    IslandConfig(
        name="correctness",
        objective=(
            "Optimize for CORRECTNESS above all else: the coder should produce "
            "thorough, correct fixes no matter how many rounds or how much cost. "
            "Take as long as needed to get it right."
        ),
        weights={
            "converged": 5.0,
            "rounds": 0.0,
            "coder_score": 3.0,
            "reviewer_score": 1.0,
            "cost": 0.0,
            "steps": 0.0,
        },
    ),
    IslandConfig(
        name="adversarial_reviewer",
        objective=(
            "Optimize for a TOUGH REVIEWER: the reviewer should catch subtle bugs, "
            "missing edge cases, and style violations that other reviewers miss. "
            "It's okay if convergence takes more rounds — the reviewer should not "
            "rubber-stamp patches."
        ),
        weights={
            "converged": 0.5,
            "rounds": 0.0,
            "coder_score": 0.5,
            "reviewer_score": 3.0,
            "cost": -0.1,
            "steps": -0.01,
        },
    ),
]


class Supervisor:
    """Orchestrates the AlphaEvolve-style prompt optimization loop.

    Each generation:
    1. PROPOSE — LLM proposes prompt mutations based on eval feedback
    2. ROLLOUT — run all variants x tasks in parallel on Modal
    3. EVALUATE — LLM judge scores each trace
    4. SELECT — best variant enters the elite pool if it beats the worst
    5. MIGRATE — periodically share best elites across islands
    """

    def __init__(
        self,
        task_pool: list[TaskConfig],
        experiments_dir: str | Path = "experiments",
        baseline_coder: str | Path | None = None,
        baseline_reviewer: str | Path | None = None,
        seed_learnings_dir: str | Path | None = None,
        islands: list[IslandConfig] | None = None,
        *,
        n_variants: int = 4,
        tasks_per_gen: int | None = None,
        max_workers: int = 20,
        max_elites: int = 3,
        migrate_every: int = 2,
        propose_model: str = "anthropic/claude-opus-4-6",
        judge_model: str = "anthropic/claude-opus-4-6",
    ):
        self.task_pool = task_pool
        self.experiments_dir = Path(experiments_dir)
        self.n_variants = n_variants
        self.tasks_per_gen = tasks_per_gen  # None = use all tasks
        self.max_workers = max_workers
        self.migrate_every = migrate_every
        self.propose_model = propose_model
        self.judge_model = judge_model

        # Default baseline prompts
        configs_dir = Path(__file__).resolve().parent.parent / "configs"
        baseline_coder = Path(baseline_coder or configs_dir / "base_coder.yaml")
        baseline_reviewer = Path(baseline_reviewer or configs_dir / "base_reviewer.yaml")
        self.baseline = PromptPair(coder_yaml=baseline_coder, reviewer_yaml=baseline_reviewer)

        # Seed learnings (hand-written repo knowledge for warm start)
        self.seed_learnings_dir = Path(seed_learnings_dir) if seed_learnings_dir else None

        # Set up islands
        island_configs = islands or DEFAULT_ISLANDS
        self.islands = []
        for ic in island_configs:
            island = Island(ic, self.experiments_dir / "islands" / ic.name, max_elites=max_elites)
            elite_dir = self.experiments_dir / "islands" / ic.name / "elites" / "elite_0"
            prompts = self.baseline.copy_to(elite_dir)
            # Copy seed learnings into the baseline elite
            if self.seed_learnings_dir and self.seed_learnings_dir.exists():
                learnings_dest = elite_dir / "learnings"
                learnings_dest.mkdir(exist_ok=True)
                for f in self.seed_learnings_dir.iterdir():
                    if f.name.endswith("_learnings.txt"):
                        shutil.copy2(f, learnings_dest / f.name)
            island.seed_elite(prompts)
            self.islands.append(island)

        self._save_task_pool()

    def _save_task_pool(self):
        pool_path = self.experiments_dir / "task_pool.json"
        if not pool_path.exists():
            pool_path.parent.mkdir(parents=True, exist_ok=True)
            pool_path.write_text(
                json.dumps(
                    [{"task_id": t.task_id, "image": t.image, "issue": t.issue[:500]} for t in self.task_pool],
                    indent=2,
                )
            )

    def _evaluate_variants(self, variants, island):
        """Evaluate all variants' traces and compute fitness."""
        for variant in variants:
            evals = []
            for task_id, trace_path in variant.traces.items():
                try:
                    ev = evaluate(trace_path, run_judge=True, judge_model=self.judge_model)
                    variant.evals[task_id] = ev
                    save_eval(ev, trace_path.parent / "eval.json")
                    evals.append(ev)
                    logger.info(
                        f"Evaluated {variant.name}/{task_id}: "
                        f"converged={ev.converged} rounds={ev.rounds} "
                        f"coder={ev.coder.score if ev.coder else '?'}/5 "
                        f"reviewer={ev.reviewer.score if ev.reviewer else '?'}/5"
                    )
                except Exception as e:
                    logger.error(f"Eval failed: {variant.name}/{task_id}: {e}")

            variant.fitness = compute_fitness(evals, island.config.weights)
            logger.info(f"Variant {variant.name} fitness: {variant.fitness:.3f}")

    def _sample_tasks(self) -> list[TaskConfig]:
        """Sample tasks for this generation (or use all if tasks_per_gen is None)."""
        if self.tasks_per_gen is None or self.tasks_per_gen >= len(self.task_pool):
            return self.task_pool
        return random.sample(self.task_pool, self.tasks_per_gen)

    def _propose_for_island(self, island: Island, tasks: list[TaskConfig]):
        """Propose variants for one island. Returns (island, elite, variants, gen_dir)."""
        island.generation += 1
        gen_dir = island.gen_dir()
        gen_dir.mkdir(parents=True, exist_ok=True)

        elite = island.best_elite()
        logger.info(
            f"Island '{island.config.name}' gen {island.generation}: "
            f"elite fitness={elite.fitness:.3f}"
        )

        variants = propose(
            elite, island.config, gen_dir,
            n_variants=self.n_variants, model=self.propose_model,
        )
        logger.info(f"Island '{island.config.name}': proposed {[v.name for v in variants]}")

        return island, elite, variants, gen_dir

    def _select_for_island(self, island, variants, gen_dir) -> dict:
        """Evaluate variants and select winner for one island."""
        self._evaluate_variants(variants, island)

        best_variant = max(variants, key=lambda v: v.fitness)
        updated = island.maybe_update_elites(best_variant)

        result = {
            "island": island.config.name,
            "generation": island.generation,
            "variants_tried": [v.name for v in variants],
            "variant_fitnesses": {v.name: v.fitness for v in variants},
            "selected": best_variant.name,
            "selected_fitness": best_variant.fitness,
            "elite_updated": updated,
            "reason": best_variant.description,
            "timestamp": time.time(),
        }
        (gen_dir / "result.json").write_text(json.dumps(result, indent=2))
        island.save_state()
        return result

    def run(self, generations: int = 5) -> list[dict]:
        """Run the full optimization loop.

        Each generation runs in three phases:
        1. PROPOSE — all islands propose variants in parallel
        2. ROLLOUT — ALL variants × tasks across ALL islands in one parallel batch
        3. EVALUATE + SELECT — score traces and update elites per island
        """
        all_results = []

        for gen in range(1, generations + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"GENERATION {gen}/{generations}")
            logger.info(f"{'='*60}\n")

            # Sample tasks once per generation (shared across islands)
            tasks = self._sample_tasks()
            logger.info(f"Tasks: {[t.task_id for t in tasks]}")

            # Phase 1: PROPOSE — all islands in parallel
            logger.info("Phase 1: Proposing variants for all islands...")
            island_plans = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.islands)) as pool:
                futures = {
                    pool.submit(self._propose_for_island, island, tasks): island
                    for island in self.islands
                }
                for f in concurrent.futures.as_completed(futures):
                    island_plans.append(f.result())

            # Phase 2: ROLLOUT — collect ALL jobs, run in one big parallel batch
            logger.info("Phase 2: Running all rollouts in parallel...")
            all_jobs = []  # (variant, task, output_dir, memory_dir)
            for island, elite, variants, gen_dir in island_plans:
                for variant in variants:
                    for task in tasks:
                        output_dir = gen_dir / variant.name / "rollouts"
                        memory_dir = gen_dir / variant.name / "memory"
                        _seed_memory(memory_dir, elite)
                        all_jobs.append((variant, task, output_dir, memory_dir))

            total_rollouts = len(all_jobs)
            logger.info(f"Launching {total_rollouts} rollouts (max_workers={self.max_workers})...")

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {}
                for variant, task, output_dir, memory_dir in all_jobs:
                    f = pool.submit(_run_single, variant, task, output_dir, memory_dir)
                    futures[f] = (variant, task.task_id)

                completed = 0
                for f in concurrent.futures.as_completed(futures):
                    variant, task_id = futures[f]
                    completed += 1
                    try:
                        _, _, trace_path = f.result()
                        if trace_path and trace_path.exists():
                            variant.traces[task_id] = trace_path
                            logger.info(f"[{completed}/{total_rollouts}] Done: {variant.name}/{task_id}")
                        else:
                            logger.warning(f"[{completed}/{total_rollouts}] No trace: {variant.name}/{task_id}")
                    except Exception as e:
                        logger.error(f"[{completed}/{total_rollouts}] Error: {variant.name}/{task_id}: {e}")

            # Phase 3: EVALUATE + SELECT — per island
            logger.info("Phase 3: Evaluating and selecting...")
            for island, elite, variants, gen_dir in island_plans:
                result = self._select_for_island(island, variants, gen_dir)
                all_results.append(result)
                logger.info(
                    f"Island '{island.config.name}' gen {island.generation}: "
                    f"selected={result['selected']} "
                    f"fitness={result['selected_fitness']:.3f} "
                    f"updated={result['elite_updated']}"
                )

            if gen % self.migrate_every == 0:
                logger.info("Migrating elites between islands...")
                migrate(self.islands)

            # Save progress
            progress = {
                "generation": gen,
                "total_generations": generations,
                "islands": {
                    island.config.name: {
                        "generation": island.generation,
                        "n_elites": len(island.elites),
                        "best_fitness": island.best_elite().fitness if island.elites else 0,
                    }
                    for island in self.islands
                },
                "timestamp": time.time(),
            }
            (self.experiments_dir / "progress.json").write_text(json.dumps(progress, indent=2))

        logger.info("\n" + "=" * 60)
        logger.info("OPTIMIZATION COMPLETE")
        for island in self.islands:
            best = island.best_elite()
            logger.info(
                f"  Island '{island.config.name}': best fitness={best.fitness:.3f} (gen {best.generation})"
            )
        logger.info("=" * 60)

        return all_results
