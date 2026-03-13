"""Island management — maintains elite pool and handles selection."""

import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path

from anneal.alphaevolve.types import Elite, IslandConfig, PromptPair, Variant

logger = logging.getLogger("anneal.alphaevolve.island")


class Island:
    """One island in the evolutionary system.

    Each island optimizes for a different objective (e.g. first-try success,
    efficiency, feedback quality) and maintains its own pool of elite prompt
    configurations.
    """

    def __init__(self, config: IslandConfig, base_dir: Path, max_elites: int = 3):
        self.config = config
        self.base_dir = base_dir
        self.max_elites = max_elites
        self.elites: list[Elite] = []
        self.generation = 0

        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "elites").mkdir(exist_ok=True)
        (self.base_dir / "generations").mkdir(exist_ok=True)

        config_path = self.base_dir / "config.json"
        if not config_path.exists():
            config_path.write_text(
                json.dumps(
                    {"name": config.name, "objective": config.objective, "weights": config.weights},
                    indent=2,
                )
            )

        self._load_state()

    def _load_state(self):
        """Load elites and generation count from disk."""
        state_path = self.base_dir / "state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            self.generation = state.get("generation", 0)

        elites_dir = self.base_dir / "elites"
        for elite_dir in sorted(elites_dir.iterdir()):
            if not elite_dir.is_dir():
                continue
            meta_path = elite_dir / "meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            learnings_dir = elite_dir / "learnings"
            self.elites.append(
                Elite(
                    prompts=PromptPair(
                        coder_yaml=elite_dir / "base_coder.yaml",
                        reviewer_yaml=elite_dir / "base_reviewer.yaml",
                    ),
                    fitness=meta.get("fitness", 0.0),
                    eval_results=meta.get("eval_results", []),
                    generation=meta.get("generation", 0),
                    learnings_dir=learnings_dir if learnings_dir.exists() else None,
                )
            )

    def save_state(self):
        """Save current state to disk."""
        state = {"generation": self.generation, "n_elites": len(self.elites)}
        (self.base_dir / "state.json").write_text(json.dumps(state, indent=2))

    def _save_elite(self, elite: Elite, index: int):
        """Save an elite to disk, including its learnings."""
        elite_dir = self.base_dir / "elites" / f"elite_{index}"
        elite.prompts.copy_to(elite_dir)

        # Copy learnings if available (skip if already in place)
        if elite.learnings_dir and elite.learnings_dir.exists():
            dest_learnings = elite_dir / "learnings"
            if elite.learnings_dir.resolve() != dest_learnings.resolve():
                dest_learnings.mkdir(exist_ok=True)
                for f in elite.learnings_dir.iterdir():
                    if f.name.endswith("_learnings.txt"):
                        shutil.copy2(f, dest_learnings / f.name)
            elite.learnings_dir = dest_learnings

        meta = {
            "fitness": elite.fitness,
            "generation": elite.generation,
            "eval_results": elite.eval_results[-20:],
        }
        (elite_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    def seed_elite(self, prompts: PromptPair):
        """Add initial elite (baseline prompts). No-op if already seeded.

        Picks up learnings if they exist alongside the prompts
        (e.g. copied there by the supervisor from seed_learnings_dir).
        """
        if self.elites:
            return
        # Check if learnings were placed alongside the prompts
        learnings_dir = prompts.coder_yaml.parent / "learnings"
        elite = Elite(
            prompts=prompts, fitness=0.0, generation=0,
            learnings_dir=learnings_dir if learnings_dir.exists() else None,
        )
        self.elites.append(elite)
        self._save_elite(elite, 0)
        self.save_state()
        logger.info(f"Island '{self.config.name}' seeded with baseline elite")

    def best_elite(self) -> Elite:
        """Return the highest-fitness elite."""
        return max(self.elites, key=lambda e: e.fitness)

    def _find_variant_learnings(self, variant: Variant) -> Path | None:
        """Find the memory dir from a variant's rollouts."""
        # Variant memory lives at gen_dir/variant_name/memory/
        # The variant's prompts are at gen_dir/variant_name/base_coder.yaml
        variant_dir = variant.prompts.coder_yaml.parent
        memory_dir = variant_dir / "memory"
        return memory_dir if memory_dir.exists() else None

    def maybe_update_elites(self, variant: Variant) -> bool:
        """If variant beats the worst elite (or there's room), add it.

        Returns True if the elite pool was updated.
        """
        learnings_dir = self._find_variant_learnings(variant)

        if len(self.elites) < self.max_elites:
            new_elite = Elite(
                prompts=variant.prompts,
                fitness=variant.fitness,
                eval_results=[asdict(ev) for ev in variant.evals.values()] if variant.evals else [],
                generation=self.generation,
                learnings_dir=learnings_dir,
            )
            self.elites.append(new_elite)
            self._save_elite(new_elite, len(self.elites) - 1)
            self.save_state()
            logger.info(
                f"Island '{self.config.name}': added elite "
                f"(fitness={variant.fitness:.3f}, {len(self.elites)}/{self.max_elites})"
            )
            return True

        worst = min(self.elites, key=lambda e: e.fitness)
        if variant.fitness > worst.fitness:
            idx = self.elites.index(worst)
            new_elite = Elite(
                prompts=variant.prompts,
                fitness=variant.fitness,
                eval_results=[asdict(ev) for ev in variant.evals.values()] if variant.evals else [],
                generation=self.generation,
                learnings_dir=learnings_dir,
            )
            self.elites[idx] = new_elite
            self._save_elite(new_elite, idx)
            self.save_state()
            logger.info(
                f"Island '{self.config.name}': replaced elite {idx} "
                f"(old={worst.fitness:.3f} -> new={variant.fitness:.3f})"
            )
            return True

        logger.info(
            f"Island '{self.config.name}': variant {variant.name} "
            f"(fitness={variant.fitness:.3f}) did not beat worst elite "
            f"(fitness={worst.fitness:.3f})"
        )
        return False

    def gen_dir(self) -> Path:
        return self.base_dir / "generations" / f"gen_{self.generation:03d}"


def migrate(islands: list[Island]):
    """Share the best elite from each island with all other islands."""
    for source in islands:
        if not source.elites:
            continue
        best = source.best_elite()
        for target in islands:
            if target is source:
                continue
            already_has = any(
                abs(e.fitness - best.fitness) < 0.01 for e in target.elites
            )
            if not already_has:
                migrated_dir = target.base_dir / "elites" / f"migrated_from_{source.config.name}"
                copied = best.prompts.copy_to(migrated_dir)
                target.maybe_update_elites(
                    Variant(
                        name=f"migrated_from_{source.config.name}",
                        description=f"Migrated from {source.config.name} island",
                        prompts=copied,
                        fitness=best.fitness,
                    )
                )
                logger.info(
                    f"Migrated elite from '{source.config.name}' to '{target.config.name}' "
                    f"(fitness={best.fitness:.3f})"
                )
