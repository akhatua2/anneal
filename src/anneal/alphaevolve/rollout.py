"""Parallel rollout execution — runs variants × tasks on Modal."""

import concurrent.futures
import logging
import shutil
from pathlib import Path

from anneal.alphaevolve.types import Elite, TaskConfig, Variant
from anneal.runner import Runner

logger = logging.getLogger("anneal.alphaevolve.rollout")


def _find_trace(output_dir: Path, task_id: str) -> Path | None:
    """Find the trace.json for a task, accounting for the timestamp subdirectory.

    Runner saves to output_dir/task_id/TIMESTAMP/trace.json.
    """
    task_dir = output_dir / task_id
    if not task_dir.exists():
        return None
    ts_dirs = sorted(
        [d for d in task_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    for ts_dir in ts_dirs:
        trace_path = ts_dir / "trace.json"
        if trace_path.exists():
            return trace_path
    return None


def _run_single(
    variant: Variant,
    task: TaskConfig,
    output_dir: Path,
    memory_dir: Path,
) -> tuple[str, str, Path | None]:
    """Run one task with one variant's prompts."""
    try:
        runner = Runner(
            image=task.image,
            coder_config=str(variant.prompts.coder_yaml),
            reviewer_config=str(variant.prompts.reviewer_yaml),
            output_dir=str(output_dir),
            memory_dir=str(memory_dir),
        )
        runner.run(issue=task.issue, issue_id=task.task_id)
        trace_path = _find_trace(output_dir, task.task_id)
        return variant.name, task.task_id, trace_path
    except Exception as e:
        logger.error(f"Rollout failed: variant={variant.name} task={task.task_id}: {e}")
        return variant.name, task.task_id, None


def _seed_memory(memory_dir: Path, parent_elite: Elite | None):
    """Seed a variant's memory dir with the parent elite's learnings.

    If the parent elite has learnings, copy them so the agents start warm.
    If not (baseline), agents start cold.
    """
    memory_dir.mkdir(parents=True, exist_ok=True)
    if parent_elite and parent_elite.learnings_dir and parent_elite.learnings_dir.exists():
        for f in parent_elite.learnings_dir.iterdir():
            if f.name.endswith("_learnings.txt"):
                shutil.copy2(f, memory_dir / f.name)
        logger.info(f"Seeded memory from parent elite learnings")


def run_rollouts(
    variants: list[Variant],
    tasks: list[TaskConfig],
    gen_dir: Path,
    *,
    parent_elite: Elite | None = None,
    max_workers: int = 12,
) -> None:
    """Run all variants × tasks in parallel. Updates variant.traces in place.

    Each variant gets isolated output and memory directories so rollouts
    don't interfere with each other. All variants start with the same
    parent elite's learnings for fair comparison.
    """
    jobs = []
    for variant in variants:
        for task in tasks:
            output_dir = gen_dir / variant.name / "rollouts"
            memory_dir = gen_dir / variant.name / "memory"
            _seed_memory(memory_dir, parent_elite)
            jobs.append((variant, task, output_dir, memory_dir))

    logger.info(
        f"Running {len(jobs)} rollouts "
        f"({len(variants)} variants x {len(tasks)} tasks, "
        f"max_workers={max_workers})..."
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for variant, task, output_dir, memory_dir in jobs:
            f = pool.submit(_run_single, variant, task, output_dir, memory_dir)
            futures[f] = (variant.name, task.task_id)

        for f in concurrent.futures.as_completed(futures):
            variant_name, task_id = futures[f]
            try:
                _, _, trace_path = f.result()
                if trace_path and trace_path.exists():
                    for v in variants:
                        if v.name == variant_name:
                            v.traces[task_id] = trace_path
                            break
                    logger.info(f"Completed: {variant_name}/{task_id}")
                else:
                    logger.warning(f"No trace: {variant_name}/{task_id}")
            except Exception as e:
                logger.error(f"Rollout error: {variant_name}/{task_id}: {e}")
