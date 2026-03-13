#!/usr/bin/env python3
"""Run the supervisor optimization loop.

Usage:
    python run_supervisor.py                          # default: 3 generations, all islands
    python run_supervisor.py --generations 5          # 5 generations
    python run_supervisor.py --islands first_try      # single island
    python run_supervisor.py --dry-run                # just show what would run
"""

import argparse
import logging
from pathlib import Path

from anneal.alphaevolve import (
    DEFAULT_ISLANDS,
    IslandConfig,
    Supervisor,
    TaskConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

# The image used for all tasks (same pytest repo)
PYTEST_IMAGE = "swebench/sweb.eval.x86_64.pytest-dev_1776_pytest-10051:latest"

# Task pool — all targeting the same pytest codebase
TASK_POOL = [
    TaskConfig(
        task_id="caplog-assert-logged",
        image=PYTEST_IMAGE,
        issue=Path("tasks/caplog-assert-logged.md").read_text(),
    ),
    TaskConfig(
        task_id="raises-match-type",
        image=PYTEST_IMAGE,
        issue=Path("tasks/raises-match-type.md").read_text(),
    ),
    TaskConfig(
        task_id="mark-timeout",
        image=PYTEST_IMAGE,
        issue=Path("tasks/mark-timeout.md").read_text(),
    ),
    TaskConfig(
        task_id="flaky-test-retry",
        image=PYTEST_IMAGE,
        issue=Path("tasks/flaky-test-retry.md").read_text(),
    ),
]


def main():
    parser = argparse.ArgumentParser(description="Run supervisor optimization loop")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--islands", nargs="*", help="Island names to run (default: all)")
    parser.add_argument("--n-variants", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=20)
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--tasks", nargs="*", help="Task IDs to use (default: all)")
    parser.add_argument("--tasks-per-gen", type=int, default=None, help="Sample N tasks per generation (default: all)")
    parser.add_argument("--seed-learnings", default="seed_learnings/pytest-dev__pytest", help="Path to seed learnings dir")
    parser.add_argument("--dry-run", action="store_true", help="Show config without running")
    args = parser.parse_args()

    # Filter islands
    if args.islands:
        island_configs = [ic for ic in DEFAULT_ISLANDS if ic.name in args.islands]
        if not island_configs:
            print(f"No matching islands. Available: {[ic.name for ic in DEFAULT_ISLANDS]}")
            return
    else:
        island_configs = DEFAULT_ISLANDS

    # Filter tasks
    if args.tasks:
        tasks = [t for t in TASK_POOL if t.task_id in args.tasks]
        if not tasks:
            print(f"No matching tasks. Available: {[t.task_id for t in TASK_POOL]}")
            return
    else:
        tasks = TASK_POOL

    print(f"Islands:     {[ic.name for ic in island_configs]}")
    print(f"Tasks:       {[t.task_id for t in tasks]}")
    print(f"Generations: {args.generations}")
    print(f"Variants:    {args.n_variants} per generation per island")
    print(f"Workers:     {args.max_workers} parallel rollouts")
    print(f"Output:      {args.experiments_dir}/")
    print()

    tasks_per_gen = args.tasks_per_gen or len(tasks)
    print(f"Tasks/gen:   {tasks_per_gen} (sampled from {len(tasks)})")

    rollouts_per_gen = len(island_configs) * args.n_variants * tasks_per_gen
    total_rollouts = rollouts_per_gen * args.generations
    est_cost = total_rollouts * 1.5  # ~$1.50 per rollout
    print(f"Rollouts per generation: {rollouts_per_gen}")
    print(f"Total rollouts: {total_rollouts}")
    print(f"Estimated cost: ~${est_cost:.0f}")
    print()

    if args.dry_run:
        print("Dry run — exiting.")
        return

    supervisor = Supervisor(
        task_pool=tasks,
        experiments_dir=args.experiments_dir,
        seed_learnings_dir=args.seed_learnings,
        islands=island_configs,
        n_variants=args.n_variants,
        tasks_per_gen=args.tasks_per_gen,
        max_workers=args.max_workers,
    )

    results = supervisor.run(generations=args.generations)

    print(f"\nCompleted {len(results)} generation runs.")
    for r in results:
        print(
            f"  {r['island']} gen {r['generation']}: "
            f"selected={r['selected']} "
            f"fitness={r['selected_fitness']:.3f}"
        )


if __name__ == "__main__":
    main()
