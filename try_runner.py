#!/usr/bin/env python3
"""Run the coder->reviewer loop on a task."""

import logging
import sys
from pathlib import Path

from anneal.runner import Runner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

TASKS_DIR = Path("tasks")
IMAGE = "swebench/sweb.eval.x86_64.pytest-dev_1776_pytest-10051:latest"

# Pick task from CLI arg or default
task_name = sys.argv[1] if len(sys.argv) > 1 else "caplog-assert-logged"
task_file = TASKS_DIR / f"{task_name}.md"

if not task_file.exists():
    available = [f.stem for f in TASKS_DIR.glob("*.md")]
    print(f"Task '{task_name}' not found. Available: {available}")
    sys.exit(1)

issue = task_file.read_text()

print(f"Task:  {task_name}")
print(f"Image: {IMAGE}")
print(f"Issue: {issue[:200]}...")
print()

runner = Runner(
    image=IMAGE,
    max_rounds=10,
    repo_slug="pytest-dev__pytest",
    output_dir="output/features",
    memory_dir="memory",
)

trace = runner.run(issue=issue, issue_id=task_name)

print(f"\n{'='*60}")
print(f"Rounds:   {len(trace.rounds)}")
print(f"Approved: {trace.outcome.reviewer_approved}")
print(f"Patch:\n{trace.final_patch[:500]}")
