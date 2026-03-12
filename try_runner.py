#!/usr/bin/env python3
"""Quick test: run the coder->reviewer loop on a SWE-bench instance."""

import logging
from datasets import load_dataset
from anneal.runner import Runner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

INSTANCE_ID = "pytest-dev__pytest-10051"

# Load instance from HuggingFace
ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
instance = next(r for r in ds if r["instance_id"] == INSTANCE_ID)

# Get image name
iid = instance["instance_id"].replace("__", "_1776_")
image = f"swebench/sweb.eval.x86_64.{iid}:latest".lower()

print(f"Instance: {INSTANCE_ID}")
print(f"Image:    {image}")
print(f"Issue:    {instance['problem_statement'][:200]}...")
print()

runner = Runner(
    image=image,
    max_rounds=2,
    repo_slug="pytest-dev__pytest",
    output_dir="output/swebench_test",
    memory_dir="memory",
)

trace = runner.run(issue=instance["problem_statement"], issue_id=INSTANCE_ID)

print(f"\n{'='*60}")
print(f"Rounds:   {len(trace.rounds)}")
print(f"Approved: {trace.outcome.reviewer_approved}")
print(f"Patch:\n{trace.final_patch[:500]}")
