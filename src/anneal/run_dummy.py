"""Run mini-swe-agent on a dummy task with claude-haiku."""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

logging.basicConfig(level=logging.INFO)

TASK = """
There is a file called /tmp/anneal_dummy/numbers.txt with the following content:
3
1
4
1
5
9
2
6

Write a Python script at /tmp/anneal_dummy/sort_numbers.py that reads numbers.txt,
sorts the numbers in ascending order, and prints them one per line.
Then run the script to verify it works.
"""

os.makedirs("/tmp/anneal_dummy", exist_ok=True)
Path("/tmp/anneal_dummy/numbers.txt").write_text("3\n1\n4\n1\n5\n9\n2\n6\n")

agent = DefaultAgent(
    LitellmModel(model_name="claude-haiku-4-5-20251001"),
    LocalEnvironment(),
    system_template="""You are a helpful assistant that can interact with a computer.

Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).
Include a THOUGHT section before your command where you explain your reasoning process.

```mswea_bash_command
your_command_here
```

When you are done, run: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
""",
    instance_template="Please solve this task: {{task}}",
    step_limit=10,
    cost_limit=1.0,
)

result = agent.run(TASK)
print(f"\nResult: {result}")
