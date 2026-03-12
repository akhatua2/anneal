"""Run mini-swe-agent on a dummy task with claude-haiku."""
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

from minisweagent import package_dir
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

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

config = yaml.safe_load((package_dir / "config" / "default.yaml").read_text())

os.makedirs("/tmp/anneal_dummy", exist_ok=True)
Path("/tmp/anneal_dummy/numbers.txt").write_text("3\n1\n4\n1\n5\n9\n2\n6\n")

agent = DefaultAgent(
    LitellmModel(model_name="claude-haiku-4-5-20251001", **config["model"]),
    LocalEnvironment(**config["environment"]),
    **config["agent"],
)

agent.run(TASK)
