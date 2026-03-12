"""Core data types for anneal traces."""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class PRAttempt:
    """The coder's side of one round."""
    patch: str
    coder_trajectory: str = ""  # path to coder_trajectory.json


@dataclass
class Review:
    """The reviewer's side of one round."""
    verdict: dict = field(default_factory=dict)
    reviewer_trajectory: str = ""  # path to reviewer_trajectory.json


@dataclass
class Outcome:
    """Ground truth signals — evaluated AFTER the loop, separate from agents."""
    tests_passed: bool | None = None
    reviewer_approved: bool = False
    rounds_to_merge: int = 0


@dataclass
class Trace:
    """Full interaction trace for one issue.

    Trace = {
      issue,
      [(pr_attempt_1, review_1), (pr_attempt_2, review_2), ...],
      outcome: {tests_passed, reviewer_approved, rounds_to_merge}
    }

    Agent trajectories are saved as separate JSON files on disk.
    The trace itself only stores paths to them.
    """
    issue: str
    rounds: list[tuple[PRAttempt, Review]] = field(default_factory=list)
    outcome: Outcome = field(default_factory=Outcome)
    final_patch: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)
