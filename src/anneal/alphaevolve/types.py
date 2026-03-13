"""Data types for the AlphaEvolve-style prompt optimization."""

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from anneal.evaluator import EvalResult


@dataclass
class TaskConfig:
    """One task in the task pool."""

    task_id: str
    image: str
    issue: str


@dataclass
class PromptPair:
    """A coder + reviewer prompt configuration (the 'genome')."""

    coder_yaml: Path
    reviewer_yaml: Path

    def copy_to(self, dest_dir: Path) -> "PromptPair":
        """Copy both YAMLs to dest_dir. Returns new PromptPair pointing there."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_coder = dest_dir / "base_coder.yaml"
        dest_reviewer = dest_dir / "base_reviewer.yaml"
        # Avoid SameFileError
        if self.coder_yaml.resolve() != dest_coder.resolve():
            shutil.copy2(self.coder_yaml, dest_coder)
        if self.reviewer_yaml.resolve() != dest_reviewer.resolve():
            shutil.copy2(self.reviewer_yaml, dest_reviewer)
        return PromptPair(coder_yaml=dest_coder, reviewer_yaml=dest_reviewer)


@dataclass
class Elite:
    """A prompt pair with its fitness score.

    Also carries repo learnings (coder_learnings.txt, reviewer_learnings.txt)
    that accumulated during the rollouts that produced this elite. Child
    variants inherit these as a warm start.
    """

    prompts: PromptPair
    fitness: float = 0.0
    eval_results: list[dict] = field(default_factory=list)
    generation: int = 0
    learnings_dir: Path | None = None  # dir containing {coder,reviewer}_learnings.txt


@dataclass
class IslandConfig:
    """Configuration for one island."""

    name: str
    objective: str
    weights: dict[str, float]
    mutable: str = "both"  # "coder", "reviewer", or "both"


@dataclass
class Variant:
    """A proposed prompt mutation with its rollout results."""

    name: str
    description: str
    prompts: PromptPair
    traces: dict[str, Path] = field(default_factory=dict)
    evals: dict[str, EvalResult] = field(default_factory=dict)
    fitness: float = 0.0
