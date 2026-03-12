"""Shared utilities for anneal."""

import json
from pathlib import Path

from anneal.types import Trace


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_text(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def save_trace(trace: Trace, output_dir: Path):
    """Save trace metadata + final patch to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(trace.to_dict(), output_dir / "trace.json")
    save_text(trace.final_patch, output_dir / "final.patch")


def load_trace(output_dir: Path) -> dict:
    """Load a trace from disk (as raw dict)."""
    return load_json(output_dir / "trace.json")
