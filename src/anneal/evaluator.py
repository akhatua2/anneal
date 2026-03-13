"""Evaluator: scores a completed trace using quantitative metrics + LLM judge."""

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from litellm import completion

logger = logging.getLogger("anneal.evaluator")


# --- Data types ---


@dataclass
class RoundMetrics:
    """Quantitative metrics for one round (no LLM needed)."""

    round_num: int
    coder_cost: float
    coder_api_calls: int
    coder_steps: int  # assistant turns
    reviewer_cost: float
    reviewer_api_calls: int
    reviewer_steps: int
    approved: bool
    has_patch: bool


@dataclass
class AgentScore:
    """LLM judge score for one agent across all rounds."""

    score: int  # 1-5
    analysis: str
    improvements: list[str]


@dataclass
class EvalResult:
    """Full evaluation of a trace."""

    # Quantitative
    converged: bool
    rounds: int
    total_cost: float
    total_steps: int
    round_metrics: list[RoundMetrics]

    # Qualitative (filled by LLM judge)
    coder: AgentScore | None = None
    reviewer: AgentScore | None = None


# --- Quantitative extraction ---


def _load_trajectory_stats(traj_path: str) -> dict:
    """Extract stats from a trajectory JSON file."""
    traj = json.loads(Path(traj_path).read_text())
    stats = traj["info"]["model_stats"]
    messages = traj.get("messages", [])
    assistant_turns = sum(1 for m in messages if m.get("role") == "assistant")
    return {
        "cost": stats.get("instance_cost", 0.0),
        "api_calls": stats.get("api_calls", 0),
        "steps": assistant_turns,
    }


def extract_metrics(trace: dict) -> EvalResult:
    """Extract quantitative metrics from a trace dict (no LLM calls)."""
    round_metrics = []
    total_cost = 0.0
    total_steps = 0

    for i, (pr, review) in enumerate(trace["rounds"], 1):
        coder_stats = _load_trajectory_stats(pr["coder_trajectory"])
        reviewer_stats = (
            _load_trajectory_stats(review["reviewer_trajectory"])
            if review.get("reviewer_trajectory")
            else {"cost": 0.0, "api_calls": 0, "steps": 0}
        )

        rm = RoundMetrics(
            round_num=i,
            coder_cost=coder_stats["cost"],
            coder_api_calls=coder_stats["api_calls"],
            coder_steps=coder_stats["steps"],
            reviewer_cost=reviewer_stats["cost"],
            reviewer_api_calls=reviewer_stats["api_calls"],
            reviewer_steps=reviewer_stats["steps"],
            approved=review["verdict"].get("approved", False),
            has_patch=bool(pr.get("patch", "").strip()),
        )
        round_metrics.append(rm)
        total_cost += rm.coder_cost + rm.reviewer_cost
        total_steps += rm.coder_steps + rm.reviewer_steps

    return EvalResult(
        converged=trace["outcome"].get("reviewer_approved", False),
        rounds=len(trace["rounds"]),
        total_cost=total_cost,
        total_steps=total_steps,
        round_metrics=round_metrics,
    )


# --- LLM Judge ---

JUDGE_SYSTEM = """\
You are evaluating the performance of two AI agents — a coder and a reviewer — \
that work together to fix software issues.

You will receive the full interaction trace: the issue, each round's patch, \
review, and key trajectory details (what commands each agent ran).

Score each agent separately. Be specific about what they did well and poorly. \
Your improvement suggestions should be concrete prompt-level changes \
(things we could add to their system prompt to make them better).

Respond in this exact JSON format:
{
  "coder_score": <1-5>,
  "coder_analysis": "<2-4 sentences>",
  "coder_improvements": ["<specific prompt change>", ...],
  "reviewer_score": <1-5>,
  "reviewer_analysis": "<2-4 sentences>",
  "reviewer_improvements": ["<specific prompt change>", ...]
}

Scoring guide:
5 = Excellent — minimal wasted effort, correct fix, good tests, precise review
4 = Good — mostly efficient, minor issues
3 = Acceptable — got the job done but with notable inefficiencies
2 = Poor — significant wasted effort, missed obvious things
1 = Failed — didn't produce useful output or gave wrong feedback\
"""


def _build_judge_prompt(trace: dict) -> str:
    """Build the user prompt for the LLM judge from a trace."""
    parts = []
    parts.append("## Issue")
    parts.append(trace["issue"][:3000])  # truncate very long issues
    parts.append("")

    for i, (pr, review) in enumerate(trace["rounds"], 1):
        parts.append(f"## Round {i}")

        # Coder summary
        parts.append(f"### Coder (Round {i})")
        coder_traj = json.loads(Path(pr["coder_trajectory"]).read_text())
        coder_stats = coder_traj["info"]["model_stats"]
        parts.append(
            f"Cost: ${coder_stats['instance_cost']:.2f}, "
            f"API calls: {coder_stats['api_calls']}, "
            f"Exit: {coder_traj['info'].get('exit_status', 'unknown')}"
        )

        # Extract what the coder actually did (assistant messages, truncated)
        coder_msgs = coder_traj.get("messages", [])
        commands_run = []
        for msg in coder_msgs:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # tool_use blocks
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            cmd = block.get("input", {}).get("command", "")
                            if cmd:
                                commands_run.append(cmd[:200])
                elif isinstance(content, str) and content.strip():
                    # reasoning text — include first 200 chars
                    commands_run.append(f"[reasoning] {content[:200]}")

        # Show first 15 and last 5 actions to keep prompt manageable
        if len(commands_run) > 20:
            shown = commands_run[:15] + ["... ({} actions omitted) ...".format(len(commands_run) - 20)] + commands_run[-5:]
        else:
            shown = commands_run
        for cmd in shown:
            parts.append(f"  - {cmd}")

        # Patch
        patch = pr.get("patch", "")
        if patch:
            parts.append(f"\n### Patch (Round {i})")
            parts.append(f"```diff\n{patch[:3000]}\n```")

        # Review
        parts.append(f"\n### Review (Round {i})")
        verdict = review.get("verdict", {})
        parts.append(f"Approved: {verdict.get('approved', False)}")
        raw_review = verdict.get("raw", "")
        if raw_review:
            parts.append(raw_review[:2000])

        if review.get("reviewer_trajectory"):
            reviewer_traj = json.loads(
                Path(review["reviewer_trajectory"]).read_text()
            )
            reviewer_stats = reviewer_traj["info"]["model_stats"]
            parts.append(
                f"\nReviewer stats — Cost: ${reviewer_stats['instance_cost']:.2f}, "
                f"API calls: {reviewer_stats['api_calls']}"
            )

        parts.append("")

    parts.append("## Outcome")
    parts.append(f"Converged: {trace['outcome'].get('reviewer_approved', False)}")
    parts.append(f"Total rounds: {len(trace['rounds'])}")

    return "\n".join(parts)


def judge(trace: dict, *, model: str = "anthropic/claude-haiku-4-5") -> tuple[AgentScore, AgentScore]:
    """Run the LLM judge on a trace. Returns coder and reviewer scores."""
    prompt = _build_judge_prompt(trace)
    logger.info(f"Judging trace ({len(prompt)} chars)...")

    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    content = response.choices[0].message.content
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", content, re.DOTALL)
    cleaned = fence_match.group(1).strip() if fence_match else content.strip()
    result = json.loads(cleaned)

    coder = AgentScore(
        score=result["coder_score"],
        analysis=result["coder_analysis"],
        improvements=result.get("coder_improvements", []),
    )
    reviewer = AgentScore(
        score=result["reviewer_score"],
        analysis=result["reviewer_analysis"],
        improvements=result.get("reviewer_improvements", []),
    )
    return coder, reviewer


# --- Main entry point ---


def evaluate(
    trace_path: str | Path,
    *,
    run_judge: bool = True,
    judge_model: str = "anthropic/claude-haiku-4-5",
) -> EvalResult:
    """Full evaluation: quantitative metrics + optional LLM judge.

    Args:
        trace_path: Path to trace.json
        run_judge: Whether to run the LLM judge (costs ~$0.05-0.10)
        judge_model: Model to use for judging

    Returns:
        EvalResult with all metrics filled in.
    """
    trace = json.loads(Path(trace_path).read_text())
    result = extract_metrics(trace)

    if run_judge:
        coder, reviewer = judge(trace, model=judge_model)
        result.coder = coder
        result.reviewer = reviewer

    return result


def save_eval(result: EvalResult, path: str | Path):
    """Save an EvalResult to disk as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2, default=str))
