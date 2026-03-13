"""Fitness computation for prompt variants."""

from anneal.evaluator import EvalResult


def compute_fitness(evals: list[EvalResult], weights: dict[str, float]) -> float:
    """Compute aggregate fitness from eval results using island weights."""
    if not evals:
        return 0.0

    total = 0.0
    for ev in evals:
        score = 0.0
        score += weights.get("converged", 0.0) * (1.0 if ev.converged else 0.0)
        score += weights.get("rounds", 0.0) * ev.rounds
        score += weights.get("cost", 0.0) * ev.total_cost
        score += weights.get("steps", 0.0) * ev.total_steps
        if ev.coder:
            score += weights.get("coder_score", 0.0) * ev.coder.score
        if ev.reviewer:
            score += weights.get("reviewer_score", 0.0) * ev.reviewer.score
        total += score

    return total / len(evals)
