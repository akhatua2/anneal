"""LLM-based proposal of prompt mutations."""

import json
import logging
import re
import shutil
from pathlib import Path

from litellm import completion

from anneal.alphaevolve.types import Elite, IslandConfig, PromptPair, Variant

logger = logging.getLogger("anneal.alphaevolve.proposer")


PROPOSE_SYSTEM_BOTH = """\
You are a prompt optimization expert. You modify system prompts for AI coding \
agents to improve their performance.

You will receive:
1. The current best prompts (coder + reviewer YAML configs)
2. The island's optimization objective
3. Recent evaluation results showing how the agents performed
4. Previously tried mutations and their outcomes

Propose exactly {n_variants} prompt variants. For each variant:
- Give it a short name (snake_case, e.g. "add_grep_first")
- Describe the change in one sentence
- Output the COMPLETE modified YAML for both coder and reviewer

Changes should be targeted and specific. Don't rewrite everything. \
Focus on the island's objective.

CRITICAL CONSTRAINTS:
- NEVER lower the reviewer's quality bar. Make the coder better, not the reviewer weaker.
- NEVER remove safety checks, validation steps, or test requirements.
- NEVER change the model_name or cost_limit fields.
- Improve agents by making them smarter, not by gaming the evaluation.

Respond in this exact JSON format:
{{
  "variants": [
    {{
      "name": "variant_name",
      "description": "what this changes and why",
      "coder_yaml": "<full YAML string>",
      "reviewer_yaml": "<full YAML string>"
    }},
    ...
  ]
}}\
"""

PROPOSE_SYSTEM_SINGLE = """\
You are a prompt optimization expert. You modify the system prompt for an AI \
{role} agent to improve its performance.

You will receive:
1. The current best {role} prompt (YAML config)
2. The island's optimization objective
3. Recent evaluation results showing how the agent performed
4. Previously tried mutations and their outcomes

Propose exactly {n_variants} prompt variants. For each variant:
- Give it a short name (snake_case, e.g. "add_grep_first")
- Describe the change in one sentence
- Output the COMPLETE modified YAML

Changes should be targeted and specific. Don't rewrite everything. \
Focus on the island's objective.

CRITICAL CONSTRAINTS:
- NEVER remove safety checks, validation steps, or test requirements.
- NEVER change the model_name or cost_limit fields.
- Improve the agent by making it smarter, not by gaming the evaluation.

Respond in this exact JSON format:
{{
  "variants": [
    {{
      "name": "variant_name",
      "description": "what this changes and why",
      "{role}_yaml": "<full YAML string>"
    }},
    ...
  ]
}}\
"""


def _format_eval_summary(evals: list[dict]) -> str:
    """Format eval results for the supervisor prompt."""
    if not evals:
        return "No evaluation results yet (baseline)."
    parts = []
    for ev in evals:
        parts.append(
            f"- Task: converged={ev.get('converged')}, rounds={ev.get('rounds')}, "
            f"cost=${ev.get('total_cost', 0):.2f}, steps={ev.get('total_steps', 0)}"
        )
        if ev.get("coder"):
            c = ev["coder"]
            parts.append(f"  Coder {c['score']}/5: {c['analysis'][:150]}")
            for imp in c.get("improvements", []):
                parts.append(f"    -> {imp}")
        if ev.get("reviewer"):
            r = ev["reviewer"]
            parts.append(f"  Reviewer {r['score']}/5: {r['analysis'][:150]}")
            for imp in r.get("improvements", []):
                parts.append(f"    -> {imp}")
    return "\n".join(parts)


def _format_past_mutations(gen_dir: Path) -> str:
    """Summarize previously tried mutations from the experiment directory."""
    parts = []
    if not gen_dir.parent.exists():
        return "No previous mutations."

    for gdir in sorted(gen_dir.parent.iterdir()):
        if not gdir.is_dir() or not gdir.name.startswith("gen_"):
            continue
        result_path = gdir / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            parts.append(
                f"- Gen {gdir.name}: tried {result.get('variants_tried', '?')}, "
                f"selected={result.get('selected', '?')}, "
                f"reason={result.get('reason', '?')[:100]}"
            )

    return "\n".join(parts) if parts else "No previous mutations."


def _call_proposer(messages: list[dict], model: str) -> dict:
    """Call the LLM and parse JSON response, with retries."""
    result = None
    for attempt in range(3):
        response = completion(model=model, messages=messages, temperature=0.7)
        content = response.choices[0].message.content
        if not content or not content.strip():
            logger.warning(f"Proposer returned empty content (attempt {attempt + 1}/3)")
            continue
        fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", content, re.DOTALL)
        cleaned = fence_match.group(1).strip() if fence_match else content.strip()
        try:
            result = json.loads(cleaned)
            break
        except json.JSONDecodeError as e:
            logger.warning(f"Proposer returned invalid JSON (attempt {attempt + 1}/3): {e}")
            continue
    if result is None:
        raise RuntimeError("Proposer failed to return valid JSON after 3 attempts")
    return result


def propose(
    elite: Elite,
    island: IslandConfig,
    gen_dir: Path,
    *,
    n_variants: int = 2,
    model: str = "anthropic/claude-sonnet-4-6",
) -> list[Variant]:
    """Ask the LLM to propose prompt mutations.

    Respects island.mutable: only sends the mutable agent's YAML to the LLM.
    The frozen agent's YAML is copied as-is.
    """
    mutable = island.mutable  # "coder", "reviewer", or "both"
    coder_yaml = elite.prompts.coder_yaml.read_text()
    reviewer_yaml = elite.prompts.reviewer_yaml.read_text()

    # Build prompt based on what's mutable
    if mutable == "both":
        system_prompt = PROPOSE_SYSTEM_BOTH.format(n_variants=n_variants)
        prompts_section = f"""### Coder (base_coder.yaml)
```yaml
{coder_yaml}
```

### Reviewer (base_reviewer.yaml)
```yaml
{reviewer_yaml}
```"""
    else:
        role = mutable  # "coder" or "reviewer"
        yaml_content = coder_yaml if role == "coder" else reviewer_yaml
        system_prompt = PROPOSE_SYSTEM_SINGLE.format(n_variants=n_variants, role=role)
        prompts_section = f"""### {role.title()} (base_{role}.yaml)
```yaml
{yaml_content}
```"""

    user_prompt = f"""## Island Objective
{island.objective}

## Fitness Weights
{json.dumps(island.weights, indent=2)}

## Current Best Prompt{"s" if mutable == "both" else ""}

{prompts_section}

## Recent Evaluation Results
{_format_eval_summary(elite.eval_results[-10:])}

## Previously Tried Mutations
{_format_past_mutations(gen_dir)}

Propose exactly {n_variants} variants."""

    logger.info(f"Proposing mutations (mutable={mutable})...")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    result = _call_proposer(messages, model)

    # Build variants, copying the frozen agent's YAML as-is
    variants = []
    for v in result["variants"]:
        variant_dir = gen_dir / v["name"]
        variant_dir.mkdir(parents=True, exist_ok=True)

        coder_path = variant_dir / "base_coder.yaml"
        reviewer_path = variant_dir / "base_reviewer.yaml"

        if mutable == "both":
            coder_path.write_text(v["coder_yaml"])
            reviewer_path.write_text(v["reviewer_yaml"])
        elif mutable == "coder":
            coder_path.write_text(v["coder_yaml"])
            shutil.copy2(elite.prompts.reviewer_yaml, reviewer_path)
        else:  # reviewer
            shutil.copy2(elite.prompts.coder_yaml, coder_path)
            reviewer_path.write_text(v["reviewer_yaml"])

        variants.append(
            Variant(
                name=v["name"],
                description=v["description"],
                prompts=PromptPair(coder_yaml=coder_path, reviewer_yaml=reviewer_path),
            )
        )

    # Save the proposal
    proposal = {
        "island": island.name,
        "mutable": mutable,
        "elite_fitness": elite.fitness,
        "n_variants": len(variants),
        "variants": [{"name": v.name, "description": v.description} for v in variants],
    }
    (gen_dir / "proposal.json").write_text(json.dumps(proposal, indent=2))

    return variants
