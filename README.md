# Anneal

An AlphaEvolve-inspired system that evolves coding and reviewing agent prompts through competitive selection.

## How it works

**Inner loop (Runner):** A coder agent fixes issues or builds features, a reviewer agent reviews the patch, and they iterate until the reviewer accepts or max rounds are hit. Both agents maintain persistent memory files (repo learnings + task notes) that carry knowledge across rounds.

**Outer loop (Supervisor):** Evolves the coder and reviewer prompts using island-based evolution:

1. **Propose** — An LLM mutates the current best prompt configs based on evaluation feedback
2. **Rollout** — All variants run against sampled tasks in parallel on Modal
3. **Evaluate** — An LLM judge scores each trace (correctness, cost, quality)
4. **Select** — Best variant enters the island's elite pool if it beats the worst elite
5. **Migrate** — Periodically share best elites across islands

Five islands optimize for different objectives: first-try success, cost efficiency, feedback quality, correctness, and adversarial reviewing.

## Architecture

```
src/anneal/
├── runner.py                  # Inner loop: coder <-> reviewer iteration
├── agents/
│   ├── factory.py             # Creates coder/reviewer agents (SWE-ReX Modal)
│   └── review_parser.py       # Parses reviewer verdict
├── configs/
│   ├── base_coder.yaml        # Coder agent prompt + config
│   └── base_reviewer.yaml     # Reviewer agent prompt + config
├── alphaevolve/
│   ├── supervisor.py          # Outer loop orchestration
│   ├── proposer.py            # LLM-based prompt mutation
│   ├── rollout.py             # Parallel rollout execution
│   ├── fitness.py             # Weighted fitness computation
│   ├── island.py              # Island management + migration
│   └── types.py               # Core types (PromptPair, Elite, Variant, etc.)
├── evaluator.py               # LLM judge for trace scoring
├── grader.py                  # SWE-bench gold test evaluation
└── types.py                   # Shared types
tasks/                         # Feature task definitions (markdown)
seed_learnings/                # Hand-written repo knowledge for warm start
```

## Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/akhatua2/anneal.git
cd anneal

# Install dependencies
uv sync

# Set API keys
cp .env.example .env  # Add your ANTHROPIC_API_KEY, MODAL_TOKEN_ID, etc.
```

Requires [Modal](https://modal.com/) for remote execution and SWE-bench Docker images.

## Usage

### Single task run (inner loop only)

```bash
uv run python try_runner.py caplog-assert-logged
```

### Full evolution (outer loop)

```bash
uv run python run_supervisor.py
```

## Models

| Role | Model |
|------|-------|
| Coder | claude-opus-4-6 |
| Reviewer | claude-sonnet-4-6 |
| Proposer | claude-opus-4-6 |
| Judge | claude-opus-4-6 |

## Tasks

Four open-ended feature tasks for the pytest codebase:
- `caplog-assert-logged` — Add `assert_logged()` method to caplog fixture
- `flaky-test-retry` — Built-in `@pytest.mark.flaky(retries=N)` marker
- `raises-match-type` — Exact exception type matching for `pytest.raises`
- `mark-timeout` — Per-test timeout marker
