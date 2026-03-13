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
                            OUTER LOOP (Supervisor)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │   ┌──────────┐    ┌──────────────────────────────────────────────────┐  │
  │   │  PROPOSE  │    │              ISLAND POOL                        │  │
  │   │           │    │                                                  │  │
  │   │ LLM reads │    │  ┌────────────┐ ┌────────────┐ ┌────────────┐  │  │
  │   │ elite +   │◄───┤  │ first_try  │ │ efficiency │ │ feedback   │  │  │
  │   │ eval      │    │  │            │ │            │ │ _quality   │  │  │
  │   │ feedback, │    │  │ elite_0    │ │ elite_0    │ │ elite_0    │  │  │
  │   │ mutates   │    │  │ elite_1    │ │ elite_1    │ │ elite_1    │  │  │
  │   │ prompts   │    │  │ elite_2    │ │ elite_2    │ │ elite_2    │  │  │
  │   └─────┬─────┘    │  └─────┬──────┘ └──────┬─────┘ └─────┬──────┘  │  │
  │         │          │        │                │             │          │  │
  │         │          │  ┌─────┴──────┐ ┌──────┴─────┐ ┌─────┴──────┐  │  │
  │         │          │  │correctness │ │adversarial │ │            │  │  │
  │         │          │  │            │ │ _reviewer  │ │  MIGRATE   │  │  │
  │         │          │  │ elite_0    │ │ elite_0    │ │ every 2    │  │  │
  │         │          │  │ elite_1    │ │ elite_1    │ │ generations│  │  │
  │         │          │  │ elite_2    │ │ elite_2    │ │  ◄──────►  │  │  │
  │         │          │  └────────────┘ └────────────┘ └────────────┘  │  │
  │         │          └──────────────────────────────────────────────────┘  │
  │         │                                          ▲                    │
  │         ▼                                          │                    │
  │   ┌──────────┐         ┌──────────┐         ┌─────┴────┐              │
  │   │ ROLLOUT  │────────►│ EVALUATE │────────►│  SELECT  │              │
  │   │          │         │          │         │          │              │
  │   │ N variant│         │ LLM judge│         │ best     │              │
  │   │ x M tasks│         │ scores   │         │ variant  │              │
  │   │ parallel │         │ each     │         │ replaces │              │
  │   │ on Modal │         │ trace    │         │ worst    │              │
  │   └──────────┘         └──────────┘         │ elite    │              │
  │         │                                    └──────────┘              │
  │         │                                                              │
  └─────────┼──────────────────────────────────────────────────────────────┘
            │
            │  Each rollout runs the inner loop:
            ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                         INNER LOOP (Runner)                             │
  │                                                                         │
  │   ┌──────────────┐                          ┌──────────────┐           │
  │   │  CODER       │    /tmp/patch.txt        │  REVIEWER    │           │
  │   │  (Opus)      │ ──────────────────────►  │  (Sonnet)    │           │
  │   │              │                          │              │           │
  │   │  reads issue │                          │  reads diff  │           │
  │   │  explores    │    VERDICT: ACCEPT       │  checks:     │           │
  │   │  codes fix   │ ◄────────────────────    │  correctness │           │
  │   │  writes tests│    or REQUEST_CHANGES    │  minimalism  │           │
  │   │              │    + line comments        │  style       │           │
  │   └──────┬───────┘                          │  tests       │           │
  │          │                                  └──────┬───────┘           │
  │          │                                         │                   │
  │          │         ┌────────────────────┐          │                   │
  │          │         │   MEMORY FILES     │          │                   │
  │          └────────►│                    │◄─────────┘                   │
  │                    │  repo_learnings.txt│                              │
  │                    │  (cross-task)      │                              │
  │                    │                    │                              │
  │                    │  task_notes.txt    │                              │
  │                    │  (per-task)        │                              │
  │                    └────────────────────┘                              │
  │                                                                         │
  │   Iterates until ACCEPT or max rounds (10)                             │
  │   Runs inside SWE-bench Docker image on Modal                          │
  └─────────────────────────────────────────────────────────────────────────┘


  EVOLUTION FLOW (per generation):

  Generation N
  ─────────────────────────────────────────────────────────────────────

  5 islands x 4 variants x 4 tasks = 80 parallel rollouts

       Island 1          Island 2          Island 3          Island 4          Island 5
    ┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐
    │ first_try │     │ efficiency│     │ feedback  │     │correctness│     │adversarial│
    │           │     │           │     │ _quality  │     │           │     │ _reviewer │
    │ propose 4 │     │ propose 4 │     │ propose 4 │     │ propose 4 │     │ propose 4 │
    │ variants  │     │ variants  │     │ variants  │     │ variants  │     │ variants  │
    └─────┬─────┘     └─────┬─────┘     └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
          │                 │                 │                 │                 │
          └────────┬────────┴────────┬────────┴────────┬───────┴────────┬────────┘
                   │                 │                 │                │
                   ▼                 ▼                 ▼                ▼
            ┌─────────────────────────────────────────────────────────────────┐
            │              80 ROLLOUTS IN PARALLEL (Modal)                    │
            │                                                                 │
            │   variant_0 x task_0    variant_0 x task_1    ...              │
            │   variant_1 x task_0    variant_1 x task_1    ...              │
            │   ...                                                          │
            └─────────────────────────────────────────────────────────────────┘
                   │                 │                 │                │
                   ▼                 ▼                 ▼                ▼
            ┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐
            │  evaluate │     │  evaluate │     │  evaluate │     │  evaluate │
            │  + select │     │  + select │     │  + select │     │  + select │
            └───────────┘     └───────────┘     └───────────┘     └───────────┘

  Every 2 generations: MIGRATE best elites across islands
```

### File structure

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

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [Modal](https://modal.com/) account (for remote sandbox execution)
- Anthropic API key
- SWE-bench Docker images (for the pytest test environment)

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/akhatua2/anneal.git
cd anneal
```

The project depends on two local packages ([mini-swe-agent](https://github.com/princeton-nlp/SWE-agent) and [swe-bench](https://github.com/princeton-nlp/SWE-bench)) which need to be cloned into `external/`:

```bash
mkdir -p external
git clone https://github.com/SWE-agent/mini-swe-agent.git external/mini-swe-agent
git clone https://github.com/princeton-nlp/SWE-bench.git external/swe-bench
```

Then install:

```bash
uv sync
```

### 2. Configure environment

Create a `.env` file in the project root:

```bash
ANTHROPIC_API_KEY=sk-ant-...
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
```

### 3. Set up Modal

```bash
uv run modal setup   # authenticate with Modal
```

### 4. Pull the SWE-bench Docker image

The tasks run inside a SWE-bench Docker image with the pytest codebase pre-installed:

```bash
docker pull swebench/sweb.eval.x86_64.pytest-dev_1776_pytest-10051:latest
```

## Usage

### Single task run (inner loop only)

Run one coder/reviewer loop on a specific task:

```bash
uv run python try_runner.py caplog-assert-logged
```

Available tasks: `caplog-assert-logged`, `flaky-test-retry`, `raises-match-type`, `mark-timeout`

Output goes to `output/features/<task-name>/<timestamp>/`.

### Full evolution (outer loop)

Run the supervisor to evolve prompts across generations:

```bash
# Default: 3 generations, all 5 islands, 4 variants each
uv run python run_supervisor.py

# Customize
uv run python run_supervisor.py --generations 5 --islands first_try efficiency
uv run python run_supervisor.py --tasks caplog-assert-logged raises-match-type
uv run python run_supervisor.py --max-workers 30

# Preview without running
uv run python run_supervisor.py --dry-run
```

Output goes to `experiments/`.

### CLI options for `run_supervisor.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--generations` | 3 | Number of evolution generations |
| `--islands` | all 5 | Which islands to run (space-separated names) |
| `--n-variants` | 4 | Prompt variants per island per generation |
| `--max-workers` | 20 | Max parallel rollouts |
| `--tasks` | all 4 | Which tasks to use (space-separated IDs) |
| `--tasks-per-gen` | all | Sample N tasks per generation |
| `--seed-learnings` | `seed_learnings/pytest-dev__pytest` | Pre-written repo knowledge |
| `--experiments-dir` | `experiments` | Output directory |
| `--dry-run` | — | Show config and estimated cost, then exit |

## Models

| Role | Model |
|------|-------|
| Coder | claude-opus-4-6 |
| Reviewer | claude-sonnet-4-6 |
| Proposer | claude-opus-4-6 |
| Judge | claude-opus-4-6 |

## Tasks

Four open-ended feature tasks for the pytest codebase:

- **caplog-assert-logged** — Add an `assert_logged()` convenience method to the `caplog` fixture
- **flaky-test-retry** — Built-in `@pytest.mark.flaky(retries=N)` marker for retrying flaky tests
- **raises-match-type** — Exact exception type matching for `pytest.raises`
- **mark-timeout** — Per-test `@pytest.mark.timeout(seconds=N)` marker

Tasks are defined as markdown files in `tasks/`. You can add your own by creating a new `.md` file describing the feature or bug fix.
