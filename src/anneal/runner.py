"""Runner: orchestrates coder -> reviewer -> coder revision loops in docker containers."""

import base64
import logging
from pathlib import Path

from anneal.agents.factory import make_coder, make_reviewer
from anneal.agents.review_parser import parse_verdict
from anneal.types import Outcome, PRAttempt, Review, Trace
from anneal.utils import save_text, save_trace

logger = logging.getLogger("anneal.runner")


def _stop_env(env):
    """Stop an environment, handling both Docker (cleanup) and Modal (stop)."""
    if hasattr(env, "cleanup"):
        env.cleanup()
    elif hasattr(env, "stop"):
        env.stop()

LEARNINGS_PATH = "/tmp/repo_learnings.txt"
TASK_NOTES_PATH = "/tmp/task_notes.txt"


def _write_file_to_container(env, container_path: str, content: str):
    encoded = base64.b64encode(content.encode()).decode()
    env.execute({"command": f"echo '{encoded}' | base64 -d > {container_path}"})


def _read_file_from_container(env, container_path: str) -> str:
    result = env.execute({"command": f"cat {container_path} 2>/dev/null || true"})
    return result.get("output", "").strip()


class Runner:
    """Orchestrates the coder/reviewer loop.

    - Coder and reviewer each get a persistent docker container.
    - Coder builds on its work across rounds (no reset).
    - Reviewer keeps its container but resets conversation each round.
    - Patch transfers coder -> reviewer (via /tmp/patch.txt).
    - Review transfers reviewer -> coder (via submission text).
    - Stops when: reviewer approves, or max rounds hit.
    - Everything is saved to disk after each step.

    Memory system (two files per agent, in /tmp/ inside the container):
    - repo_learnings.txt: codebase knowledge that persists across PRs. Separate
      per role — coder learns navigation/patterns, reviewer learns conventions/quality.
      Saved to memory/{repo_slug}/coder_learnings.txt and reviewer_learnings.txt.
    - task_notes.txt: scratch notes for the current PR. Persists across rounds
      within the same container. Separate per agent.
    """

    def __init__(
        self,
        image: str,
        *,
        max_rounds: int = 3,
        repo_slug: str = "",
        coder_config: str | None = None,
        reviewer_config: str | None = None,
        coder_skill_docs: str = "",
        reviewer_skill_docs: str = "",
        coder_overrides: dict | None = None,
        reviewer_overrides: dict | None = None,
        output_dir: str | Path = "output",
        memory_dir: str | Path = "memory",
    ):
        self.image = image
        self.max_rounds = max_rounds
        self.repo_slug = repo_slug
        self.coder_config = coder_config
        self.reviewer_config = reviewer_config
        self.coder_skill_docs = coder_skill_docs
        self.reviewer_skill_docs = reviewer_skill_docs
        self.coder_overrides = coder_overrides
        self.reviewer_overrides = reviewer_overrides
        self.output_dir = Path(output_dir)
        self.memory_dir = Path(memory_dir)

    def _learnings_path(self, role: str) -> Path:
        return self.memory_dir / self.repo_slug / f"{role}_learnings.txt"

    def _load_learnings(self, role: str) -> str:
        path = self._learnings_path(role)
        return path.read_text() if path.exists() else ""

    def _save_learnings(self, role: str, content: str):
        save_text(content, self._learnings_path(role))

    def _save_trajectory(self, agent, issue_dir: Path, round_num: int, role: str) -> str:
        traj_path = issue_dir / f"round_{round_num}" / f"{role}_trajectory.json"
        agent.save(traj_path)
        return str(traj_path)

    def run(self, issue: str, issue_id: str = "") -> Trace:
        trace = Trace(issue=issue)
        review_feedback = ""
        issue_slug = issue_id or str(int(trace.timestamp))
        issue_dir = self.output_dir / issue_slug

        # Load persistent learnings (separate per role)
        coder_learnings = self._load_learnings("coder")
        reviewer_learnings = self._load_learnings("reviewer")

        # Build factory kwargs
        coder_kwargs = dict(image=self.image, skill_docs=self.coder_skill_docs)
        if self.coder_config:
            coder_kwargs["config_path"] = self.coder_config
        if self.coder_overrides:
            coder_kwargs["overrides"] = self.coder_overrides

        reviewer_kwargs = dict(image=self.image, skill_docs=self.reviewer_skill_docs)
        if self.reviewer_config:
            reviewer_kwargs["config_path"] = self.reviewer_config
        if self.reviewer_overrides:
            reviewer_kwargs["overrides"] = self.reviewer_overrides

        # Both containers persist across rounds
        coder, coder_env = make_coder(**coder_kwargs, review_feedback=review_feedback)
        reviewer, reviewer_env = make_reviewer(**reviewer_kwargs)

        try:
            # Seed each container with its own learnings from previous PRs
            _write_file_to_container(coder_env, LEARNINGS_PATH, coder_learnings)
            _write_file_to_container(coder_env, TASK_NOTES_PATH, "")
            _write_file_to_container(reviewer_env, LEARNINGS_PATH, reviewer_learnings)
            _write_file_to_container(reviewer_env, TASK_NOTES_PATH, "")

            for round_num in range(1, self.max_rounds + 1):
                logger.info(f"=== Round {round_num}/{self.max_rounds} ===")
                round_dir = issue_dir / f"round_{round_num}"

                # --- Coder phase ---
                logger.info("Coder is working...")
                if round_num > 1:
                    coder.extra_template_vars["review_feedback"] = review_feedback
                    coder.messages = []
                    coder.cost = 0.0
                    coder.n_calls = 0

                try:
                    result = coder.run(task=issue)
                    patch = result.get("submission", "")
                except Exception as e:
                    logger.error(f"Coder failed: {e}")
                    patch = ""

                # Read back coder's learnings and notes, save to disk
                coder_learnings = _read_file_from_container(coder_env, LEARNINGS_PATH)
                self._save_learnings("coder", coder_learnings)
                save_text(coder_learnings, round_dir / "coder_learnings.txt")
                save_text(_read_file_from_container(coder_env, TASK_NOTES_PATH), round_dir / "coder_task_notes.txt")

                # Save coder outputs immediately
                coder_traj_path = self._save_trajectory(coder, issue_dir, round_num, "coder")
                save_text(patch, round_dir / "patch.txt")

                pr_attempt = PRAttempt(patch=patch, coder_trajectory=coder_traj_path)

                if not patch.strip():
                    logger.warning("Coder produced no patch, skipping review")
                    review = Review(verdict={"raw": "No patch produced", "approved": False})
                    trace.rounds.append((pr_attempt, review))
                    save_trace(trace, issue_dir)
                    continue

                # --- Reviewer phase ---
                logger.info("Reviewer is reviewing...")
                if round_num > 1:
                    reviewer.messages = []
                    reviewer.cost = 0.0
                    reviewer.n_calls = 0
                    reviewer_env.execute({"command": "cd /testbed && git checkout . && git clean -fd"})

                # Write patch into reviewer's container
                reviewer_env.execute({"command": f"cat << 'PATCH_EOF' > /tmp/patch.txt\n{patch}\nPATCH_EOF"})

                try:
                    result = reviewer.run(task=issue)
                    review_text = result.get("submission", "")
                    verdict = parse_verdict(review_text)
                    reviewer_traj_path = self._save_trajectory(reviewer, issue_dir, round_num, "reviewer")
                except Exception as e:
                    logger.error(f"Reviewer failed: {e}")
                    verdict = {"raw": f"Reviewer error: {e}", "approved": False}
                    review_text = ""
                    reviewer_traj_path = ""

                # Read back reviewer's learnings and notes, save to disk
                reviewer_learnings = _read_file_from_container(reviewer_env, LEARNINGS_PATH)
                self._save_learnings("reviewer", reviewer_learnings)
                save_text(reviewer_learnings, round_dir / "reviewer_learnings.txt")
                save_text(_read_file_from_container(reviewer_env, TASK_NOTES_PATH), round_dir / "reviewer_task_notes.txt")

                # Save review immediately
                save_text(review_text, round_dir / "review.txt")

                review = Review(verdict=verdict, reviewer_trajectory=reviewer_traj_path)
                trace.rounds.append((pr_attempt, review))

                logger.info(f"Verdict: {'APPROVED' if verdict['approved'] else 'REQUEST_CHANGES'}")
                save_trace(trace, issue_dir)

                if verdict["approved"]:
                    trace.outcome = Outcome(reviewer_approved=True, rounds_to_merge=round_num)
                    trace.final_patch = patch
                    break

                # Full review text goes back to coder as feedback
                review_feedback = verdict["raw"]
                logger.info(f"Feeding back to coder:\n{review_feedback}")

        finally:
            _stop_env(coder_env)
            _stop_env(reviewer_env)

        if not trace.final_patch and trace.rounds:
            trace.final_patch = trace.rounds[-1][0].patch
            trace.outcome = Outcome(reviewer_approved=False, rounds_to_merge=len(trace.rounds))

        save_trace(trace, issue_dir)
        save_text(trace.final_patch, issue_dir / "final.patch")
        logger.info(f"Done. Rounds: {len(trace.rounds)}, Approved: {trace.outcome.reviewer_approved}")
        return trace
