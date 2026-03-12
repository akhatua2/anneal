"""Runner: orchestrates coder → reviewer → coder revision loops."""

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from anneal.agents.base_coder import BaseCoder
from anneal.agents.base_reviewer import BaseReviewer

logger = logging.getLogger("anneal.runner")


@dataclass
class RoundResult:
    """One round of coder attempt + reviewer feedback."""
    round_num: int
    coder_exit: dict
    diff: str
    verdict: dict
    coder_messages: list[dict] = field(default_factory=list)
    reviewer_messages: list[dict] = field(default_factory=list)


@dataclass
class Trace:
    """Full interaction trace for one issue."""
    issue: str
    rounds: list[RoundResult] = field(default_factory=list)
    final_diff: str = ""
    approved: bool = False
    tests_passed: bool | None = None
    total_rounds: int = 0
    timestamp: float = field(default_factory=time.time)


class Runner:
    """Runs the coder→reviewer→revision loop until approved or max rounds."""

    def __init__(
        self,
        repo_path: str | Path,
        *,
        coder_model: str = "anthropic/claude-haiku-4-5-20251001",
        reviewer_model: str = "anthropic/claude-haiku-4-5-20251001",
        max_rounds: int = 3,
        coder_step_limit: int = 15,
        reviewer_step_limit: int = 10,
        coder_cost_limit: float = 1.0,
        reviewer_cost_limit: float = 0.5,
        coder_skill_docs: str = "",
        reviewer_skill_docs: str = "",
        test_cmd: str | None = None,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.coder_model = coder_model
        self.reviewer_model = reviewer_model
        self.max_rounds = max_rounds
        self.coder_step_limit = coder_step_limit
        self.reviewer_step_limit = reviewer_step_limit
        self.coder_cost_limit = coder_cost_limit
        self.reviewer_cost_limit = reviewer_cost_limit
        self.coder_skill_docs = coder_skill_docs
        self.reviewer_skill_docs = reviewer_skill_docs
        self.test_cmd = test_cmd

    def _git(self, *args: str) -> str:
        """Run a git command in the repo and return stdout."""
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _get_diff(self) -> str:
        """Get the current uncommitted diff (staged + unstaged)."""
        # Capture both tracked changes and new files
        tracked = self._git("diff", "HEAD")
        untracked_files = self._git("ls-files", "--others", "--exclude-standard")
        if not untracked_files:
            return tracked

        # For untracked files, show their content as a pseudo-diff
        parts = [tracked] if tracked else []
        for f in untracked_files.splitlines():
            f = f.strip()
            if not f:
                continue
            try:
                content = (self.repo_path / f).read_text()
                parts.append(f"--- /dev/null\n+++ b/{f}\n@@ -0,0 +1 @@\n" +
                             "\n".join(f"+{line}" for line in content.splitlines()))
            except Exception:
                parts.append(f"--- /dev/null\n+++ b/{f}\n(binary or unreadable)")
        return "\n".join(parts)

    def _reset_repo(self):
        """Reset repo to clean state (HEAD)."""
        self._git("checkout", ".")
        self._git("clean", "-fd")

    def _run_tests(self) -> bool | None:
        """Run the test command if configured. Returns True/False/None."""
        if not self.test_cmd:
            return None
        result = subprocess.run(
            self.test_cmd,
            shell=True,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0

    def _make_coder(self, review_feedback: str = "") -> BaseCoder:
        """Create a fresh coder agent for this round."""
        coder = BaseCoder(
            model_name=self.coder_model,
            step_limit=self.coder_step_limit,
            cost_limit=self.coder_cost_limit,
            skill_docs=self.coder_skill_docs,
        )
        # Point the coder's environment at the repo
        coder.env.config.cwd = str(self.repo_path)
        return coder

    def _make_reviewer(self) -> BaseReviewer:
        """Create a fresh reviewer agent for this round."""
        reviewer = BaseReviewer(
            model_name=self.reviewer_model,
            step_limit=self.reviewer_step_limit,
            cost_limit=self.reviewer_cost_limit,
            skill_docs=self.reviewer_skill_docs,
        )
        reviewer.env.config.cwd = str(self.repo_path)
        return reviewer

    def _format_feedback(self, verdict: dict) -> str:
        """Turn a parsed verdict into actionable feedback for the coder."""
        lines = []
        for axis, info in verdict.get("axes", {}).items():
            if not info["passed"]:
                lines.append(f"[{axis.upper()} FAIL] {info['reason']}")
        if verdict.get("comments"):
            lines.append("")
            lines.append("Reviewer comments:")
            for c in verdict["comments"]:
                lines.append(f"  - {c}")
        if verdict.get("test_suggestions"):
            lines.append("")
            lines.append("Suggested tests to add/check:")
            for t in verdict["test_suggestions"]:
                lines.append(f"  - {t}")
        return "\n".join(lines)

    def run(self, issue: str) -> Trace:
        """Execute the full coder→reviewer loop for an issue."""
        trace = Trace(issue=issue)
        review_feedback = ""

        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"=== Round {round_num}/{self.max_rounds} ===")

            # Reset repo before each coder attempt
            self._reset_repo()

            # --- Coder phase ---
            logger.info("Coder is working...")
            coder = self._make_coder()
            try:
                coder_exit = coder.solve(issue, review_feedback=review_feedback)
            except Exception as e:
                logger.error(f"Coder failed: {e}")
                coder_exit = {"exit_status": "error", "submission": ""}

            diff = self._get_diff()
            if not diff.strip():
                logger.warning("Coder produced no diff, skipping review")
                trace.rounds.append(RoundResult(
                    round_num=round_num,
                    coder_exit=coder_exit,
                    diff="",
                    verdict={"raw": "", "approved": False, "axes": {}, "comments": ["No changes produced"], "test_suggestions": []},
                    coder_messages=coder.messages,
                    reviewer_messages=[],
                ))
                continue

            # --- Reviewer phase ---
            logger.info("Reviewer is reviewing...")
            reviewer = self._make_reviewer()
            try:
                reviewer.review(issue, diff)
            except Exception as e:
                logger.error(f"Reviewer failed: {e}")

            verdict = reviewer.parse_verdict()

            round_result = RoundResult(
                round_num=round_num,
                coder_exit=coder_exit,
                diff=diff,
                verdict=verdict,
                coder_messages=coder.messages,
                reviewer_messages=reviewer.messages,
            )
            trace.rounds.append(round_result)

            logger.info(f"Verdict: {'APPROVED' if verdict['approved'] else 'REQUEST_CHANGES'}")
            for axis, info in verdict.get("axes", {}).items():
                logger.info(f"  {axis}: {'PASS' if info['passed'] else 'FAIL'} — {info['reason']}")

            if verdict["approved"]:
                trace.approved = True
                break

            # Format feedback for next round
            review_feedback = self._format_feedback(verdict)
            logger.info(f"Feeding back to coder:\n{review_feedback}")

        # Final state
        trace.final_diff = self._get_diff()
        trace.total_rounds = len(trace.rounds)
        trace.tests_passed = self._run_tests()

        logger.info(f"Done. Rounds: {trace.total_rounds}, Approved: {trace.approved}, Tests: {trace.tests_passed}")
        return trace
