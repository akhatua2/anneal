"""SWE-bench grader: evaluates agent patches using the swebench evaluation harness."""

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import docker

from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
)
from swebench.harness.run_evaluation import run_instance
from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec

logger = logging.getLogger("anneal.grader")

DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
DATASET_SPLIT = "test"


@dataclass
class GradeResult:
    instance_id: str
    resolved: bool
    completed: bool
    report: dict
    report_path: Path | None


class SWEBenchGrader:
    """Grade agent patches against SWE-bench ground-truth tests.

    Uses swebench's run_instance() to spin up a fresh Docker container,
    apply the patch, run tests, and grade the result.
    """

    def __init__(
        self,
        instance: dict,
        *,
        model_name: str = "anneal",
        namespace: str | None = "swebench",
        timeout: int = 1800,
    ):
        self.instance = instance
        self.instance_id = instance[KEY_INSTANCE_ID]
        self.model_name = model_name
        self.timeout = timeout
        self.test_spec: TestSpec = make_test_spec(instance, namespace=namespace)

    @classmethod
    def from_dataset(
        cls,
        instance_id: str,
        *,
        dataset_name: str = DATASET_NAME,
        split: str = DATASET_SPLIT,
        **kwargs,
    ) -> "SWEBenchGrader":
        """Load a single instance from HuggingFace and create a grader."""
        from datasets import load_dataset

        logger.info(f"Loading instance {instance_id} from {dataset_name} ({split})...")
        ds = load_dataset(dataset_name, split=split)
        matches = [row for row in ds if row[KEY_INSTANCE_ID] == instance_id]
        if not matches:
            raise ValueError(
                f"Instance {instance_id} not found in {dataset_name}/{split}"
            )
        return cls(matches[0], **kwargs)

    def grade(self, patch: str, *, run_id: str | None = None) -> GradeResult:
        """Grade a patch against the SWE-bench test suite.

        Args:
            patch: The git diff patch string to evaluate.
            run_id: Optional run identifier (auto-generated if not provided).

        Returns:
            GradeResult with resolution status and test details.
        """
        if run_id is None:
            run_id = f"anneal_{uuid.uuid4().hex[:8]}"

        pred = {
            KEY_INSTANCE_ID: self.instance_id,
            KEY_MODEL: self.model_name,
            KEY_PREDICTION: patch,
        }

        logger.info(f"Grading {self.instance_id} (run_id={run_id})...")
        client = docker.from_env()

        try:
            result = run_instance(
                test_spec=self.test_spec,
                pred=pred,
                rm_image=False,
                force_rebuild=False,
                client=client,
                run_id=run_id,
                timeout=self.timeout,
            )
        finally:
            client.close()

        # Load the full report if it exists
        from swebench.harness.constants import RUN_EVALUATION_LOG_DIR, LOG_REPORT

        report_path = (
            RUN_EVALUATION_LOG_DIR
            / run_id
            / self.model_name.replace("/", "__")
            / self.instance_id
            / LOG_REPORT
        )
        report = {}
        if report_path.exists():
            report = json.loads(report_path.read_text())

        return GradeResult(
            instance_id=self.instance_id,
            resolved=result.get("resolved", False),
            completed=result.get("completed", False),
            report=report,
            report_path=report_path if report_path.exists() else None,
        )

    @property
    def fail_to_pass(self) -> list[str]:
        return self.test_spec.FAIL_TO_PASS

    @property
    def pass_to_pass(self) -> list[str]:
        return self.test_spec.PASS_TO_PASS

    @property
    def problem_statement(self) -> str:
        return self.instance.get("problem_statement", "")
