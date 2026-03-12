"""BaseReviewer: a mini-swe-agent v2 wrapper that acts as a senior engineer reviewing code."""

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel

REVIEWER_SYSTEM_TEMPLATE = """\
You are a senior software engineer performing a code review.

You are opinionated and hold a high bar. You care about:

1. **Correctness**: Does the change actually fix the issue? Are there edge cases missed?
2. **Minimalism**: Is every changed line necessary? AI coders tend to over-generate — flag unnecessary additions, speculative error handling, and unrelated refactors.
3. **Style conformance**: Does the code match the repo's existing conventions? Naming, formatting, idioms, import ordering.
4. **Hygiene**: No stray comments, debug prints, TODOs, or commented-out code.
5. **Architectural fit**: Does the change fit how this codebase is structured? Or does it introduce a foreign pattern?

You can run commands to inspect the code, read files, run tests, and verify the diff.

When you have completed your review, submit your verdict by running:
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

Your FINAL response before submitting MUST contain a structured review in this exact format:

<review>
VERDICT: APPROVE | REQUEST_CHANGES
CORRECTNESS: PASS | FAIL — <one line reason>
MINIMALISM: PASS | FAIL — <one line reason>
STYLE: PASS | FAIL — <one line reason>
HYGIENE: PASS | FAIL — <one line reason>
ARCHITECTURAL_FIT: PASS | FAIL — <one line reason>

COMMENTS:
- <specific actionable feedback, one per line>
- ...

TEST_SUGGESTIONS:
- <test case the coder should add or check, if any>
- ...
</review>

{% if skill_docs -%}
## Learned Guidelines
The following guidelines were learned from previous iterations. Follow them.

{{ skill_docs }}
{% endif -%}
"""

REVIEWER_INSTANCE_TEMPLATE = """\
Review the following code change.

## Issue being addressed
{{task}}

## Diff to review
```
{{diff}}
```

## Your job

1. Read the diff and surrounding code for context
2. Run existing tests to check if they pass
3. Check for correctness, minimalism, style, hygiene, and architectural fit
4. Write test cases if you think the fix has gaps
5. Produce your structured <review> verdict
6. Submit with: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`

## Command Execution Rules

You are operating in an environment where:

1. You issue at least one command via the bash tool
2. The system executes the command(s) in a subshell
3. You see the result(s)
4. You write your next command(s)

Each response should include:

1. **Reasoning text** where you explain your analysis and plan
2. At least one bash tool call with your command

**CRITICAL REQUIREMENTS:**

- Your response SHOULD include reasoning text explaining what you're doing
- Your response MUST include AT LEAST ONE bash tool call
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...`
- Submit with: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` (alone, not combined)

<system_information>
{{system}} {{release}} {{version}} {{machine}}
</system_information>
"""

REVIEWER_OBSERVATION_TEMPLATE = """\
{%- if output.output | length < 10000 -%}
{
  "returncode": {{ output.returncode }},
  "output": {{ output.output | tojson }}
  {%- if output.exception_info %}, "exception_info": {{ output.exception_info | tojson }}{% endif %}
}
{%- else -%}
{
  "returncode": {{ output.returncode }},
  "output_head": {{ output.output[:5000] | tojson }},
  "output_tail": {{ output.output[-5000:] | tojson }},
  "elided_chars": {{ output.output | length - 10000 }},
  "warning": "Output too long."
  {%- if output.exception_info %}, "exception_info": {{ output.exception_info | tojson }}{% endif %}
}
{%- endif -%}
"""

REVIEWER_FORMAT_ERROR_TEMPLATE = """\
Tool call error:

<error>
{{error}}
</error>

Every response needs to use the 'bash' tool at least once to execute commands.

Call the bash tool with your command as the argument:
- Tool: bash
- Arguments: {"command": "your_command_here"}

If you want to end the task, please issue the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
without any other command.
"""


class BaseReviewer(DefaultAgent):
    """Wraps mini-swe-agent's DefaultAgent with a senior reviewer persona (v2 tool-calling)."""

    def __init__(
        self,
        model_name: str = "anthropic/claude-haiku-4-5-20251001",
        *,
        step_limit: int = 10,
        cost_limit: float = 0.5,
        skill_docs: str = "",
        **kwargs,
    ):
        model = LitellmModel(
            model_name=model_name,
            observation_template=REVIEWER_OBSERVATION_TEMPLATE,
            format_error_template=REVIEWER_FORMAT_ERROR_TEMPLATE,
            model_kwargs={"drop_params": True},
            cost_tracking="ignore_errors",
        )
        env = LocalEnvironment()
        super().__init__(
            model,
            env,
            system_template=REVIEWER_SYSTEM_TEMPLATE,
            instance_template=REVIEWER_INSTANCE_TEMPLATE,
            step_limit=step_limit,
            cost_limit=cost_limit,
            **kwargs,
        )
        self.extra_template_vars["skill_docs"] = skill_docs
        self.extra_template_vars["diff"] = ""

    def review(self, issue: str, diff: str) -> dict:
        """Review a diff produced by a coder for a given issue."""
        self.extra_template_vars["diff"] = diff
        return self.run(task=issue)

    def parse_verdict(self) -> dict:
        """Extract structured review data from the agent's message history."""
        for msg in reversed(self.messages):
            content = msg.get("content", "")
            if not content or "<review>" not in content:
                continue
            review_text = content.split("<review>")[1].split("</review>")[0].strip()
            result = {
                "raw": review_text,
                "approved": "VERDICT: APPROVE" in review_text,
                "axes": {},
                "comments": [],
                "test_suggestions": [],
            }
            for axis in ["CORRECTNESS", "MINIMALISM", "STYLE", "HYGIENE", "ARCHITECTURAL_FIT"]:
                for line in review_text.splitlines():
                    if line.strip().startswith(f"{axis}:"):
                        passed = "PASS" in line.split(":", 1)[1]
                        reason = line.split("—")[-1].strip() if "—" in line else ""
                        result["axes"][axis.lower()] = {"passed": passed, "reason": reason}
            in_comments = False
            in_tests = False
            for line in review_text.splitlines():
                stripped = line.strip()
                if stripped.startswith("COMMENTS:"):
                    in_comments, in_tests = True, False
                    continue
                if stripped.startswith("TEST_SUGGESTIONS:"):
                    in_comments, in_tests = False, True
                    continue
                if any(stripped.startswith(k) for k in [
                    "VERDICT:", "CORRECTNESS:", "MINIMALISM:", "STYLE:", "HYGIENE:", "ARCHITECTURAL_FIT:"
                ]):
                    in_comments, in_tests = False, False
                    continue
                if stripped.startswith("- ") and in_comments:
                    result["comments"].append(stripped[2:])
                elif stripped.startswith("- ") and in_tests:
                    result["test_suggestions"].append(stripped[2:])
            return result
        return {"raw": "", "approved": False, "axes": {}, "comments": [], "test_suggestions": []}
