"""BaseCoder: a mini-swe-agent v2 wrapper that acts as a software engineer fixing issues."""

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel

CODER_SYSTEM_TEMPLATE = """\
You are a software engineer tasked with resolving issues in a codebase.

You produce minimal, clean diffs. You do NOT:
- Add unnecessary comments, docstrings, or logging
- Refactor code beyond what the issue requires
- Add speculative error handling or feature flags
- Leave debug prints or TODOs behind

You DO:
- Read existing code before changing it
- Match the repo's naming conventions, formatting, and patterns
- Change only what is needed to fix the issue
- Verify your fix works before submitting

{% if skill_docs -%}
## Learned Guidelines
The following guidelines were learned from previous iterations. Follow them.

{{ skill_docs }}
{% endif -%}
"""

CODER_INSTANCE_TEMPLATE = """\
Please fix this issue: {{task}}

You can execute bash commands and edit files to implement the necessary changes.

## Workflow

1. Read the relevant files to understand the codebase and the issue
2. Write a minimal fix
3. Run any existing tests to verify correctness
4. Submit when tests pass

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
- Submit your changes and finish your work by issuing: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
  Do not combine it with any other command.

<system_information>
{{system}} {{release}} {{version}} {{machine}}
</system_information>

{% if review_feedback -%}
## Review Feedback (from previous attempt)
The reviewer requested changes. Address ALL of the following:

{{ review_feedback }}
{% endif -%}
"""

CODER_OBSERVATION_TEMPLATE = """\
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

CODER_FORMAT_ERROR_TEMPLATE = """\
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


class BaseCoder(DefaultAgent):
    """Wraps mini-swe-agent's DefaultAgent with a coder persona (v2 tool-calling)."""

    def __init__(
        self,
        model_name: str = "anthropic/claude-haiku-4-5-20251001",
        *,
        step_limit: int = 15,
        cost_limit: float = 1.0,
        skill_docs: str = "",
        **kwargs,
    ):
        model = LitellmModel(
            model_name=model_name,
            observation_template=CODER_OBSERVATION_TEMPLATE,
            format_error_template=CODER_FORMAT_ERROR_TEMPLATE,
            model_kwargs={"drop_params": True},
            cost_tracking="ignore_errors",
        )
        env = LocalEnvironment()
        super().__init__(
            model,
            env,
            system_template=CODER_SYSTEM_TEMPLATE,
            instance_template=CODER_INSTANCE_TEMPLATE,
            step_limit=step_limit,
            cost_limit=cost_limit,
            **kwargs,
        )
        self.extra_template_vars["skill_docs"] = skill_docs
        self.extra_template_vars["review_feedback"] = ""

    def solve(self, issue: str, review_feedback: str = "") -> dict:
        """Solve an issue, optionally incorporating review feedback from a prior attempt."""
        self.extra_template_vars["review_feedback"] = review_feedback
        return self.run(task=issue)
