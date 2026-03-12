"""Factory functions to build coder/reviewer agents from YAML configs.

Everything tunable lives in the YAML. This module just wires up
mini-swe-agent's get_model/get_environment/get_agent with runtime
template variables (skill_docs, review_feedback, patch).
"""

from pathlib import Path

from minisweagent.agents import get_agent
from minisweagent.config import get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.utils.serialize import recursive_merge

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _load_config(config_path: str | Path, overrides: dict | None = None) -> dict:
    """Load a YAML config and merge any overrides."""
    config = get_config_from_spec(config_path)
    if overrides:
        config = recursive_merge(config, overrides)
    return config


def make_coder(
    image: str,
    *,
    config_path: str | Path = CONFIGS_DIR / "base_coder.yaml",
    skill_docs: str = "",
    review_feedback: str = "",
    overrides: dict | None = None,
):
    """Build a coder agent with a Modal environment.

    Returns (agent, env) so the caller can manage the container lifecycle.
    """
    config = _load_config(config_path, overrides)

    env_config = config.get("environment", {})
    env_config.setdefault("environment_class", "swerex_modal")
    env_config["image"] = image
    env = get_environment(env_config)

    model = get_model(config=config.get("model", {}))

    agent = get_agent(model, env, config.get("agent", {}), default_type="default")
    agent.extra_template_vars["skill_docs"] = skill_docs
    agent.extra_template_vars["review_feedback"] = review_feedback

    return agent, env


def make_reviewer(
    image: str,
    *,
    config_path: str | Path = CONFIGS_DIR / "base_reviewer.yaml",
    skill_docs: str = "",
    overrides: dict | None = None,
):
    """Build a reviewer agent with its own Modal environment.

    Returns (agent, env) so the caller can manage the container lifecycle.
    """
    config = _load_config(config_path, overrides)

    env_config = config.get("environment", {})
    env_config.setdefault("environment_class", "swerex_modal")
    env_config["image"] = image
    env = get_environment(env_config)

    model = get_model(config=config.get("model", {}))

    agent = get_agent(model, env, config.get("agent", {}), default_type="default")
    agent.extra_template_vars["skill_docs"] = skill_docs

    return agent, env
