from __future__ import annotations

import getpass
import importlib
import json
import time
from typing import Any, Dict, List

from pydantic import BaseModel

from .defaults import (
    DEFAULT_ENGINE_OPTIONS,
)
from .models import (
    AgentSrcType,
    AttackMode,
    BrowserEnvArgs,
    DefenseMode,
    ExperimentConfig,
    WebAgentEnvArgs,
)


def build_model_dump_exclude(ignore_keys: List[str]) -> Dict[str, Any]:
    """Convert list of ignore keys into nested exclude dictionary for model_dump."""
    exclude: Dict[str, Any] = {}
    for key in ignore_keys:
        parts = key.split(".")
        d = exclude
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = True
    return exclude


def get_experiment_config(
    name: str,
    description: str,
    tasks: List[str],
    engine_model: str,
    webagent_src: List[AgentSrcType],
    defense: DefenseMode = DefenseMode.NONE,
    defense_kwargs: Dict[str, Any] | None = None,
    attack: AttackMode = AttackMode.NONE,
    attack_kwargs: Dict[str, Any] | None = None,
    max_steps: int = 12,
    n_parallel_agents: int = 2,
) -> ExperimentConfig:
    agent_env_args: Dict[str, Any] = {}
    for src in webagent_src:
        if src == AgentSrcType.HTML:
            agent_env_args["use_html"] = True
        elif src == AgentSrcType.A11Y:
            agent_env_args["use_axtree"] = True
        elif src == AgentSrcType.SCREENSHOT:
            agent_env_args["use_screenshot"] = True

    engine_options = DEFAULT_ENGINE_OPTIONS.model_copy()
    engine_options.sampling_params.model = engine_model
    engine_options.max_steps = max_steps
    engine_options.n_parallel_agents = n_parallel_agents

    return ExperimentConfig(
        experiment_name=name,
        experiment_description=description,
        tasks=tasks,
        engine_options=engine_options,
        agent_env_args=WebAgentEnvArgs(**agent_env_args),
        env_args=BrowserEnvArgs(
            defense=defense,
            defense_kwargs=defense_kwargs or {},
            attack=attack,
            attack_kwargs=attack_kwargs or {},
        ),
    )


def generate_experiment_name(
    basename: str,
    webagent_srcs: List[AgentSrcType],
    engine_model: str,
    defense: DefenseMode,
    attack: AttackMode,
    timestamp: bool = False,
) -> str:
    src_str = "-".join(src.value for src in webagent_srcs)

    defense_str = f"-{str(defense)}" if defense else ""
    attack_str = f"-{str(attack)}" if attack else ""
    return f"{basename}-{src_str}{defense_str}{attack_str}-{engine_model}" + (
        "-" + str(int(time.time())) if timestamp else ""
    )


def generate_defense_harness_experiment_name(
    basename: str,
    defense: DefenseMode,
    defense_model: str | None = None,
    timestamp: bool = False,
) -> str:
    """Generate a name for a defense harness experiment."""

    model_suffix = f"-{defense_model}" if defense_model else ""
    name = f"{basename}-{str(defense)}{model_suffix}"
    if timestamp:
        name += f"-{int(time.time())}"
    return name


def are_configs_equal(
    config1: dict | str | BaseModel, config2: dict | str | BaseModel, ignore_keys: List[str] | None = None
) -> tuple[bool, dict]:
    if isinstance(config1, BaseModel):
        config1 = config1.model_dump()
    if isinstance(config2, BaseModel):
        config2 = config2.model_dump()

    if isinstance(config1, str):
        config1 = json.loads(config1)
    if isinstance(config2, str):
        config2 = json.loads(config2)

    def _remove_keys(cfg, keys):
        for key_path in keys:
            parts = key_path.split(".")
            d = cfg
            for part in parts[:-1]:
                if isinstance(d, dict) and part in d:
                    d = d[part]
                else:
                    d = None
                    break
            if isinstance(d, dict) and parts[-1] in d:
                del d[parts[-1]]

    _remove_keys(config1, ignore_keys or [])
    _remove_keys(config2, ignore_keys or [])

    c1 = json.loads(json.dumps(config1, sort_keys=True, default=lambda s: str(s)))
    c2 = json.loads(json.dumps(config2, sort_keys=True, default=lambda s: str(s)))

    differences: dict = {}

    def find_differences(d1, d2, path=""):
        if isinstance(d1, dict) and isinstance(d2, dict):
            all_keys = set(d1.keys()) | set(d2.keys())
            for key in all_keys:
                new_path = f"{path}.{key}" if path else key
                if key not in d1:
                    differences[new_path] = {"config1": None, "config2": d2[key]}
                elif key not in d2:
                    differences[new_path] = {"config1": d1[key], "config2": None}
                else:
                    find_differences(d1[key], d2[key], new_path)
        elif d1 != d2:
            differences[path] = {"config1": d1, "config2": d2}

    find_differences(c1, c2)
    return len(differences) == 0, differences


def load_user_config(module_path_template: str) -> Any:
    """
    Load a user-specific configuration module dynamically.

    Args:
        module_path_template: A template string for the module path that contains '{user}'
                            which will be replaced with the current username.
                            Example: "src.user_configs.{user}"

    Returns:
        The loaded configuration module

    Raises:
        ImportError: If the user-specific configuration module cannot be found
    """
    user_name = getpass.getuser()
    try:
        module_path = module_path_template.format(user=user_name)
        return importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Could not find config file for user {user_name}. Expected at {module_path.replace('.', '/')}.py"
        ) from e
