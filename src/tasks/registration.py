# .venv/lib/python3.10/site-packages/browsergym/core/registration.py

from functools import partial
from typing import Any, Callable, Type, cast

import gymnasium as gym
from browsergym.core.registration import frozen_partial
from browsergym.core.task import AbstractBrowserTask

from src.environments.safeenv import SafeBrowserEnv


def register_safe_env_task(
    id: str,
    task_class: Type[AbstractBrowserTask],
    task_kwargs: dict | None = None,
    default_task_kwargs: dict | None = None,
    nondeterministic: bool = True,
    **kwargs,
) -> None:
    """Register a browser task as a gym environment."""
    task_kwargs = task_kwargs or {}
    default_task_kwargs = default_task_kwargs or {}
    if task_kwargs and default_task_kwargs:
        # check overlap between frozen and default task_kwargs
        clashing_kwargs = set(task_kwargs) & set(default_task_kwargs)  # key set intersection
        if clashing_kwargs:
            raise ValueError(
                f"Illegal attempt to register Browsergym environment {id} with both frozen and default values for task parameters {clashing_kwargs}."
            )

    task_entrypoint: Callable[..., AbstractBrowserTask] = task_class

    # freeze task_kwargs (cannot be overriden at environment creation)
    task_entrypoint = frozen_partial(task_class, **task_kwargs)

    # pre-set default_task_kwargs (can be overriden at environment creation)
    task_entrypoint = cast(Callable[..., AbstractBrowserTask], partial(task_entrypoint, **default_task_kwargs))

    def _env_creator(*env_args: Any, **env_kwargs: Any) -> SafeBrowserEnv:
        return SafeBrowserEnv(task_entrypoint, *env_args, **env_kwargs)

    gym.register(
        id=f"browsergym/{id}",
        entry_point=_env_creator,
        nondeterministic=nondeterministic,
        **kwargs,
    )
