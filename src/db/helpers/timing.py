from __future__ import annotations

import time
from functools import wraps
from typing import Callable, Optional, TypeVar, ParamSpec, cast

from src.db.client import Session, safe_commit
from src.db.tables import FunctionTiming

P = ParamSpec("P")
R = TypeVar("R")


def record_timing(*, name: Optional[str] = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to record function execution duration in the database.

    It also stores the model and defense names when available.
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                duration = time.time() - start
                model_name = None
                defense_name = None
                if args:
                    instance = args[0]
                    if hasattr(instance, "sampling_params"):
                        params = getattr(instance, "sampling_params") or {}
                        if isinstance(params, dict):
                            model_name = params.get("model")
                    defense_name = instance.__class__.__name__
                timing = FunctionTiming(
                    function_name=name or fn.__name__,
                    duration=duration,
                    model_name=model_name,
                    defense_name=defense_name,
                )
                with Session() as session:
                    session.add(timing)
                    safe_commit(session)

        return cast(Callable[P, R], wrapper)

    return decorator
