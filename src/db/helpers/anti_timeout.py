from __future__ import annotations

import functools
import signal
from typing import Any, Callable, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def timeout_handler(_: Any, __: Any):
    """Signal handler for timeouts"""
    raise TimeoutError("SQL execution timed out")


def timed_sql_execution(timeout_seconds: int = 30) -> Callable[[F], F]:
    """Decorator to run a SQL function with a timeout."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)
                return result
            except Exception as e:
                signal.alarm(0)
                raise e

        return cast(F, wrapper)

    return decorator
