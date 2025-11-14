from __future__ import annotations

import inspect
import time
from typing import TYPE_CHECKING, Any, List

from src.api_clients.utils import truncate_messages

if TYPE_CHECKING:  # pragma: no cover - typing alias for mypy
    from openai import AsyncOpenAI as _AsyncOpenAIBase
else:  # pragma: no cover - runtime stub
    _AsyncOpenAIBase = object


class AsyncOpenAILoggingProxy(_AsyncOpenAIBase):
    """Async-aware proxy around an OpenAI client that records requests.

    This wrapper detects coroutine functions and awaits them before logging timing
    and response metadata. Attribute traversal returns wrapped proxies to ensure
    nested clients (e.g., `client.chat.completions`) are also logged.
    """

    def __init__(self, target: Any, log: List[dict], path: str = "") -> None:
        self._target = target
        self._log = log
        self._path = path

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        path = f"{self._path}.{name}" if self._path else name

        if callable(attr):
            # Async coroutine function
            if inspect.iscoroutinefunction(attr):

                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    start = time.time()
                    resp = await attr(*args, **kwargs)
                    end = time.time()

                    # Grab headers if caller used with_raw_response
                    headers = getattr(resp, "headers", {})
                    if headers:
                        resp = resp.parse()

                    finish_reason = None
                    if hasattr(resp, "choices"):
                        try:
                            finish_reason = resp.choices[0].finish_reason
                            if finish_reason == "content_filter":
                                raise Exception("Content filter for request!")
                        except (KeyError, AttributeError, IndexError):
                            pass

                    truncated_messages = None
                    if "messages" in kwargs:
                        truncated_messages = truncate_messages(kwargs["messages"])  # type: ignore[arg-type]

                    self._log.append(
                        {
                            "callee": path,
                            "messages": truncated_messages,
                            "model": kwargs.get("model"),
                            "response": resp.model_dump() if hasattr(resp, "model_dump") else str(resp),
                            "input_tokens": getattr(resp.usage, "prompt_tokens", None)
                            if hasattr(resp, "usage")
                            else None,
                            "cached_input_tokens": getattr(
                                getattr(resp.usage, "prompt_tokens_details", None), "cached_tokens", None
                            )
                            if hasattr(resp, "usage")
                            else None,
                            "output_tokens": getattr(resp.usage, "completion_tokens", None)
                            if hasattr(resp, "usage")
                            else None,
                            "request_start": start,
                            "request_end": end,
                            "request_duration": end - start,
                            "finish_reason": finish_reason,
                            "cached": False if "x-litellm-response-cost" in headers else True,
                        }
                    )
                    return resp

                return async_wrapper

            # If it's callable but not async, we don't support it here.
            raise TypeError(f"{path} is not an async callable; use the sync logging proxy instead")

        # Attribute access returns another proxy so nested structures are logged
        return AsyncOpenAILoggingProxy(attr, self._log, path)


__all__ = ["AsyncOpenAILoggingProxy"]
