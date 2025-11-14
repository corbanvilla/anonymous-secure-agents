from __future__ import annotations

import base64
from typing import Any, List


def is_base64_content(content: Any) -> bool:
    if not isinstance(content, str):
        return False
    if content.startswith("data:") or ";base64," in content:
        return True
    try:
        base64.b64decode(content, validate=True)
        return True
    except Exception:
        return False


def truncate_content(content: Any, max_length: int = 1250) -> Any:
    if isinstance(content, str):
        if is_base64_content(content):
            return "[BASE64_DATA_TRUNCATED]"
        return content[:max_length] + "... [TRUNCATED]" if len(content) > max_length else content

    if isinstance(content, list):
        return [truncate_content(item, max_length=max_length) for item in content]

    if isinstance(content, dict):
        if "image_url" in content:
            url = content["image_url"].get("url", "")
            if is_base64_content(url):
                return {"type": "text", "text": "[BASE64_DATA_TRUNCATED]"}

        if "text" in content:
            content = content.copy()
            content["text"] = truncate_content(content["text"], max_length=max_length)
        return content

    return content


def truncate_messages(messages: List[dict]) -> List[dict]:
    if not messages:
        return messages
    truncated: List[dict] = []
    for msg in messages:
        truncated_msg = msg.copy()
        if "content" in msg:
            truncated_msg["content"] = truncate_content(msg["content"])
        truncated.append(truncated_msg)
    return truncated


__all__ = [
    "is_base64_content",
    "truncate_content",
    "truncate_messages",
]
