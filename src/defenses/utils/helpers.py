import base64
import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def load_text(path: str) -> str:
    txt = open(path, "r", encoding="utf-8", errors="ignore").read()
    logger.info(f"Loaded HTML text ({len(txt)} chars)")
    return txt


def load_b64(path: str) -> str:
    data = open(path, "rb").read()
    b64 = base64.b64encode(data).decode("ascii")
    logger.info(f"Loaded image ({len(data)} bytes)")
    return b64


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)
    logger.info(f"Saved JSON to {path}")
