from __future__ import annotations

import os
import time
from typing import List

import requests

from src.db.helpers.logging import log_error
from src.log import log


def _collect_urls(prefix: str) -> List[str]:
    return [
        url for key, url in os.environ.items() if key.startswith(prefix) and "_TOKEN" not in key and "RESET" not in key
    ]


def _all_urls_ready(urls: List[str]) -> bool:
    if not urls:
        return False
    for url in urls:
        try:
            response = requests.head(url, allow_redirects=True, timeout=5)
            if response.status_code != 200:
                return False
        except requests.RequestException:
            return False
    return True


def await_all_web_servers_ready():
    vwa_urls = _collect_urls("VWA_")
    wa_urls = _collect_urls("WA_")
    max_ready_attempts = 30
    attempt = 0
    while attempt < max_ready_attempts:
        if _all_urls_ready(vwa_urls) or _all_urls_ready(wa_urls):
            log.info("All servers reachable")
            return
        attempt += 1
        log.info(f"Waiting for servers to come online... (attempt {attempt}/{max_ready_attempts})")
        time.sleep(10)

    raise TimeoutError("Servers not reachable within the expected time")


def reset_web_servers() -> None:
    """Reset the WebArena or VisualWebArena servers and wait until ready."""

    base_url = os.environ.get("VWA_FULL_RESET") or os.environ.get("WA_FULL_RESET")
    if not base_url:
        raise ValueError("VWA_FULL_RESET or WA_FULL_RESET environment variable not set")

    log.info("Initiating web servers reset...")

    reset_url = f"{base_url}/reset"
    status_url = f"{base_url}/status"

    try:
        response = requests.get(reset_url)
        response.raise_for_status()
    except requests.RequestException as e:
        log.error(f"Failed to initiate server reset: {e}")
        log_error(str(e), phase="execution")
        # raise

    max_attempts = 60
    attempt = 0
    while attempt < max_attempts:
        try:
            response = requests.get(status_url)
            response.raise_for_status()
            if response.text.strip() == "Ready for duty!":
                log.info("Server reset completed successfully")
                break
        except requests.RequestException as e:
            log.error(f"Error checking server status: {e}")
            log_error(str(e), phase="execution")
            raise
        attempt += 1
        log.info(f"Reset in progress... (attempt {attempt}/{max_attempts})")
        time.sleep(10)
    else:
        raise TimeoutError("Server reset did not complete within the expected time")

    await_all_web_servers_ready()
