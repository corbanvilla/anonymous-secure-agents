import io
import logging
import time
from abc import ABC
from typing import Any, Dict, cast

import numpy as np
from browsergym.core.observation import extract_screenshot
from browsergym.core.spaces import AnyBox, Unicode
from openai import AsyncOpenAI, OpenAI

from src.api_clients.async_logging import AsyncOpenAILoggingProxy
from src.api_clients.logging import OpenAILoggingProxy


class AbstractDefense(ABC):
    """
    Abstract base class for implementing browser safety defenses.
    Defenses are responsible for analyzing and filtering DOM elements
    to prevent malicious content from being displayed.
    """

    # JavaScript code for hiding/unhiding elements
    _JS_SETUP_FUNCTIONS = """
(function() {
  // Inject .hidden-everything rule via adoptedStyleSheets
  const sheet = new CSSStyleSheet();
  sheet.replaceSync(`
    .hidden-everything * {
      visibility: hidden !important;
    }
  `);
  document.adoptedStyleSheets = [...document.adoptedStyleSheets, sheet];

  // Hide all content
  function hideEverything() {
    document.body.classList.add('hidden-everything');
  }

  // Unhide elements by bid attribute (including children)
  function unhide(...bids) {
    bids.forEach(bid => {
      const el = document.querySelector(`[bid="${bid}"]`);
      if (el) {
        el.style.setProperty('visibility', 'visible', 'important');
        el.querySelectorAll('*').forEach(child =>
          child.style.setProperty('visibility', 'visible', 'important')
        );
      }
    });
  }

  // Unhide everything and remove any forced visibility overrides
  function unhideEverything() {
    document.body.classList.remove('hidden-everything');
    document.querySelectorAll('[style*="visibility"]').forEach(el => {
      el.style.removeProperty('visibility');
    });
  }

  // Expose functions globally
  window.hideEverything = hideEverything;
  window.unhide = unhide;
  window.unhideEverything = unhideEverything;
})();
"""

    _JS_HIDE_ALL = """
window.hideEverything();
"""

    _JS_UNHIDE_ALL = """
window.unhideEverything();
"""

    # Base observation space for all defenses
    BASE_OBS_SPACE = {
        "screenshot_censored": AnyBox(
            low=0,
            high=255,
            shape=(-1, -1, 3),
            dtype="uint8",
        ),
        "html_censored": Unicode(),
        "a11y_censored": Unicode(),
    }

    @classmethod
    def get_defense_id(cls) -> str:
        """Return a unique identifier for this defense type."""
        return cls.__name__

    @classmethod
    def obs_space(cls) -> Dict[str, Any]:
        """
        Define the observation space for this defense.
        """
        if space := getattr(cls, "DEFENSE_OBS_SPACE", None):
            return space | cls.BASE_OBS_SPACE

        return cls.BASE_OBS_SPACE

    def __init__(self, **kwargs) -> None:
        """Initialize the defense with any necessary parameters."""
        super().__init__()
        self._log = io.StringIO()
        self.logger = logging.getLogger(f"{self.__class__.__name__}-{id(self)}")
        handler = logging.StreamHandler(self._log)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.handlers = [handler]
        self.logger.setLevel(logging.DEBUG)
        self._log_pos = 0
        self._messages: list[dict] = []
        self._async_messages: list[dict] = []
        self._last_msg_idx = 0

    @property
    def log(self) -> logging.Logger:
        """Return the logger."""
        return self.logger

    @property
    def logs(self) -> str:
        """Return all log text."""
        return self._log.getvalue()

    @property
    def new_logs(self) -> str:
        """Return logs since the last call."""
        self._log.seek(self._log_pos)
        data = self._log.read()
        self._log_pos = self._log.tell()
        return data

    @property
    def messages(self) -> list[dict]:
        return self._messages

    @property
    def async_messages(self) -> list[dict]:
        return self._async_messages

    @property
    def async_messages_stats(self) -> dict:
        return {
            "total_async_requests": len(self._async_messages),
            "total_input_tokens": sum(msg["input_tokens"] for msg in self._async_messages),
            "total_output_tokens": sum(msg["output_tokens"] for msg in self._async_messages),
            "total_request_duration": sum(msg["request_duration"] for msg in self._async_messages),
            "first_request_time": self._async_messages[0]["request_start"] if self._async_messages else None,
            "last_request_time": self._async_messages[-1]["request_end"] if self._async_messages else None,
            "cached_requests": sum(1 for msg in self._async_messages if msg["cached"] is True),
            "cache_hit_rate": (
                sum(1 for msg in self._async_messages if msg["cached"] is True) / len(self._async_messages)
                if self._async_messages
                else 0
            ),
            "cached_input_tokens": sum(msg["input_tokens"] for msg in self._async_messages if msg["cached"] is True),
            "cached_output_tokens": sum(msg["output_tokens"] for msg in self._async_messages if msg["cached"] is True),
        }

    @property
    def new_messages(self) -> list[dict]:
        new = self._messages[self._last_msg_idx :]
        self._last_msg_idx = len(self._messages)
        return new

    def _wrap_openai_client(self, client: OpenAI) -> OpenAI:
        """Return a logging proxy around ``client``."""
        return cast(OpenAI, OpenAILoggingProxy(client, self._messages))

    def _wrap_async_openai_client(self, client: AsyncOpenAI) -> AsyncOpenAI:
        """Return a logging proxy around ``client``."""
        return cast(AsyncOpenAI, AsyncOpenAILoggingProxy(client, self._async_messages))

    def run_defense(self, observation: Dict[str, Any], page) -> Dict[str, Any]:
        """
        Run the defense logic on the current observation.

        Args:
            observation: The current environment observation containing DOM and accessibility tree data
            page: The browser page object used for capturing censored screenshots

        Returns:
            Dict containing additional observation fields with defense results
        """
        raise NotImplementedError

    @classmethod
    def to_json(cls) -> Dict[str, Any]:
        """Serialize the defense configuration to JSON."""
        data = {
            "defense_id": cls.get_defense_id(),
        }
        return data

    def _unhide_by_id_js(self, ids: list[str]) -> str:
        """Generate JavaScript to unhide elements with given IDs."""
        ids_str = ", ".join([f"{id}" for id in ids])
        js = f"window.unhide({ids_str});"
        self.logger.info(f"Unhiding elements with command: {js}")
        return js

    def _setup_censoring_logic(self, page) -> None:
        """Setup the JavaScript functions for hiding/unhiding elements."""
        page.evaluate(self._JS_SETUP_FUNCTIONS)
        time.sleep(1)

    def capture_censored_screenshot(self, page, allowed_bids: list[str]) -> np.ndarray:
        """
        Capture a screenshot with only allowed elements visible.

        Args:
            page: The browser page object
            allowed_bids: List of bid attributes for elements that should be visible

        Returns:
            Dict containing the censored screenshot
        """

        # Setup censoring functions if needed
        self._setup_censoring_logic(page)

        # Hide everything
        page.evaluate(self._JS_HIDE_ALL)
        time.sleep(1)

        # Unhide allowed elements
        unhide_js = self._unhide_by_id_js(allowed_bids)
        page.evaluate(unhide_js)
        time.sleep(1)

        # Capture censored screenshot
        screenshot_censored = extract_screenshot(page)

        # Restore original visibility
        page.evaluate(self._JS_UNHIDE_ALL)
        time.sleep(1)

        return screenshot_censored
