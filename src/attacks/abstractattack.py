import io
import logging
from abc import ABC
from typing import Any, Dict, Optional, cast

from browsergym.core.spaces import AnyBox, Unicode
from openai import OpenAI

from src.api_clients.logging import OpenAILoggingProxy


class AbstractAttack(ABC):
    """Base class for all attack implementations."""

    BASE_OBS_SPACE = {
        "html_attack": Unicode(),
        "a11y_attack": Unicode(),
        "screenshot_attack": AnyBox(low=0, high=255, shape=(-1, -1, 3), dtype="uint8"),
    }

    def __init__(self, **_: object) -> None:
        super().__init__()
        self._log = io.StringIO()
        self.logger = logging.getLogger(f"{self.__class__.__name__}-{id(self)}")
        handler = logging.StreamHandler(self._log)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.handlers = [handler]
        self.logger.setLevel(logging.DEBUG)
        self._log_pos = 0
        self._messages: list[dict] = []
        self._last_msg_idx = 0

    @classmethod
    def obs_space(cls) -> Dict[str, Any]:
        """Define the observation space for this attack."""
        if space := getattr(cls, "ATTACK_OBS_SPACE", None):
            return space | cls.BASE_OBS_SPACE

        return cls.BASE_OBS_SPACE

    @property
    def log(self) -> logging.Logger:
        """Return the logger."""
        return self.logger

    @property
    def logs(self) -> str:
        """Return all log text."""
        return self._log.getvalue()

    @property
    def attack_executed_last_step(self) -> bool:
        """Return whether the attack was executed in the last step."""
        raise NotImplementedError

    @property
    def new_logs(self) -> str:
        """Return logs since the last call."""
        self._log.seek(self._log_pos)
        data = self._log.read()
        self._log_pos = self._log.tell()
        return data

    @property
    def messages(self) -> list[dict]:
        """Return all recorded OpenAI requests."""
        return self._messages

    @property
    def new_messages(self) -> list[dict]:
        """Return messages since the last call."""
        new = self._messages[self._last_msg_idx :]
        self._last_msg_idx = len(self._messages)
        return new

    def _wrap_openai_client(self, client: OpenAI) -> OpenAI:
        """Return a logging proxy around ``client``."""
        return cast(OpenAI, OpenAILoggingProxy(client, self._messages))

    @classmethod
    def get_attack_id(cls) -> str:
        """Return a unique identifier for the attack."""
        return cls.__name__

    @classmethod
    def to_json(cls) -> dict:
        """Serialize the attack configuration to a JSON compatible dict."""
        return {
            "attack_id": cls.get_attack_id(),
        }

    def step(self, page, observation: dict) -> bool:
        """
        Execute one step of the attack.

        Args:
            page: The browser page object
            observation: The current environment observation

        Returns:
            bool: Whether the attack was executed in this step
        """
        raise NotImplementedError

    def get_attack_bid(self, page) -> Optional[str]:
        """
        Get the bid attribute of the element that was attacked.

        Args:
            page: The browser page object

        Returns:
            Optional[str]: The bid attribute of the attacked element if found
        """
        raise NotImplementedError

    def attack_success(self, observation: dict, history: list[dict], page) -> bool:
        """Return True if the attack was successful, False otherwise.

        Args:
            observation: The current environment observation
            history: List of dictionaries containing previous observations and actions
            page: The browser page object

        Returns:
            bool: Whether the attack was successful
        """
        raise NotImplementedError
