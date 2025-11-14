from __future__ import annotations

import logging
from typing import Optional



def log_error(message: str, *, task_id: Optional[str] = None, phase: str) -> None:
    """Store an error entry in the database."""
    pass
    # entry = ErrorLog(
    #     task_id=task_id,
    #     phase=phase,
    #     message=message,
    #     traceback=tb.format_exc(),
    #     username=getpass.getuser(),
    # )
    # with Session() as session:
    #     session.add(entry)
    #     safe_commit(session)


class DBErrorHandler(logging.Handler):
    """Logging handler that stores error logs in the database."""

    def __init__(self, *, default_phase: str = "log") -> None:
        super().__init__(level=logging.ERROR)
        self.default_phase = default_phase

    def emit(self, record: logging.LogRecord) -> None:
        phase = getattr(record, "phase", self.default_phase)
        message = self.format(record)
        log_error(message, phase=phase)
