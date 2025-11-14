import logging
from typing import Any, Dict

from src.db.helpers.logging import DBErrorHandler


# ANSI color codes
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages based on level."""

    COLORS = {
        "DEBUG": Colors.BLUE,
        "INFO": Colors.GREEN,
        "WARNING": Colors.YELLOW,
        "ERROR": Colors.RED,
        "CRITICAL": Colors.MAGENTA,
    }

    def format(self, record):
        # Add color to the level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{Colors.RESET}"
        return super().format(record)


def setup_logging():
    """Configure logging with colors and proper formatting."""

    # Create formatters
    console_formatter = ColoredFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Use standard formatter without colors for database
    db_formatter = logging.Formatter("%(name)s: %(message)s")

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Add console handler if none exists
    if not root_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # Route errors to the database
    db_handler = DBErrorHandler()
    db_handler.setFormatter(db_formatter)
    root_logger.addHandler(db_handler)

    # Silence httpx
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)

    return root_logger


def log_config_differences(differences: Dict[str, Dict[str, Any]]):
    """Log configuration differences with colors."""
    logger = logging.getLogger(__name__)
    for path, diff in differences.items():
        logger.error(f"  {path}:")
        logger.error(f"    Current: {Colors.GREEN}{diff['config1']}{Colors.RESET}")
        logger.error(f"    Stored:  {Colors.RED}{diff['config2']}{Colors.RESET}")


# Initialize logging when module is imported
log = setup_logging()
