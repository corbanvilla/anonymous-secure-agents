import logging

from src.db.client import Session
from src.db.tables import ErrorLog
from src.log import log  # noqa: F401 - initializes logging


def test_error_logging_stores_record():
    logger = logging.getLogger("test.db")
    logger.error("handler test")

    with Session() as session:
        entry = session.query(ErrorLog).order_by(ErrorLog.id.desc()).first()

    assert entry.message.endswith("handler test")
