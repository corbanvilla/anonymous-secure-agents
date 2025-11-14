from __future__ import annotations

from src.db.redis_client import (
    DEFENSE_SKIP_IDS_KEY,
    DEFENSE_STEP_QUEUE,
    TRAJECTORY_QUEUE,
    redis_client,
)


def get_pending_records_count() -> int:
    """Return the number of records waiting to be committed from Redis."""
    return (
        redis_client.llen(TRAJECTORY_QUEUE)
        + redis_client.llen(DEFENSE_STEP_QUEUE)
        + sum(redis_client.llen(key) for key in redis_client.keys(f"{DEFENSE_SKIP_IDS_KEY}*"))
    )
