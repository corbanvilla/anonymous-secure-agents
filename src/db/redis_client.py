import os

import redis

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB = int(os.environ.get("REDIS_DB", 0))

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True,
    encoding="utf-8",
)

TRAJECTORY_KEY_PREFIX = "trajectory:"
TRAJECTORY_QUEUE = "trajectory_queue"
DEFENSE_STEP_KEY_PREFIX = "defense_step:"
DEFENSE_STEP_QUEUE = "defense_step_queue"
DEFENSE_SKIP_IDS_KEY = "defense_skip_ids:"
