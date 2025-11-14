from __future__ import annotations

import json
import time
from typing import Any, List, Optional, cast

from src.db.client import Session, safe_commit
from src.db.config import LLM_LOG_DROP_FIELDS
from src.db.redis_client import (
    DEFENSE_SKIP_IDS_KEY,
    DEFENSE_STEP_KEY_PREFIX,
    DEFENSE_STEP_QUEUE,
    redis_client,
)
from src.db.tables import DefenseHarnessExperiment as DBDefenseHarnessExperiment
from src.db.tables import DefenseHarnessStep as DBDefenseHarnessStep
from src.log import log

from ..helpers.anti_timeout import timed_sql_execution


def store_defense_skip_ids(defense_experiment_id: int, skip_observation_ids: List[int]):
    if not skip_observation_ids:
        return
    skip_ids_key = f"{DEFENSE_SKIP_IDS_KEY}{defense_experiment_id}"
    skip_ids_str = [str(i) for i in skip_observation_ids]
    redis_client.rpush(skip_ids_key, *skip_ids_str)
    log.info(f"Stored {len(skip_observation_ids)} skip IDs for defense experiment {defense_experiment_id}")


def commit_defense_skip_ids() -> dict[int, int]:
    skip_ids_pattern = f"{DEFENSE_SKIP_IDS_KEY}*"
    skip_ids_keys = cast(List[str], redis_client.keys(skip_ids_pattern))
    if not skip_ids_keys:
        return {}

    results: dict[int, int] = {}
    for skip_ids_key in skip_ids_keys:
        try:
            defense_experiment_id = int(str(skip_ids_key).replace(DEFENSE_SKIP_IDS_KEY, ""))
            skip_ids_result = cast(List[str], redis_client.lrange(str(skip_ids_key), 0, -1))
            if not skip_ids_result:
                continue
            skip_ids = [int(s) for s in skip_ids_result]
            with Session() as session:
                defense_exp = session.query(DBDefenseHarnessExperiment).get(defense_experiment_id)
                if defense_exp:
                    existing = defense_exp.skip_observation_ids or []
                    all_skip_ids = list(set(existing + skip_ids))
                    defense_exp.skip_observation_ids = all_skip_ids
                    safe_commit(session)
                    redis_client.delete(str(skip_ids_key))
                    results[defense_experiment_id] = len(skip_ids)
                    log.info(f"Added {len(skip_ids)} skip IDs to defense experiment {defense_experiment_id}")
        except (ValueError, TypeError) as e:
            log.error(f"Error processing skip IDs key {skip_ids_key}: {e}")
            continue
    return results


def store_incremental_defense_steps(defense_experiment_id: int, defense_steps: List[dict]):
    for step in defense_steps:
        step_key = f"{DEFENSE_STEP_KEY_PREFIX}{defense_experiment_id}:{step['observation_id']}"
        step_data = {
            "defense_experiment_id": defense_experiment_id,
            "step": step,
            "timestamp": time.time(),
        }
        redis_client.set(step_key, json.dumps(step_data))
        redis_client.lpush(DEFENSE_STEP_QUEUE, step_key)
    log.info(f"Stored {len(defense_steps)} defense steps for experiment {defense_experiment_id}")


@timed_sql_execution(timeout_seconds=15)
def commit_incremental_defense_steps() -> Optional[str]:
    defense_step_key = redis_client.lindex(DEFENSE_STEP_QUEUE, -1)
    if not defense_step_key:
        return None

    defense_step_data_str = redis_client.get(str(defense_step_key))
    if not defense_step_data_str:
        log.warning(f"Defense step data not found for key {defense_step_key}")
        redis_client.lrem(DEFENSE_STEP_QUEUE, 0, str(defense_step_key))
        return None

    try:
        defense_step_data = json.loads(str(defense_step_data_str))
        defense_experiment_id = defense_step_data["defense_experiment_id"]
        step = defense_step_data["step"]
        llm_logs = step.get("llm_logs", [])
        for log_entry in llm_logs:
            for field in LLM_LOG_DROP_FIELDS:
                log_entry.pop(field, None)
        with Session() as session:
            defense_step = cast(
                Any,
                DBDefenseHarnessStep,
            )(
                defense_experiment_id=defense_experiment_id,
                observation_id=step["observation_id"],
                full_action=step["full_action"],
                function=step["function"],
                required_bid=step["required_bid"],
                allowed_bids=step["allowed_bids"],
                all_bids=step["all_bids"],
                error_message=step.get("error_message"),
                llm_logs=llm_logs,
                relevant_cap_set=step["relevant_cap_set"],
                async_messages_stats=step["async_messages_stats"],
            )
            session.add(defense_step)
            safe_commit(session)
            redis_client.rpop(DEFENSE_STEP_QUEUE)
            redis_client.delete(str(defense_step_key))
            log.info(f"Successfully committed defense step {defense_step_key} to database")
            return defense_step_key
    except Exception as e:
        log.error(f"Error committing defense step {defense_step_key} to database: {e}")
        raise
