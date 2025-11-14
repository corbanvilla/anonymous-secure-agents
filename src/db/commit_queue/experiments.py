from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import asdict
from typing import Any, List, Optional, cast

import numpy as np
from PIL import Image as PILImage
from rllm.agents.agent import Trajectory

from src.db.client import Session, safe_commit
from src.db.config import (
    DROP_FIELDS,
    LARGE_FILES_DIR,
    SCREENSHOT_DIR,
    STORE_FILE_FIELD_PREFIXES,
    STORE_SCREENSHOT_FIELDS,
)
from src.db.redis_client import (
    TRAJECTORY_KEY_PREFIX,
    TRAJECTORY_QUEUE,
    redis_client,
)
from src.db.tables import Observation as DBObservation
from src.db.tables import Trajectory as DBTrajectoryModel
from src.log import log

from ..helpers.anti_timeout import timed_sql_execution
from ..helpers.experiments import get_experiment_id


def store_trajectories_in_db(
    experiment_name: str,
    task_ids: List[str],
    trajectories: List[Trajectory],
    chat_completions: Optional[List[Any]] = None,
    evaluation_results: Optional[dict] = None,
) -> int:
    log.info(
        "Storing %s trajectory(ies) for experiment %s in Redis",
        len(trajectories),
        experiment_name,
    )
    if len(trajectories) != len(task_ids):
        raise ValueError(
            f"Number of trajectories ({len(trajectories)}) must match number of task IDs ({len(task_ids)})"
        )
    if chat_completions is not None and len(chat_completions) != len(task_ids):
        raise ValueError(
            f"Number of chat_completions ({len(chat_completions)}) must match number of task IDs ({len(task_ids)})"
        )

    exp_id = get_experiment_id(experiment_name)
    if exp_id is None:
        raise ValueError(f"Experiment with name {experiment_name!r} does not exist")

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert(v) for v in obj]
        else:
            return obj

    for i, (traj, task_id) in enumerate(zip(trajectories, task_ids)):
        traj_dict = asdict(traj)
        obs_entries = []
        for step in traj_dict.get("steps", []):
            step.pop("next_observation", None)
            obs = step.pop("observation", {})

            for screenshot_key in STORE_SCREENSHOT_FIELDS:
                if screenshot_key in obs:
                    img = obs[screenshot_key]
                    if isinstance(img, np.ndarray):
                        pil_img = PILImage.fromarray(img)
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        img_bytes = buf.getvalue()
                    else:
                        img_bytes = img
                    digest = hashlib.sha256(img_bytes).hexdigest()
                    filename = f"{digest}.png"
                    filepath = SCREENSHOT_DIR / filename
                    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
                    if not filepath.exists():
                        filepath.write_bytes(img_bytes)
                    obs[screenshot_key] = filename

            for k in DROP_FIELDS:
                obs.pop(k, None)

            stored_keys = []
            for prefix in STORE_FILE_FIELD_PREFIXES:
                for key in list(obs.keys()):
                    if key.startswith(prefix):
                        filename = hashlib.sha256(json.dumps(obs[key]).encode()).hexdigest()
                        filepath = LARGE_FILES_DIR / f"{key}_{filename}.json"
                        LARGE_FILES_DIR.mkdir(parents=True, exist_ok=True)
                        if not filepath.exists():
                            filepath.write_text(json.dumps(obs[key]))
                        obs[key] = filepath.name
                        stored_keys.append(key)
            obs["__stored_keys__"] = stored_keys
            obs_entries.append(_convert(obs))

        traj_json = _convert(traj_dict)
        chat = _convert(chat_completions[i]) if chat_completions is not None else None

        trajectory_key = f"{TRAJECTORY_KEY_PREFIX}{exp_id}:{task_id}"
        trajectory_data = {
            "experiment_id": exp_id,
            "task_id": task_id,
            "trajectory": traj_json,
            "evaluation_results": evaluation_results,
            "chat": chat,
            "observations": obs_entries,
            "timestamp": time.time(),
        }
        redis_client.set(trajectory_key, json.dumps(trajectory_data))
        redis_client.lpush(TRAJECTORY_QUEUE, trajectory_key)

    log.info(
        "Successfully stored %s trajectory(ies) in Redis for experiment %s",
        len(trajectories),
        experiment_name,
    )
    return exp_id


@timed_sql_execution(timeout_seconds=15)
def commit_latest_records_to_db() -> Optional[str]:
    trajectory_key = redis_client.lindex(TRAJECTORY_QUEUE, -1)
    if not trajectory_key:
        return None

    trajectory_data_str = redis_client.get(trajectory_key)
    if not trajectory_data_str:
        log.warning(f"Trajectory data not found for key {trajectory_key}")
        redis_client.lrem(TRAJECTORY_QUEUE, 0, trajectory_key)
        return None

    try:
        trajectory_data = json.loads(trajectory_data_str)
        with Session() as session:
            traj_record = cast(
                Any,
                DBTrajectoryModel,
            )(
                experiment_id=trajectory_data["experiment_id"],
                task_id=trajectory_data["task_id"],
                trajectory=trajectory_data["trajectory"],
                chat=trajectory_data["chat"],
            )
            session.add(traj_record)
            session.flush()

            for step_number, obs_data in enumerate(trajectory_data["observations"]):
                obs_record = cast(
                    Any,
                    DBObservation,
                )(data=obs_data, trajectory_id=traj_record.id, step_number=step_number)
                session.add(obs_record)

            safe_commit(session)
            redis_client.rpop(TRAJECTORY_QUEUE)
            redis_client.delete(trajectory_key)
            log.info(f"Successfully committed trajectory {trajectory_key} to database")
            return trajectory_key
    except Exception as e:
        log.error(f"Error committing trajectory {trajectory_key} to database: {e}")
        raise
