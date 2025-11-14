from __future__ import annotations

import json
from typing import Any, List, Optional, Tuple, cast

import numpy as np
from PIL import Image as PILImage
from rllm.agents.agent import Step, Trajectory
from sqlalchemy.orm import joinedload

from src.db.client import Session
from src.db.config import LARGE_FILES_DIR, SCREENSHOT_DIR, STORE_SCREENSHOT_FIELDS
from src.db.tables import Experiment as DBExperiment
from src.db.tables import Observation as DBObservation
from src.db.tables import Trajectory as DBTrajectoryModel


def load_experiment_trajectories(
    experiment_name: str,
) -> Tuple[Optional[DBExperiment], List[str], List[str], List[str], List[str], float]:
    with Session() as session:
        exp = (
            session.query(DBExperiment)
            .filter(DBExperiment.name == experiment_name)
            .options(cast(Any, joinedload(DBExperiment.trajectories)))
            .one_or_none()
        )
    if exp is None:
        return None, [], [], [], [], 0.0

    valid_task_ids = {traj.task_id for traj in exp.trajectories}
    all_task_ids = set(exp.task_ids)
    incomplete_task_ids = list(all_task_ids - valid_task_ids)

    success_task_ids = []
    failed_task_ids = []
    for traj in exp.trajectories:
        if traj.success:
            success_task_ids.append(traj.task_id)
        else:
            failed_task_ids.append(traj.task_id)

    success_rate = len(success_task_ids) / len(valid_task_ids) if valid_task_ids else 0.0

    return (
        exp,
        list(valid_task_ids),
        list(incomplete_task_ids),
        success_task_ids,
        failed_task_ids,
        success_rate,
    )


def load_trajectory(
    experiment_name: str,
    task_id: str,
    load_screenshots: bool = True,
    load_large_files: bool = False,
) -> Tuple[Trajectory, Optional[str]]:
    with Session() as session:
        query = (
            session.query(DBTrajectoryModel)
            .join(DBExperiment)
            .filter(DBExperiment.name == experiment_name, DBTrajectoryModel.task_id == task_id)
        )
        query = query.options(cast(Any, joinedload(DBTrajectoryModel.observations)))
        traj_record = query.one()

    traj_json = traj_record.trajectory
    obs_models = sorted(traj_record.observations, key=lambda o: o.id)

    steps = []
    for step_dict, obs_model in zip(traj_json.get("steps", []), obs_models):
        obs_data = obs_model.data.copy()
        stored_keys = obs_data.pop("__stored_keys__", [])
        if load_large_files:
            for key in stored_keys:
                file = obs_data[key]
                path = LARGE_FILES_DIR / file
                if path.exists():
                    with open(path, "r") as f:
                        obs_data[key] = json.load(f)
        if load_screenshots:
            for key in STORE_SCREENSHOT_FIELDS:
                filename = obs_data.get(key)
                if isinstance(filename, str):
                    path = SCREENSHOT_DIR / filename
                    if path.exists():
                        img = PILImage.open(path)
                        obs_data[key] = np.array(img)
        step = Step(
            observation=obs_data,
            thought=step_dict.get("thought", ""),
            action=step_dict.get("action"),
            reward=step_dict.get("reward", 0.0),
            done=step_dict.get("done", False),
            info=step_dict.get("info", {}),
            step=step_dict.get("step", 0),
            model_response=step_dict.get("model_response", ""),
            mc_return=step_dict.get("mc_return", 0.0),
        )
        steps.append(step)

    traj_obj = Trajectory(
        steps=steps,
        reward=traj_json.get("reward", 0.0),
        termination_reason=traj_json.get("termination_reason"),
    )

    chat_log: Optional[str] = cast(Optional[str], traj_record.chat)
    return traj_obj, chat_log


def load_observation(obs_id: int, load_screenshots: bool = True, load_large_files: bool = False) -> dict:
    if load_screenshots:
        raise NotImplementedError("Loading screenshots is not yet implemented for single observations")
    with Session() as session:
        obs_model = session.query(DBObservation).filter(DBObservation.id == obs_id).one()
        obs_data = dict(cast(dict, obs_model.data)).copy()
        stored_keys = obs_data.pop("__stored_keys__", [])
        if load_large_files:
            for key in stored_keys:
                file = obs_data[key]
                path = LARGE_FILES_DIR / file
                if path.exists():
                    with open(path, "r") as f:
                        obs_data[key] = json.load(f)
        return obs_data
