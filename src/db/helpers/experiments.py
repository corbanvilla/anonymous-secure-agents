from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, cast

from sqlalchemy.sql import and_

from src.db.client import Session, safe_commit
from src.db.tables import Experiment as DBExperiment
from src.db.tables import TaskDataset as DBTaskDataset
from src.db.tables import Trajectory as DBTrajectoryModel
from src.experiments.config import ExperimentConfig

from .anti_timeout import timed_sql_execution

_experiment_name_to_id_cache: Dict[str, int] = {}


@timed_sql_execution(timeout_seconds=30)
def get_experiment_id(experiment_name: str) -> Optional[int]:
    if experiment_name in _experiment_name_to_id_cache:
        return _experiment_name_to_id_cache[experiment_name]
    with Session() as session:
        exp_id = session.query(DBExperiment.id).filter(DBExperiment.name == experiment_name).scalar()
        if exp_id is None:
            return None
        _experiment_name_to_id_cache[experiment_name] = exp_id
        return exp_id


@timed_sql_execution(timeout_seconds=15)
def create_experiment(
    experiment_name: str,
    task_ids: List[str],
    config: dict,
    description: Optional[str] = None,
) -> int:
    config_serialized = json.loads(
        json.dumps(
            config,
            default=lambda o: o.to_json() if hasattr(o, "to_json") else str(o),
            sort_keys=True,
        )
    )
    username = os.getlogin()
    with Session() as session:
        exp = cast(
            Any,
            DBExperiment,
        )(
            name=experiment_name,
            username=username,
            config=config_serialized,
            task_ids=task_ids,
            description=description,
        )
        session.add(exp)
        session.flush()
        safe_commit(session)
        exp_id = exp.id
        _experiment_name_to_id_cache[experiment_name] = exp_id
        return exp_id


@timed_sql_execution(timeout_seconds=15)
def update_experiment_status(experiment_name: str, is_running: bool) -> None:
    with Session() as session:
        exp = cast(Any, session.query(DBExperiment).filter(DBExperiment.name == experiment_name).one())
        exp.is_running = is_running
        safe_commit(session)


def get_running_experiments():
    with Session() as session:
        return session.query(DBExperiment).filter(DBExperiment.is_running).one_or_none()


def load_experiment_by_name(experiment_name: str):
    with Session() as session:
        return session.query(DBExperiment).filter(DBExperiment.name == experiment_name).one_or_none()


def load_recent_experiments(
    show_tests: bool = True,
    selected_user: str | None = None,
    selected_model: str | None = None,
):
    with Session() as session:
        query = session.query(DBExperiment)
        if not show_tests:
            query = query.filter(~DBExperiment.name.ilike("%test%"))
        if selected_user and selected_user != "All":
            query = query.filter(DBExperiment.username == selected_user)
        if selected_model and selected_model != "All":
            model_field = DBExperiment.config["engine_options"]["sampling_params"]["model"].astext
            query = query.filter(model_field == selected_model)
        query = query.order_by(DBExperiment.created_at.desc())
        return query.all()


def filter_experiment_names(
    show_tests: bool = True,
    selected_user: str | None = None,
    selected_model: str | None = None,
    selected_attack: str | None = None,
    selected_defense: str | None = None,
    prefix: str | None = None,
    selected_dataset: str | None = None,
    show_hidden: bool = False,
    favorites_only: bool = False,
) -> List[str]:
    with Session() as session:
        dataset_cond = and_(
            DBExperiment.task_ids.op("<@")(DBTaskDataset.tasks),
            DBTaskDataset.tasks.op("<@")(DBExperiment.task_ids),
        )
        query = session.query(DBExperiment.name).select_from(DBExperiment).outerjoin(DBTaskDataset, dataset_cond)
        if not show_tests:
            query = query.filter(~DBExperiment.name.ilike("%test%"))
        if not show_hidden:
            query = query.filter(DBExperiment.hidden.is_(False))
        if favorites_only:
            query = query.filter(DBExperiment.favorite.is_(True))
        if selected_user and selected_user != "All":
            query = query.filter(DBExperiment.username == selected_user)
        if selected_model and selected_model != "All":
            model_field = DBExperiment.config["engine_options"]["sampling_params"]["model"].astext
            query = query.filter(model_field == selected_model)
        if prefix:
            query = query.filter(DBExperiment.name.ilike(f"%{prefix}%"))
        if selected_dataset and selected_dataset != "All":
            if selected_dataset == "Unknown":
                query = query.filter(DBTaskDataset.name.is_(None))
            else:
                query = query.filter(DBTaskDataset.name == selected_dataset)
        attack_field = DBExperiment.config["env_args"]["attack"]
        defense_field = DBExperiment.config["env_args"]["defense"]
        if selected_attack and selected_attack != "Any":
            if selected_attack == "None":
                query = query.filter(attack_field.is_(None))
            else:
                query = query.filter(
                    (attack_field["attack_id"].astext == selected_attack)
                    | (attack_field["type"].astext == selected_attack)
                    | (attack_field.astext == selected_attack)
                )
        if selected_defense and selected_defense != "Any":
            if selected_defense == "None":
                query = query.filter(defense_field.is_(None))
            else:
                query = query.filter(
                    (defense_field["defense_id"].astext == selected_defense)
                    | (defense_field["type"].astext == selected_defense)
                    | (defense_field.astext == selected_defense)
                )
        query = query.order_by(DBExperiment.created_at.desc())
        return [name for (name,) in query.all()]


def get_experiment_trajectories(experiment_name: str):
    with Session() as session:
        exp = session.query(DBExperiment).filter(DBExperiment.name == experiment_name).one_or_none()
        if exp is None:
            return None, None
        trajs = (
            session.query(DBTrajectoryModel.task_id, DBTrajectoryModel.id)
            .filter(DBTrajectoryModel.experiment_id == exp.id)
            .all()
        )
        traj_map = {task_id: traj_id for task_id, traj_id in trajs}
        config = ExperimentConfig.model_validate(exp.config)
        task_ids_list = cast(List[str], exp.task_ids)
        return config, {task_id: traj_map.get(task_id) for task_id in task_ids_list}


def _extract_attack_id(attack: Any) -> str | None:
    if attack is None:
        return None
    if isinstance(attack, dict):
        return attack.get("attack_id") or attack.get("type")
    return str(attack)


def _extract_defense_id(defense: Any) -> str | None:
    if defense is None:
        return None
    if isinstance(defense, dict):
        return defense.get("defense_id") or defense.get("type")
    return str(defense)


def get_all_attacks() -> List[str]:
    with Session() as session:
        query = session.query(DBExperiment.config["env_args"]["attack"]).distinct()
        ids: set[str] = set()
        for (attack,) in query.all():
            aid = _extract_attack_id(attack)
            if aid:
                ids.add(aid)
        return sorted(ids)


def get_all_defenses() -> List[str]:
    with Session() as session:
        query = session.query(DBExperiment.config["env_args"]["defense"]).distinct()
        ids: set[str] = set()
        for (defense,) in query.all():
            did = _extract_defense_id(defense)
            if did:
                ids.add(did)
        return sorted(ids)
