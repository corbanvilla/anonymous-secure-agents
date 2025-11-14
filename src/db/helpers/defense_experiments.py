from __future__ import annotations

import json
import os
from typing import Any, Optional, cast

from src.db.client import Session, safe_commit
from src.db.tables import DefenseHarnessExperiment as DBDefenseHarnessExperiment
from src.experiments.config.models import DefenseHarnessExperimentConfig

from .anti_timeout import timed_sql_execution


@timed_sql_execution(timeout_seconds=30)
def get_defense_experiment_id(experiment_name: str) -> tuple[Optional[int], Optional[DefenseHarnessExperimentConfig]]:
    with Session() as session:
        exp = (
            session.query(DBDefenseHarnessExperiment.id, DBDefenseHarnessExperiment.config)
            .filter(DBDefenseHarnessExperiment.name == experiment_name)
            .one_or_none()
        )
        if exp is None:
            return None, None
        config = DefenseHarnessExperimentConfig.model_validate(exp.config)
        return exp.id, config


@timed_sql_execution(timeout_seconds=30)
def create_defense_experiment(
    experiment_name: str,
    reference_experiment_id: int,
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
            DBDefenseHarnessExperiment,
        )(
            name=experiment_name,
            username=username,
            config=config_serialized,
            reference_experiment_id=reference_experiment_id,
            skip_observation_ids=[],
            description=description,
        )
        session.add(exp)
        session.flush()
        safe_commit(session)
        return exp.id
