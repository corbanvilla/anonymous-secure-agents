from __future__ import annotations

from typing import List, Tuple

from sqlalchemy.sql import any_

from src.db.client import Session
from src.db.tables import DefenseHarnessExperiment as DBDefenseHarnessExperiment
from src.db.tables import DefenseHarnessStep as DBDefenseHarnessStep
from src.db.tables import Experiment as DBExperiment
from src.db.tables import Observation as DBObservation
from src.db.tables import Trajectory as DBTrajectoryModel

from .anti_timeout import timed_sql_execution


@timed_sql_execution(timeout_seconds=30)
def get_missing_defense_evals(defense_experiment_id: int) -> List[Tuple[int, str]]:
    with Session() as session:
        defense_exp = (
            session.query(DBDefenseHarnessExperiment)
            .filter(DBDefenseHarnessExperiment.id == defense_experiment_id)
            .one()
        )
        query = (
            session.query(
                DBObservation.id,
                DBTrajectoryModel.trajectory["steps"][DBObservation.step_number]["action"].astext,
            )
            .join(DBTrajectoryModel, DBObservation.trajectory_id == DBTrajectoryModel.id)
            .join(DBExperiment, DBTrajectoryModel.experiment_id == DBExperiment.id)
            .filter(DBExperiment.id == defense_exp.reference_experiment_id)
            .filter(DBTrajectoryModel.success)
            .outerjoin(
                DBDefenseHarnessStep,
                (DBDefenseHarnessStep.observation_id == DBObservation.id)
                & (DBDefenseHarnessStep.defense_experiment_id == defense_experiment_id),
            )
            .filter(DBDefenseHarnessStep.id.is_(None))
            .filter(~DBObservation.id.op("=")(any_(defense_exp.skip_observation_ids)))
            .order_by(DBObservation.id)
            .distinct()
        )
        return [(row[0], row[1]) for row in query.all()]
