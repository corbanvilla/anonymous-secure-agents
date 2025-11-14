from __future__ import annotations

from typing import Any, List, Optional, cast

from src.db.client import Session, safe_commit
from src.db.tables import TaskDataset as DBTaskDataset

from .anti_timeout import timed_sql_execution


@timed_sql_execution(timeout_seconds=15)
def create_task_dataset_if_not_exists(name: str, tasks: List[str]) -> None:
    with Session() as session:
        existing = session.query(DBTaskDataset).filter(DBTaskDataset.name == name).one_or_none()
        if existing is None:
            ds = cast(Any, DBTaskDataset)(name=name, tasks=tasks)
            session.add(ds)
            safe_commit(session)


def get_all_datasets() -> List[str]:
    with Session() as session:
        query = session.query(DBTaskDataset.name).order_by(DBTaskDataset.name.asc())
        return [name for (name,) in query.all()]


def get_dataset_name_for_task_ids(task_ids: List[str]) -> Optional[str]:
    with Session() as session:
        datasets = session.query(DBTaskDataset.name, DBTaskDataset.tasks).all()

    task_set = set(task_ids)
    for name, tasks in datasets:
        if set(tasks) == task_set:
            return name
    return None
