import uuid

from src.db.client import Session
from src.db.helpers.datasets import create_task_dataset_if_not_exists
from src.db.tables import TaskDataset


def test_create_task_dataset_if_not_exists():
    name = f"test-ds-{uuid.uuid4()}"
    tasks = ["t1", "t2"]

    create_task_dataset_if_not_exists(name, tasks)
    create_task_dataset_if_not_exists(name, tasks)

    with Session() as session:
        records = session.query(TaskDataset).filter(TaskDataset.name == name).all()
        assert len(records) == 1
        assert records[0].tasks == tasks
