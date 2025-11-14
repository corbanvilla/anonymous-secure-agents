import uuid

from src.db.helpers.trajectories import load_experiment_trajectories


def test_load_experiment_trajectories_not_found():
    name = f"nonexistent-{uuid.uuid4()}"
    exp, valid_ids, incomplete, success, failed, rate = load_experiment_trajectories(name)
    assert exp is None
    assert valid_ids == []
    assert incomplete == []
    assert success == []
    assert failed == []
    assert rate == 0.0
