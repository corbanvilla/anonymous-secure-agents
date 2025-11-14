from src.db.helpers.experiments import (
    filter_experiment_names,
    get_experiment_trajectories,
)
from src.experiments.config import ExperimentConfig


def test_all_experiment_configs_load():
    names = filter_experiment_names(show_tests=True)
    assert names
    for name in names:
        cfg, _ = get_experiment_trajectories(name)
        assert isinstance(cfg, ExperimentConfig)
