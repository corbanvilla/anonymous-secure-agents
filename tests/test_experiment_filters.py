import os
import uuid

from src.db.client import Session, safe_commit
from src.db.helpers.datasets import create_task_dataset_if_not_exists
from src.db.helpers.experiments import (
    create_experiment,
    filter_experiment_names,
    get_all_attacks,
    get_all_defenses,
    get_experiment_trajectories,
    load_recent_experiments,
)
from src.db.tables import Experiment
from src.experiments.config import get_experiment_config


def test_filter_exclude_tests():
    exps = load_recent_experiments(show_tests=False)
    assert exps
    for exp in exps:
        assert "test" not in exp.name.lower()


def test_filter_by_user():
    user = "user_one"
    exps = load_recent_experiments(show_tests=True, selected_user=user)
    assert exps
    for exp in exps:
        assert exp.username == user


def test_filter_by_model():
    model = "gpt-4.1-mini"
    exps = load_recent_experiments(show_tests=True, selected_model=model)
    assert exps
    for exp in exps:
        assert exp.config["engine_options"]["sampling_params"]["model"] == model


def test_filter_names_prefix_server_side(monkeypatch):
    name = f"test-{uuid.uuid4()}"
    config = get_experiment_config(
        name=name,
        description="test",
        tasks=[],
        engine_model="gpt-4.1-mini",
        webagent_src=["html"],
    ).model_dump()
    monkeypatch.setattr(os, "getlogin", lambda: "user_one")
    create_experiment(name, [], config)

    names = filter_experiment_names(show_tests=True, selected_user="user_one", prefix="test")
    assert name in names


def test_filter_user_and_model():
    user = "user_one"
    model = "gpt-4.1-mini"
    exps = load_recent_experiments(show_tests=True, selected_user=user, selected_model=model)
    assert exps
    for exp in exps:
        assert exp.username == user
        assert exp.config["engine_options"]["sampling_params"]["model"] == model


def test_filter_no_tests_and_model():
    model = "gpt-4.1-mini"
    exps = load_recent_experiments(show_tests=False, selected_model=model)
    assert exps
    for exp in exps:
        assert "test" not in exp.name.lower()
        assert exp.config["engine_options"]["sampling_params"]["model"] == model


def test_filter_no_tests_user_and_model():
    user = "user_one"
    model = "gpt-4.1-mini"
    exps = load_recent_experiments(show_tests=False, selected_user=user, selected_model=model)
    assert exps
    for exp in exps:
        assert "test" not in exp.name.lower()
        assert exp.username == user
        assert exp.config["engine_options"]["sampling_params"]["model"] == model


def test_filter_names_multiple_conditions(monkeypatch):
    user = "user_one"
    model = "gpt-4.1-mini"
    prefix = "VWA-"
    name = f"{prefix}{uuid.uuid4()}"
    config = get_experiment_config(
        name=name,
        description="test",
        tasks=[],
        engine_model=model,
        webagent_src=["html"],
    ).model_dump()
    monkeypatch.setattr(os, "getlogin", lambda: user)
    create_experiment(name, [], config)

    names = filter_experiment_names(
        show_tests=False,
        selected_user=user,
        selected_model=model,
        prefix=prefix,
    )
    assert name in names


def test_get_all_attacks_and_defenses():
    attacks = get_all_attacks()
    defenses = get_all_defenses()
    assert attacks
    assert defenses


def test_get_all_datasets():
    from src.db.helpers.datasets import get_all_datasets

    datasets = get_all_datasets()
    assert datasets


def test_filter_by_attack():
    names = filter_experiment_names(show_tests=True, selected_attack="eia")
    assert names
    for n in names:
        cfg, _ = get_experiment_trajectories(n)
        atk = cfg.env_args.attack
        assert str(atk) == "eia"


def test_filter_by_defense(monkeypatch):
    name = f"defense-test-{uuid.uuid4()}"
    config = get_experiment_config(
        name=name,
        description="test",
        tasks=[],
        engine_model="gpt-4.1-mini",
        webagent_src=["html"],
    ).model_dump()
    config["env_args"]["defense"] = {"defense_id": "one_stage_defense"}
    monkeypatch.setattr(os, "getlogin", lambda: "user_one")
    create_experiment(name, [], config)

    names = filter_experiment_names(show_tests=True, selected_defense="one_stage_defense")
    assert name in names
    for n in names:
        cfg, _ = get_experiment_trajectories(n)
        defense = cfg.env_args.defense
        assert str(defense).startswith("one_stage_defense")


def test_filter_by_dataset(monkeypatch):
    name = f"ds-test-{uuid.uuid4()}"
    dataset = f"ds-{uuid.uuid4()}"
    tasks = ["t1", "t2"]
    create_task_dataset_if_not_exists(dataset, tasks)

    config = get_experiment_config(
        name=name,
        description="test",
        tasks=tasks,
        engine_model="gpt-4.1",
        webagent_src=["html"],
    ).model_dump()
    monkeypatch.setattr(os, "getlogin", lambda: "testuser")
    create_experiment(name, tasks, config)
    names = filter_experiment_names(show_tests=True, selected_dataset=dataset)
    assert name in names


def test_show_hidden_and_favorites(monkeypatch):
    name = f"hidden-fav-{uuid.uuid4()}"
    config = get_experiment_config(
        name=name,
        description="test",
        tasks=[],
        engine_model="gpt-4.1",
        webagent_src=["html"],
    ).model_dump()
    monkeypatch.setattr(os, "getlogin", lambda: "testuser")
    create_experiment(name, [], config)
    with Session() as session:
        exp = session.query(Experiment).filter(Experiment.name == name).one()
        exp.hidden = True
        exp.favorite = True
        safe_commit(session)

    names_hidden = filter_experiment_names(show_tests=True)
    assert name not in names_hidden
    names_shown = filter_experiment_names(show_tests=True, show_hidden=True)
    assert name in names_shown
    names_fav = filter_experiment_names(show_tests=True, favorites_only=True, show_hidden=True)
    assert name in names_fav
