from src.experiments.config.utils import generate_defense_harness_experiment_name


def test_generate_defense_harness_name():
    name = generate_defense_harness_experiment_name("base", "one_stage", "gpt4")
    assert name == "base-one_stage-gpt4"


def test_generate_defense_harness_name_timestamp():
    name = generate_defense_harness_experiment_name("base", "one_stage", "gpt4", timestamp=True)
    assert name.startswith("base-one_stage-gpt4-")
