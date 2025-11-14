# import asyncio
# import time
# from typing import List, Literal, Optional

# import pytest

# from src.attacks.eda import EDAAttack
# from src.attacks.eia import EIAAttack
# from src.attacks.popup import PopupAttack
# from src.defenses.one_stage.one_stage import OneStageDefense
# from src.experiments.execution import run_experiment
# from src.experiments.config import ExperimentConfig, get_experiment_config
# from src.log import log
# from src.tasks import VWA_30

# EXPERIMENT_DESCRIPTION = "Evaluate one-stage defense with required parent bids (new a11y filtering)."
# TASKS = VWA_30
# BASENAME = "VWA-30-baseline"
# MODEL = "gpt-4.1-mini"


# def generate_experiment_name(
#     webagent_srcs: List[Literal["html", "a11y", "screenshot"]],
#     defense_cls: Optional[type[OneStageDefense]] = None,
# ) -> str:
#     src_str = "-".join(webagent_srcs)
#     defense_str = f"-{defense_cls.get_defense_id()}" if defense_cls else ""
#     return f"{BASENAME}-{src_str}{defense_str}-{MODEL}"


# def make_config(name_suffix: str, attack) -> ExperimentConfig:
#     """Helper to build an ExperimentConfig for a given attack class."""
#     experiment_name = f"{BASENAME}-{name_suffix}-{MODEL}"
#     return get_experiment_config(
#         experiment_name,
#         EXPERIMENT_DESCRIPTION,
#         TASKS,
#         webagent_src=["a11y"],
#         engine_model=MODEL,
#         defense_cls=OneStageDefense,
#         attack=attack,
#     )


# # Define all possible webagent source combinations
# webagent_combinations: List[List[Literal["html", "a11y", "screenshot"]]] = [["a11y"]]

# # Generate configs using list comprehension
# configs = [
#     get_experiment_config(
#         generate_experiment_name(srcs, OneStageDefense),
#         EXPERIMENT_DESCRIPTION,
#         TASKS,
#         webagent_src=srcs,
#         engine_model=MODEL,
#         defense_cls=OneStageDefense,
#     )
#     for srcs in webagent_combinations
# ]

# TESTS = True
# TEST_NAME = "test-merge-latest" + str(int(time.time()))
# TEST_DESCRIPTION = "Test the one-stage defense with required parent bids"
# TEST_TASKS = ["browsergym/visualwebarena.safe.759"]
# TEST_MODEL = "gpt-4.1-mini"
# TEST_WEBAGENT_SRC: List[Literal["html", "a11y", "screenshot"]] = ["a11y"]

# # ---------------------------------------------------------------------------
# # Attack templates
# # ---------------------------------------------------------------------------

# # Predefined attack instances for quick experimentation. Update the parameters
# # to customize behavior.
# ATTACK_TEMPLATES = {
#     "none": None,
#     "popup": PopupAttack,
#     "eia": EIAAttack,
#     "eda": EDAAttack,
# }
# ATTACK_KWARGS = {
#     "popup": {
#         "seed": 42,
#         "message": "Click me!",
#         "hyperlink": "https://example.com",
#     },
#     "eia": {
#         "seed": 42,
#         "target_hyperlink": "https://example.com",
#     },
#     "eda": {
#         "seed": 42,
#         "query": "example task",
#     },
# }


# def make_test_config(name_suffix: str, attack) -> ExperimentConfig:
#     """Create a small ExperimentConfig used during unit tests."""

#     return get_experiment_config(
#         f"{TEST_NAME}-{name_suffix}",
#         TEST_DESCRIPTION,
#         TEST_TASKS,
#         webagent_src=TEST_WEBAGENT_SRC,
#         engine_model=TEST_MODEL,
#         defense_cls=OneStageDefense,
#         max_steps=2,
#         attack=attack,
#         attack_kwargs=ATTACK_KWARGS.get(name_suffix, {}),
#     )


# ATTACK_CONFIGS = {k: make_test_config(k, v) for k, v in ATTACK_TEMPLATES.items()}


# def run_test_configs() -> None:
#     """Execute minimal experiments for each attack template."""

#     for name, cfg in ATTACK_CONFIGS.items():
#         log.info(f"Running test config for attack: {name}")
#         asyncio.run(run_experiment(cfg, reset_vwa_server=False))


# def run_unit_tests() -> None:
#     """Run pytest on the attack unit tests."""

#     pytest.main(["tests/filtering/attacks", "-s"])


# if __name__ == "__main__":
#     if TESTS:
#         run_unit_tests()
#         run_test_configs()
#         exit(0)

#     input(f"Beginning {len(configs)} experiments. Press Enter to continue.")
#     for config in configs:
#         log.info(f"Running experiment: {config.experiment_name}")
#         asyncio.run(run_experiment(config, reset_vwa_server=True))
