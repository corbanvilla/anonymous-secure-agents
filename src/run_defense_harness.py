from pprint import pprint

from src.experiments.config import (
    DefenseHarnessExperimentConfig,
    DefenseMode,
    generate_defense_harness_experiment_name,
    utils,
)
from src.experiments.harness import run_harness_experiment

# Load user config
config = utils.load_user_config("src.user_configs.defense_harness.{user}")

assert config.DEFENSE_MODE != DefenseMode.NONE, "DEFENSE_MODE must be set to a non-none value"

EXPERIMENT_CONFIGS = DefenseHarnessExperimentConfig(
    experiment_name=generate_defense_harness_experiment_name(
        config.BASENAME,
        config.DEFENSE_MODE,
        config.DEFENSE_KWARGS.get("sampling_params", {}).get("model"),
        timestamp=config.TIMESTAMP_EXP_NAME,
    ),
    experiment_description=config.EXPERIMENT_DESCRIPTION,
    defense=config.DEFENSE_MODE,
    defense_kwargs=config.DEFENSE_KWARGS,
    reference_experiment_name=config.REFERENCE_EXPERIMENT_NAME,
    n_parallel_agents=config.N_PARALLEL_AGENTS,
)

if __name__ == "__main__":
    print(f"Experiment: {config.BASENAME}")
    pprint(EXPERIMENT_CONFIGS.model_dump())

    if not config.TESTS:
        try:
            input("Press Enter to begin experiment.")
        except KeyboardInterrupt:
            print("Exiting...")
            exit(0)

    run_harness_experiment(EXPERIMENT_CONFIGS, test_mode=config.TESTS)
