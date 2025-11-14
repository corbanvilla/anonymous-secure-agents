import asyncio
from pprint import pprint

from src.experiments.config import (
    generate_experiment_name,
    get_experiment_config,
    utils,
)
from src.experiments.execution import run_experiment
from src.log import log

# Load user config
config = utils.load_user_config("src.user_configs.experiments.{user}")

EXPERIMENT_CONFIGS = [
    get_experiment_config(
        generate_experiment_name(
            config.BASENAME,
            srcs,
            config.ENGINE_MODEL,
            config.DEFENSE_MODE,
            config.ATTACK_MODE,
            config.TIMESTAMP_EXP_NAME,
        ),
        config.EXPERIMENT_DESCRIPTION,
        config.TASKS,
        webagent_src=srcs,
        engine_model=config.ENGINE_MODEL,
        defense=config.DEFENSE_MODE,
        defense_kwargs=config.DEFENSE_KWARGS,
        attack=config.ATTACK_MODE,
        attack_kwargs=config.ATTACK_KWARGS,
        n_parallel_agents=config.N_PARALLEL_AGENTS,
        max_steps=config.MAX_STEPS,
    )
    for srcs in config.WEBAGENT_SRC_COMBINATIONS
]

if __name__ == "__main__":
    print(f"Experiments: {[c.experiment_name for c in EXPERIMENT_CONFIGS]}")
    for c in EXPERIMENT_CONFIGS:
        pprint(c.model_dump())

    if not config.TESTS:
        try:
            input(f"Press Enter to begin {len(EXPERIMENT_CONFIGS)} experiments.")
        except KeyboardInterrupt:
            print("Exiting...")
            exit(0)

    for experiment_config in EXPERIMENT_CONFIGS:
        log.info(f"Running experiment: {experiment_config.experiment_name}")
        asyncio.run(run_experiment(experiment_config, reset_vwa_server=config.RESTART_VWA_SERVER))

        # Exit after first test experiment under test mode
        if config.TESTS:
            exit(0)
