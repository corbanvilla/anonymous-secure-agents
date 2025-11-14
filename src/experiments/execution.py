import os
import traceback
from typing import Any, AsyncGenerator, Dict, List, Tuple, cast

from rllm.agents.agent import Trajectory
from rllm.engine.async_agent_execution_engine import AsyncAgentExecutionEngine
from rllm.environments.browsergym.browsergym_process import BrowserGymProcess
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer

from src.agents.rllm.web_agent import WebAgent
from src.db.commit_queue.experiments import store_trajectories_in_db
from src.db.helpers.experiments import (
    create_experiment,
    get_experiment_trajectories,
    get_running_experiments,
    update_experiment_status,
)
from src.db.helpers.logging import log_error
from src.experiments.config import (
    DEFAULT_IGNORE_KEYS,
    ExperimentConfig,
    are_configs_equal,
)
from src.experiments.infra import await_all_web_servers_ready, reset_web_servers
from src.log import log, log_config_differences

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'


async def run_experiment(experiment_config: ExperimentConfig, reset_vwa_server: bool = False):
    """
    Run an experiment.

    Args:
        experiment_config (ExperimentConfig): The experiment configuration.

    Returns:
        None
    """

    # Determine pending tasks for this experiment
    existing_config, existing_experiment = get_experiment_trajectories(experiment_config.experiment_name)
    tasks_to_run: List[str]
    if existing_experiment is None:
        tasks_to_run = experiment_config.tasks
        log.info(f"Experiment {experiment_config.experiment_name} not found in DB. Creating and running all tasks.")
        create_experiment(
            experiment_config.experiment_name,
            tasks_to_run,
            experiment_config.model_dump(),
            experiment_config.experiment_description,
        )
    else:
        # Validate that the stored config matches the current config
        assert existing_config is not None
        is_equal, differences = are_configs_equal(
            experiment_config,
            existing_config,
            ignore_keys=DEFAULT_IGNORE_KEYS,
        )
        if not is_equal:
            log.error(f"Experiment config mismatch for '{experiment_config.experiment_name}':")
            log_config_differences(differences)
            raise ValueError(f"Experiment config mismatch for '{experiment_config.experiment_name}'")
        tasks_to_run = [t for t, tid in existing_experiment.items() if tid is None]
        log.info(f"Experiment {experiment_config.experiment_name} found in DB with {len(tasks_to_run)} pending tasks.")

        # Don't reset if we're continuing an experiment
        reset_vwa_server = False
        await_all_web_servers_ready()

    if not tasks_to_run:
        log.info(f"All tasks complete for experiment {experiment_config.experiment_name}. Exiting.")
        return

    try:
        if reset_vwa_server:
            if exp := get_running_experiments():
                log.info(f"User '{exp.username}' is already running an experiment (name: '{exp.name}'). Exiting.")
                return

            update_experiment_status(experiment_config.experiment_name, is_running=True)
            reset_web_servers()

        async for task, trajectory, chat_completions in run_tasks(tasks_to_run, experiment_config):
            log.info(f"Storing {task} trajectory in DB!")
            store_trajectories_in_db(
                experiment_config.experiment_name,
                [task],
                [trajectory],
                chat_completions=[chat_completions],
            )
            log.info(f"Finished storing {task} trajectory")

        log.info(f"All batches stored. Experiment name: {experiment_config.experiment_name}")
    except Exception as e:
        log.error(f"Error running experiment {experiment_config.experiment_name}: {e}")
        log.error(f"Full traceback:\n{traceback.format_exc()}")
        log_error(str(e), phase="execution")
    finally:
        update_experiment_status(experiment_config.experiment_name, is_running=False)


async def run_tasks(
    tasks: List[str], experiment_config: ExperimentConfig
) -> AsyncGenerator[Tuple[str, Trajectory, List[Dict[str, str]]], Any]:
    """Run tasks concurrently."""

    engine_options = experiment_config.engine_options.model_dump()
    engine_options["sampling_params"] = experiment_config.engine_options.sampling_params.model_dump(
        exclude_defaults=True
    )
    browser_env_args = experiment_config.env_args.model_dump()
    agent_env_args = experiment_config.agent_env_args.model_dump()
    tokenizer_path = experiment_config.tokenizer_model_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_cache=False)
    engine_options["tokenizer"] = tokenizer

    envs = [
        BrowserGymProcess(
            **browser_env_args,
            env_id=task,
            task_id=task,
        )
        for task in tasks
    ]
    agents = [WebAgent(**agent_env_args) for _ in range(len(tasks))]
    engine = AsyncAgentExecutionEngine(
        agents=agents,
        envs=envs,
        tasks=tasks,
        **engine_options,
        retry_limit=1,
    )

    try:
        pbar = tqdm(total=len(tasks), desc="Running tasks")
        async for result in engine.trajectory_generator():
            if result is None:
                log.info("Missing result from engine!")
                continue

            idx = int(cast(int, result["idx"]))
            trajectory = cast(Trajectory, result["trajectory"])
            chat_completions = result["chat_completions_text"]
            pbar.update(1)
            yield tasks[idx], trajectory, chat_completions
        pbar.close()

    except Exception as e:
        log.error(f"Error executing task: {e}")
        log.error(f"Full traceback:\n{traceback.format_exc()}")
        log_error(str(e), phase="execution")
        # raise
