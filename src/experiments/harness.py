import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.db.commit_queue.defense_harness import (
    store_defense_skip_ids,
    store_incremental_defense_steps,
)
from src.db.helpers.defense_experiments import (
    create_defense_experiment,
    get_defense_experiment_id,
)
from src.db.helpers.defense_step_evals import get_missing_defense_evals
from src.db.helpers.experiments import get_experiment_id
from src.db.helpers.trajectories import load_observation
from src.defenses.defense_harness import has_capabilities_for_action, parse_action

# TODO - move these out to env folder (or atleast what's relevant there)
from src.defenses.one_stage.models.dom import AnnotatedDom
from src.environments.observations.dom_parser import (
    flatten_dom_to_str_get_dict,
    prune_dom_dict,
)
from src.experiments.config import (
    DEFAULT_IGNORE_KEYS_DEFENSE_HARNESS,
    DefenseHarnessExperimentConfig,
    are_configs_equal,
)
from src.log import log, log_config_differences

BATCH_SIZE = 1


def run_harness_experiment(experiment_config: DefenseHarnessExperimentConfig, test_mode: bool = False):
    # Ensure experiment exists
    experiment_name = experiment_config.experiment_name
    experiment_description = experiment_config.experiment_description
    reference_experiment_name = experiment_config.reference_experiment_name
    n_parallel_agents = experiment_config.n_parallel_agents if not test_mode else 1

    # Ensure reference experiment exists
    log.info(f"Checking if reference experiment {reference_experiment_name} exists")
    reference_experiment_id = get_experiment_id(reference_experiment_name)
    if not reference_experiment_id:
        raise ValueError(f"Reference experiment {reference_experiment_name} not found")

    # Ensure defense experiment exists
    defense_experiment_id, defense_experiment_config = get_defense_experiment_id(experiment_name)
    if defense_experiment_id is None:
        log.info(f"Creating defense experiment {experiment_name}")
        defense_experiment_id = create_defense_experiment(
            experiment_name, reference_experiment_id, experiment_config.model_dump(), experiment_description
        )
    else:
        log.info(f"Defense experiment {experiment_name} already exists")
        assert defense_experiment_config is not None
        is_equal, differences = are_configs_equal(
            experiment_config,
            defense_experiment_config,
            ignore_keys=DEFAULT_IGNORE_KEYS_DEFENSE_HARNESS,
        )
        if not is_equal:
            log.error(f"Defense experiment config mismatch for {experiment_name}:")
            log_config_differences(differences)
            raise ValueError(f"Defense experiment config mismatch for {experiment_name}")

    # Get missing defense evals
    missing_defense_evals = get_missing_defense_evals(defense_experiment_id)
    log.info(f"Located ({len(missing_defense_evals)}) missing defense evals!")

    # Prepare work items and skip items with no bid
    work_items = []
    skip_observation_ids = []

    for obs_id, action in missing_defense_evals:
        try:
            function_name, bid = parse_action(action)
        except ValueError:
            print(f"Skipping observation {obs_id} (action: {action}) not recognized!")
            skip_observation_ids.append(obs_id)
            continue

        if not bid:
            print(f"Skipping observation {obs_id} (action: {action}) because it has no bid")
            skip_observation_ids.append(obs_id)
        else:
            work_items.append((obs_id, action, function_name, bid))

    log.info(f"Will process {len(work_items)} observations with {n_parallel_agents} parallel agents")

    if test_mode:
        log.info("Test mode: stopping after first completion")
        work_items = work_items[:1]

    # Execute steps in parallel
    defense_steps = []
    completed_count = 0

    with ThreadPoolExecutor(max_workers=n_parallel_agents) as executor:
        # Submit all tasks
        future_to_work_item = {
            executor.submit(execute_step, obs_id, experiment_config): (obs_id, action, function_name, bid)
            for obs_id, action, function_name, bid in work_items
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(work_items), desc="Processing defense evaluations", unit="eval") as pbar:
            for future in as_completed(future_to_work_item):
                obs_id, action, function_name, bid = future_to_work_item[future]
                completed_count += 1

                try:
                    defense_result = future.result()
                    defense_step = {
                        "observation_id": obs_id,
                        "full_action": action,
                        "function": function_name,
                        "required_bid": bid,
                        "allowed_bids": defense_result["allowed_bids"],
                        "all_bids": defense_result["all_bids"],
                        "error_message": defense_result["error_message"],
                        "llm_logs": defense_result["llm_logs"],
                        "relevant_cap_set": has_capabilities_for_action(function_name, bid, defense_result["cap_set"])
                        if defense_result["cap_set"] is not None
                        else None,
                        "async_messages_stats": defense_result["async_messages_stats"],
                    }
                    defense_steps.append(defense_step)

                    # Store in batches
                    if len(defense_steps) >= BATCH_SIZE:
                        log.info(f"Completed {completed_count}/{len(work_items)} defense steps, storing batch")
                        store_incremental_defense_steps(defense_experiment_id, defense_steps)
                        defense_steps = []

                except Exception as e:
                    log.error(f"Error processing observation {obs_id}: {e}")
                    # You might want to handle this differently based on your needs

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Completed": completed_count, "Batched": len(defense_steps)})

    # Store any remaining items
    if defense_steps or skip_observation_ids:
        log.info("Storing remaining defense steps and skip IDs")
        store_incremental_defense_steps(defense_experiment_id, defense_steps)
        store_defense_skip_ids(defense_experiment_id, skip_observation_ids)

    log.info(f"Completed processing {completed_count} observations")


def execute_step(obs_id: int, experiment_config: DefenseHarnessExperimentConfig):
    defense_cls = experiment_config.defense.value
    assert isinstance(defense_cls, type)
    defense = defense_cls(**experiment_config.defense_kwargs)
    obs = load_observation(obs_id, load_screenshots=False, load_large_files=True)
    setattr(defense, "capture_censored_screenshot", lambda _, __: None)  # override capture screenshot
    log.info(f"Running defense {defense_cls.get_defense_id()} on observation {obs_id}")

    dom = obs["dom_object"]
    _, dom_elements = flatten_dom_to_str_get_dict(dom)
    pruned_dom = prune_dom_dict(dom_elements)
    annotated_dom = AnnotatedDom.from_a11y_tree(pruned_dom)
    all_bids = annotated_dom.all_bids

    try:
        result = defense.run_defense(obs, None)
    except Exception as e:
        print(f"Error running defense: {e}")
        traceback.print_exc()
        return {
            "all_bids": all_bids,
            "allowed_bids": [],
            "error_message": str(e),
            "llm_logs": defense.messages,
            "cap_set": None,
            "async_messages_stats": None,
        }

    print(f"ASYNC MESSAGES STATS: {defense.async_messages_stats}")

    return {
        "all_bids": all_bids,
        "allowed_bids": result["allowed_bids"],
        "error_message": None,
        "llm_logs": defense.messages,
        "cap_set": result["cap_set"],
        "async_messages_stats": defense.async_messages_stats,
    }
