import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from tabulate import tabulate  # type: ignore[import]
from tqdm import tqdm

from src.db.helpers.trajectories import (
    load_experiment_trajectories,
    load_trajectory,
)
from src.tasks import vwa_config, wa_config

SAVE_PATH = Path(__file__).parent.parent.parent / ".analysis"


def task_to_site(task_id: str) -> str:
    """Map a task id to its site name."""
    return wa_config.task_site_map.get(task_id) or vwa_config.task_site_map.get(task_id) or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--out", type=Path, default=SAVE_PATH / "step_errors.json")
    args = parser.parse_args()

    exp, valid_ids, _incomplete, success_ids, failed_ids, _rate = load_experiment_trajectories(args.experiment)
    if exp is None:
        raise SystemExit(f"Experiment {args.experiment!r} not found")

    errors: list[str] = []
    task_counts: Counter[str] = Counter()
    site_counts: Counter[str] = Counter()
    site_per_task_counts: Counter[str] = Counter()
    task_has_error: defaultdict[str, set[str]] = defaultdict(set)
    success_tasks_with_errors = set()
    failed_tasks_with_errors = set()

    for task_id in tqdm(valid_ids, desc="Loading trajectories"):
        traj, _ = load_trajectory(args.experiment, task_id, load_screenshots=False)
        site = task_to_site(task_id)
        has_error = False
        for step in traj.steps:
            obs = step.observation
            err = None
            if isinstance(obs, dict):
                err = obs.get("last_action_error")
            if err:
                has_error = True
                errors.append(err)
                task_counts[task_id] += 1
                site_counts[site] += 1
                if site not in task_has_error[task_id]:
                    task_has_error[task_id].add(site)
                    site_per_task_counts[site] += 1

        # Track unique tasks with errors
        if has_error:
            if task_id in success_ids:
                success_tasks_with_errors.add(task_id)
            elif task_id in failed_ids:
                failed_tasks_with_errors.add(task_id)

    args.out.write_text(json.dumps(errors, indent=2))
    print(f"Saved {len(errors)} errors to {args.out}")

    df_tasks = pd.DataFrame(task_counts.items(), columns=["task_id", "count"]).sort_values("count", ascending=False)
    df_sites = pd.DataFrame(site_counts.items(), columns=["site", "count"]).sort_values("count", ascending=False)
    df_sites_per_task = pd.DataFrame(site_per_task_counts.items(), columns=["site", "tasks_with_errors"]).sort_values(
        "tasks_with_errors", ascending=False
    )
    df_outcome_errors = pd.DataFrame(
        [
            ["Successful Tasks", len(success_tasks_with_errors)],
            ["Failed Tasks", len(failed_tasks_with_errors)],
            ["Total Tasks with Errors", len(success_tasks_with_errors) + len(failed_tasks_with_errors)],
        ],
        columns=["outcome", "tasks_with_errors"],
    )

    print("\nErrors by Task:\n" + tabulate(df_tasks.values, headers=list(df_tasks.columns), tablefmt="psql"))

    # Add total errors table
    df_total = pd.DataFrame([["Total Errors", len(errors)]], columns=["metric", "count"])
    print("\nTotal Errors:\n" + tabulate(df_total.values, headers=list(df_total.columns), tablefmt="psql"))

    print("\nErrors by Site:\n" + tabulate(df_sites.values, headers=list(df_sites.columns), tablefmt="psql"))
    print(
        "\nUnique Tasks with Errors by Site:\n"
        + tabulate(df_sites_per_task.values, headers=list(df_sites_per_task.columns), tablefmt="psql")
    )
    print(
        "\nTasks with Errors by Outcome:\n"
        + tabulate(df_outcome_errors.values, headers=list(df_outcome_errors.columns), tablefmt="psql")
    )


if __name__ == "__main__":
    main()
