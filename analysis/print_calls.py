import argparse
from collections import Counter

import pandas as pd
from tabulate import tabulate  # type: ignore
from tqdm import tqdm

from src.db.helpers.trajectories import (
    load_experiment_trajectories,
    load_trajectory,
)


def extract_action_name(action: str | None) -> str:
    if not action:
        return ""
    action_name = action.split("(")[0]
    return action_name[:20] if len(action_name) > 10 else action_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="Experiment name")
    args = parser.parse_args()

    exp, valid_ids, _incomplete, success_ids, failed_ids, _rate = load_experiment_trajectories(args.experiment)
    if exp is None:
        raise SystemExit(f"Experiment {args.experiment!r} not found")

    counts_all: Counter[str] = Counter()
    counts_failed: Counter[str] = Counter()
    counts_success: Counter[str] = Counter()

    for task_id in tqdm(valid_ids, desc="Loading trajectories"):
        traj, _ = load_trajectory(args.experiment, task_id, load_screenshots=False)
        for step in traj.steps:
            action_name = extract_action_name(step.action)
            if not action_name:
                continue
            counts_all[action_name] += 1
            if task_id in failed_ids:
                counts_failed[action_name] += 1
            if task_id in success_ids:
                counts_success[action_name] += 1

    df_all = pd.DataFrame(counts_all.items(), columns=["action", "count"]).sort_values("count", ascending=False)
    df_failed = pd.DataFrame(counts_failed.items(), columns=["action", "count"]).sort_values("count", ascending=False)
    df_success = pd.DataFrame(counts_success.items(), columns=["action", "count"]).sort_values("count", ascending=False)

    # Find actions that only appear in failed trajectories
    failed_only_actions = set(counts_failed.keys()) - set(counts_success.keys())
    df_harmful = pd.DataFrame(
        [(action, counts_failed[action]) for action in failed_only_actions], columns=["action", "count"]
    ).sort_values("count", ascending=False)

    print("All Trajectories:\n" + tabulate(df_all.values, headers=list(df_all.columns), tablefmt="psql"))
    print("\nFailed Trajectories:\n" + tabulate(df_failed.values, headers=list(df_failed.columns), tablefmt="psql"))
    print(
        "\nSuccessful Trajectories:\n" + tabulate(df_success.values, headers=list(df_success.columns), tablefmt="psql")
    )
    print(
        "\nPotentially Unhelpful Actions (Only in Failed Trajectories):\n"
        + tabulate(df_harmful.values, headers=list(df_harmful.columns), tablefmt="psql")
    )


if __name__ == "__main__":
    main()
