import argparse
from collections import Counter

import pandas as pd
from tabulate import tabulate  # type: ignore
from tqdm import tqdm

from src.db.helpers.trajectories import (
    load_experiment_trajectories,
    load_trajectory,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="Experiment name")
    args = parser.parse_args()

    (
        exp,
        _valid_ids,
        _incomplete,
        _success_ids,
        failed_ids,
        _rate,
    ) = load_experiment_trajectories(args.experiment)
    if exp is None:
        raise SystemExit(f"Experiment {args.experiment!r} not found")

    counts: Counter[str] = Counter()
    for task_id in tqdm(failed_ids, desc="Processing failed tasks"):
        traj, _ = load_trajectory(args.experiment, task_id, load_screenshots=False)
        if traj.termination_reason == "MAX_STEPS":
            continue
        if not traj.steps:
            continue
        step = traj.steps[-1]
        action = step.action
        if not action and isinstance(step.observation, dict):
            action = step.observation.get("last_action")
        if action:
            # Truncate calls longer than 20 characters
            if len(action) > 20:
                action = action[:17] + "..."
            counts[action] += 1

    df = pd.DataFrame(counts.items(), columns=["call", "count"]).sort_values("count", ascending=False)
    print(tabulate(df.values, headers=list(df.columns), tablefmt="psql"))


if __name__ == "__main__":
    main()
