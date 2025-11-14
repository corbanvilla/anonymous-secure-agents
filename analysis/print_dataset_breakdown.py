import argparse
from collections import Counter
from typing import List, cast

import pandas as pd
from sqlalchemy import func
from tabulate import tabulate  # type: ignore

from src.db.client import Session
from src.db.tables import Experiment, Observation, TaskDataset, Trajectory
from src.tasks import vwa_config, wa_config


def task_to_site(task_id: str) -> str:
    """Map a task id to its site name (pattern reused from other analysis scripts)."""
    return wa_config.task_site_map.get(task_id) or vwa_config.task_site_map.get(task_id) or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Show site-type distribution for a dataset")
    parser.add_argument("dataset", type=str, help="Dataset name (from task_datasets table)")
    parser.add_argument("--experiment", type=str, help="Optional experiment name to summarize observations")
    args = parser.parse_args()

    with Session() as session:
        dataset = session.query(TaskDataset).filter(TaskDataset.name == args.dataset).first()
        if not dataset:
            raise SystemExit(f"Error: Dataset {args.dataset!r} not found")

        # dataset.tasks is a Python list at runtime
        task_ids: List[str] = cast(List[str], dataset.tasks)

    counts: Counter[str] = Counter()
    for task_id in task_ids:
        site = task_to_site(task_id)
        counts[site] += 1

    df = pd.DataFrame(counts.items(), columns=["site", "count"]).sort_values("count", ascending=False)

    print(f"\nDataset: {args.dataset}")
    print(f"Total tasks: {len(task_ids)}")
    print(tabulate(df.values, headers=list(df.columns), tablefmt="psql"))

    # Optional: Experiment observations summary
    if args.experiment:
        with Session() as session:
            exp = session.query(Experiment).filter(Experiment.name == args.experiment).first()
            if not exp:
                raise SystemExit(f"Error: Experiment {args.experiment!r} not found")

            num_traj = (
                session.query(func.count(Trajectory.id)).filter(Trajectory.experiment_id == exp.id).scalar()
            ) or 0
            total_obs = (
                session.query(func.count(Observation.id))
                .join(Trajectory, Observation.trajectory_id == Trajectory.id)
                .filter(Trajectory.experiment_id == exp.id)
                .scalar()
            ) or 0
            avg_per_traj = (total_obs / num_traj) if num_traj else 0.0

        obs_df = pd.DataFrame(
            [
                ["experiment", args.experiment],
                ["trajectories", int(num_traj)],
                ["total_observations", int(total_obs)],
                ["avg_observations_per_trajectory", round(avg_per_traj, 2)],
            ],
            columns=["metric", "value"],
        )
        print("\nExperiment Observation Summary:")
        print(tabulate(obs_df.values, headers=list(obs_df.columns), tablefmt="psql"))


if __name__ == "__main__":
    main()
