import argparse
from collections import defaultdict
from typing import Any, List, cast

import pandas as pd
from sqlalchemy import func
from tabulate import tabulate  # type: ignore

from src.db.client import Session
from src.db.tables import Experiment, Observation, TaskDataset, Trajectory
from src.tasks import vwa_config, wa_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Count task success/failure distribution")
    parser.add_argument("--min-runs", type=int, default=0, help="Minimum number of total runs required")
    parser.add_argument("--dataset", type=str, help="Filter tasks to those in the specified dataset")
    parser.add_argument("--experiment-name", type=str, help="Filter trajectories to a specific experiment name")
    parser.add_argument(
        "--success-only",
        action="store_true",
        dest="success_only",
        help="Only include successful tasks",
    )
    parser.add_argument("--limit", type=int, help="Maximum number of rows to display")
    parser.add_argument("--list", action="store_true", help="Print only the task IDs")
    args = parser.parse_args()

    def task_to_site(task_id: str) -> str:
        return wa_config.task_site_map.get(task_id) or vwa_config.task_site_map.get(task_id) or "unknown"

    with Session() as session:
        # If dataset is specified, get the task IDs from that dataset
        task_filter = None
        if args.dataset:
            dataset = session.query(TaskDataset).filter(TaskDataset.name == args.dataset).first()
            if not dataset:
                print(f"Error: Dataset '{args.dataset}' not found")
                return
            # dataset.tasks is annotated on the ORM model as a Column[List[str]],
            # but at runtime on an instance it's a Python list; cast for type-checkers
            dataset_tasks: List[str] = cast(List[str], dataset.tasks)
            task_filter = Trajectory.task_id.in_(dataset_tasks)

        # Query success/failure counts for each task
        query = session.query(
            Trajectory.task_id,
            Trajectory.success,
            func.count().label("count"),
        ).group_by(Trajectory.task_id, Trajectory.success)

        if task_filter is not None:
            query = query.filter(task_filter)

        # If experiment name is specified, filter trajectories to that experiment
        if args.experiment_name:
            experiment = session.query(Experiment).filter(Experiment.name == args.experiment_name).first()
            if not experiment:
                print(f"Error: Experiment '{args.experiment_name}' not found")
                return
            query = query.filter(Trajectory.experiment_id == experiment.id)

        rows = query.all()

    success_counts = defaultdict(int)
    failure_counts = defaultdict(int)
    for task_id, success, count in rows:
        if success:
            success_counts[task_id] = count
        else:
            failure_counts[task_id] = count

    # Create a DataFrame with both success and failure counts
    df = pd.DataFrame(
        {
            "task_id": list(set(success_counts.keys()) | set(failure_counts.keys())),
        }
    )
    df["success_count"] = df["task_id"].map(success_counts)
    df["failure_count"] = df["task_id"].map(failure_counts)
    df = df.fillna(0)

    # Convert counts to integers and sort by success_count
    df["success_count"] = df["success_count"].astype(int)
    df["failure_count"] = df["failure_count"].astype(int)
    df["total"] = df["success_count"] + df["failure_count"]

    # Filter by minimum runs
    if args.min_runs > 0:
        df = df[df["total"] >= args.min_runs]

    # Filter to only tasks with at least one success if requested
    if args.success_only:
        df = df[df["success_count"] > 0]

    df = df.sort_values(by="success_count", ascending=False)

    # Limit number of rows if specified
    if args.limit is not None:
        df = df.head(args.limit)

    print("\nTask Success/Failure Counts:")
    if args.dataset:
        print(f"Dataset: {args.dataset}")
    if args.experiment_name:
        print(f"Experiment: {args.experiment_name}")
    if args.min_runs > 0:
        print(f"Minimum runs: {args.min_runs}")
    if args.success_only:
        print("Success-only: true")
    if args.limit is not None:
        print(f"Showing top {args.limit} rows")
    if args.list:
        print(df["task_id"].tolist())
    else:
        print(tabulate(df.values, headers=list(df.columns), tablefmt="psql"))

        # Second table: aggregate failures by site type
        site_failures_df = (
            df.assign(site=df["task_id"].map(task_to_site)).groupby("site", as_index=False)["failure_count"].sum()
        )
        # Stable two-pass sort to satisfy pandas typings and preserve order
        site_failures_df = cast(Any, site_failures_df).sort_values(by="site", ascending=True)
        site_failures_df = cast(Any, site_failures_df).sort_values(by="failure_count", ascending=False)

        print("\nFailures by site:")
        print(tabulate(site_failures_df.values, headers=list(site_failures_df.columns), tablefmt="psql"))

        # Print user query/goal for failed tasks (not as a table)
        failed_df = df[df["failure_count"] > 0].copy()
        if not failed_df.empty:
            print("\nFailed task goals:")
            # Stable two-pass sort: task_id asc, then failure_count desc
            failed_df = cast(Any, failed_df).sort_values(by="task_id", ascending=True)
            failed_df = cast(Any, failed_df).sort_values(by="failure_count", ascending=False)

            with Session() as session:
                exp_id = None
                if args.experiment_name:
                    exp = session.query(Experiment).filter(Experiment.name == args.experiment_name).first()
                    exp_id = exp.id if exp else None

                for task_id in failed_df["task_id"].tolist():
                    q = (
                        session.query(Observation)
                        .join(Trajectory, Observation.trajectory_id == Trajectory.id)
                        .filter(Trajectory.task_id == task_id)
                        .order_by(Observation.id.asc())
                    )
                    if exp_id is not None:
                        q = q.filter(Trajectory.experiment_id == exp_id)

                    obs = q.first()
                    goal_text = None
                    if obs is not None:
                        data = getattr(obs, "data", None)
                        if isinstance(data, dict):
                            goal_text = data.get("goal") or data.get("goal_object")

                    site = task_to_site(task_id)
                    if goal_text is None:
                        print(f"- {task_id} [{site}]: <no goal found>")
                    else:
                        print(f"- {task_id} [{site}]: {goal_text}")


if __name__ == "__main__":
    main()
