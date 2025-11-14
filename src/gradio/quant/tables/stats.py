from typing import List

import pandas as pd
from sqlalchemy.orm import Session as SQLAlchemySession

from src.db.client import engine
from src.db.tables import Experiment


def calculate_experiment_metrics(experiment_names: List[str]) -> pd.DataFrame:
    """
    Calculate metrics for each experiment and return as a DataFrame with experiments as columns
    and metrics as rows.
    """
    metrics = {}

    with SQLAlchemySession(engine) as session:
        for exp_name in experiment_names:
            # Load experiment with trajectories in a single query
            exp = session.query(Experiment).filter_by(name=exp_name).first()

            if not exp:
                continue

            # Calculate metrics for this experiment
            total_tasks = len(exp.trajectories)
            successful_tasks = sum(1 for t in exp.trajectories if t.success)
            success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0

            # Calculate average steps using the steps array in the trajectory JSONB
            success_steps = [len(t.trajectory["steps"]) for t in exp.trajectories if t.success]
            failed_steps = [len(t.trajectory["steps"]) for t in exp.trajectories if not t.success]

            avg_success_steps = sum(success_steps) / len(success_steps) if success_steps else 0
            avg_failed_steps = sum(failed_steps) / len(failed_steps) if failed_steps else 0

            # Calculate median steps
            median_success_steps = pd.Series(success_steps).median() if success_steps else 0
            median_failed_steps = pd.Series(failed_steps).median() if failed_steps else 0

            metrics[exp_name] = {
                "Task Success Rate": f"{success_rate:.1%}",
                "Avg. Steps (Success)": f"{avg_success_steps:.1f}",
                "Avg. Steps (Failed)": f"{avg_failed_steps:.1f}",
                "Median Steps (Success)": f"{median_success_steps:.1f}",
                "Median Steps (Failed)": f"{median_failed_steps:.1f}",
            }

    # Convert metrics dict to DataFrame with metrics as a column
    rows = []
    metric_names = [
        "Task Success Rate",
        "Avg. Steps (Success)",
        "Avg. Steps (Failed)",
        "Median Steps (Success)",
        "Median Steps (Failed)",
    ]

    for metric in metric_names:
        row = {"Metric": metric}
        for exp_name in metrics:
            row[exp_name] = metrics[exp_name][metric]
        rows.append(row)

    return pd.DataFrame(rows)
