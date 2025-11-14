from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from src.db.client import Session
from src.db.tables import Experiment as DBExperiment
from src.db.tables import Trajectory as DBTrajectoryModel


def _fetch_stop_reasons(experiment_names: Iterable[str]) -> list[tuple[str, str]]:
    if not experiment_names:
        return []
    with Session() as session:
        results = (
            session.query(
                DBExperiment.name,
                DBTrajectoryModel.trajectory["termination_reason"].astext,
            )
            .join(DBExperiment, DBExperiment.id == DBTrajectoryModel.experiment_id)
            .filter(DBExperiment.name.in_(list(experiment_names)))
            .filter(
                # Filter for failed experiments
                (DBTrajectoryModel.success.is_(False)) | (DBTrajectoryModel.success.is_(None))
            )
            .all()
        )
    return [(r[0], r[1]) for r in results if r[1]]


def plot_stop_reason_by_experiment(experiment_names: list[str]) -> Figure:
    """Return Figure showing stop reason distributions by experiment.

    Shows individual experiments within each reason group.
    The figure size is fixed at 12x8 inches.
    """
    results = _fetch_stop_reasons(experiment_names)

    if not results:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        ax.text(0.5, 0.5, "No Data", ha="center", va="center")
        ax.axis("off")
        return fig

    # Create DataFrame with experiment names and reasons
    df = pd.DataFrame(results, columns=["experiment", "reason"])

    # Use static figure size
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # Count occurrences of each experiment-reason combination
    exp_counts = df.groupby(["reason", "experiment"]).size().reset_index().rename(columns={0: "count"})
    sns.barplot(
        data=exp_counts,
        x="reason",
        y="count",
        hue="experiment",
        ax=ax,
    )
    ax.set_xlabel("Stop Reason")
    ax.set_ylabel("Count")
    ax.set_title("Stop Reasons by Experiment (Failed Tasks Only)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    return fig


def plot_stop_reason_aggregated(experiment_names: list[str]) -> Figure:
    """Return a 300 PPI figure showing aggregated stop reason distributions.

    Shows the total counts across all experiments for each stop reason.
    The figure size is fixed at 12x8 inches.
    """
    results = _fetch_stop_reasons(experiment_names)
    sns.set_theme(context="paper", style="white")

    if not results:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        ax.text(0.5, 0.5, "No Data", ha="center", va="center")
        ax.axis("off")
        return fig

    # Create DataFrame with experiment names and reasons
    df = pd.DataFrame(results, columns=["experiment", "reason"])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # Aggregated counts
    agg_df = df["reason"].value_counts().reset_index()
    agg_df.columns = ["reason", "count"]
    sns.barplot(data=agg_df, x="reason", y="count", ax=ax)
    ax.set_xlabel("Stop Reason")
    ax.set_ylabel("Count")
    ax.set_title("Aggregated Stop Reasons (Failed Tasks Only)")
    ax.tick_params(axis="x", rotation=45)

    return fig


# save to file when called directly
if __name__ == "__main__":
    with Session() as session:
        test_experiments = session.query(DBExperiment.name).order_by(DBExperiment.id.desc()).limit(3)
    assert test_experiments, "No experiments found"
    fig = plot_stop_reason_by_experiment([exp.name for exp in test_experiments])
    fig.savefig(".plots/stop_reason.png")

    fig = plot_stop_reason_aggregated([exp.name for exp in test_experiments])
    fig.savefig(".plots/stop_reason_aggregated.png")
