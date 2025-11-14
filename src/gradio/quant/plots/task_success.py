from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sqlalchemy import func

from src.db.client import Session
from src.db.tables import Experiment as DBExperiment
from src.db.tables import Trajectory as DBTrajectoryModel
from src.tasks import vwa_config, wa_config


def _fetch_rewards_and_steps(
    experiment_names: Iterable[str],
) -> list[tuple[str, str, bool, int]]:
    """Return experiment, task, success flag, and num steps for given experiments."""
    if not experiment_names:
        return []
    with Session() as session:
        results = (
            session.query(
                DBExperiment.name,
                DBTrajectoryModel.task_id,
                DBTrajectoryModel.success,
                func.jsonb_array_length(DBTrajectoryModel.trajectory["steps"]),
            )
            .join(DBExperiment, DBExperiment.id == DBTrajectoryModel.experiment_id)
            .filter(DBExperiment.name.in_(list(experiment_names)))
            .all()
        )
    return [
        (
            exp,
            task,
            bool(success),
            int(steps or 0),
        )
        for exp, task, success, steps in results
    ]


def _task_to_site(task_id: str) -> str:
    return wa_config.task_site_map.get(task_id) or vwa_config.task_site_map.get(task_id) or "unknown"


def plot_task_success_rate(experiment_names: list[str]) -> Figure:
    results = _fetch_rewards_and_steps(experiment_names)
    sns.set_theme(context="paper", style="white")

    if not results:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        ax.text(0.5, 0.5, "No Data", ha="center", va="center")
        ax.axis("off")
        return fig

    df = pd.DataFrame(results, columns=["experiment", "task", "success", "steps"])
    rates = df.groupby("experiment")["success"].mean().reset_index(name="rate")

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    sns.barplot(data=rates, x="experiment", y="rate", ax=ax)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Task Success Rate")
    ax.tick_params(axis="x", rotation=45)
    return fig


def plot_task_success_rate_by_site(experiment_names: list[str]) -> Figure:
    results = _fetch_rewards_and_steps(experiment_names)
    sns.set_theme(context="paper", style="white")

    if not results:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        ax.text(0.5, 0.5, "No Data", ha="center", va="center")
        ax.axis("off")
        return fig

    df = pd.DataFrame(results, columns=["experiment", "task", "success", "steps"])
    df["site"] = df["task"].apply(_task_to_site)
    grouped = df.groupby(["site", "experiment"])["success"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    sns.barplot(data=grouped, x="site", y="success", hue="experiment", ax=ax)
    ax.set_xlabel("Site")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Task Success Rate by Site")
    ax.tick_params(axis="x", rotation=45)
    return fig


def plot_steps_histogram_success(experiment_names: list[str]) -> Figure:
    results = _fetch_rewards_and_steps(experiment_names)
    sns.set_theme(context="paper", style="white")

    if not results:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        ax.text(0.5, 0.5, "No Data", ha="center", va="center")
        ax.axis("off")
        return fig

    df = pd.DataFrame(results, columns=["experiment", "task", "success", "steps"])
    # Filter for successful trajectories only
    success_df = df[df["success"]]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    sns.histplot(data=success_df, x="steps", hue="experiment", multiple="dodge", bins=20, ax=ax)
    ax.set_xlabel("Number of Steps")
    ax.set_ylabel("Count")
    ax.set_title("Steps Distribution for Successful Trajectories")
    ax.tick_params(axis="x")
    return fig


def plot_steps_histogram_failed(experiment_names: list[str]) -> Figure:
    results = _fetch_rewards_and_steps(experiment_names)
    sns.set_theme(context="paper", style="white")

    if not results:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        ax.text(0.5, 0.5, "No Data", ha="center", va="center")
        ax.axis("off")
        return fig

    df = pd.DataFrame(results, columns=["experiment", "task", "success", "steps"])
    # Filter for failed trajectories only
    failed_df = df[~df["success"]]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    sns.histplot(data=failed_df, x="steps", hue="experiment", multiple="dodge", bins=20, ax=ax)
    ax.set_xlabel("Number of Steps")
    ax.set_ylabel("Count")
    ax.set_title("Steps Distribution for Failed Trajectories")
    ax.tick_params(axis="x")
    return fig


def plot_steps_histogram_by_site(experiment_names: list[str]) -> Figure:
    results = _fetch_rewards_and_steps(experiment_names)
    sns.set_theme(context="paper", style="white")

    if not results:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        ax.text(0.5, 0.5, "No Data", ha="center", va="center")
        ax.axis("off")
        return fig

    df = pd.DataFrame(results, columns=["experiment", "task", "success", "steps"])
    df["site"] = df["task"].apply(_task_to_site)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    sns.histplot(data=df, x="steps", hue="site", multiple="dodge", bins=20, ax=ax)
    ax.set_xlabel("Number of Steps")
    ax.set_ylabel("Count")
    ax.set_title("Steps by Site")
    return fig


def plot_steps_box_plot(experiment_names: list[str]) -> Figure:
    results = _fetch_rewards_and_steps(experiment_names)
    sns.set_theme(context="paper", style="white")

    if not results:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        ax.text(0.5, 0.5, "No Data", ha="center", va="center")
        ax.axis("off")
        return fig

    df = pd.DataFrame(results, columns=["experiment", "task", "success", "steps"])
    df["outcome"] = df["success"].map({True: "Success", False: "Failure"})

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    sns.boxplot(data=df, x="experiment", y="steps", hue="outcome", ax=ax)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Number of Steps")
    ax.set_title("Steps Distribution by Outcome (Box Plot)")
    ax.tick_params(axis="x", rotation=45)
    return fig
