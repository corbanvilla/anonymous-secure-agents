import argparse

import pandas as pd
import scipy.stats as stats
from tabulate import tabulate  # type: ignore
from typing import cast, List
from tqdm import tqdm

try:
    import tiktoken
except Exception:
    tiktoken = None  # type: ignore[assignment]

from src.db.helpers.trajectories import (
    load_experiment_trajectories,
    load_trajectory,
)

if tiktoken is None:
    print("Tiktoken not found, using len()")


def count_tokens(text: str) -> int:
    if tiktoken is None:
        return len(text)
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def get_experiment_stats(experiment: str) -> tuple[pd.Series | None, list[int] | None]:
    exp, valid_ids, _incomplete, _success_ids, _failed_ids, _rate = load_experiment_trajectories(experiment)
    if exp is None:
        print(f"Warning: Experiment {experiment!r} not found")
        return None, None

    rows: list[dict] = []
    for task_id in tqdm(valid_ids, desc=f"{experiment} tasks", leave=False):
        traj, _ = load_trajectory(experiment, task_id, load_screenshots=False)
        for step in traj.steps:
            obs = step.observation
            if isinstance(obs, dict) and "a11y" in obs:
                tokens = count_tokens(str(obs["a11y"]))
                rows.append({"tokens": tokens})

    if not rows:
        print(f"Warning: No data found for experiment {experiment!r}")
        return None, None

    df = pd.DataFrame(rows)["tokens"]
    return df.describe(), df.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="+", help="One or more experiment names")
    args = parser.parse_args()

    # Collect stats for each experiment
    all_stats = {}
    raw_data = {}
    for exp in tqdm(args.experiments, desc="Experiments"):
        statistics, data = get_experiment_stats(exp)
        if statistics is not None:
            all_stats[exp] = statistics
            raw_data[exp] = data

    if not all_stats:
        raise SystemExit("No valid data found for any experiments")

    # Convert to DataFrame and format for display
    stats_df = pd.DataFrame(all_stats)
    stats_df = stats_df.round(1)  # Round to 1 decimal place

    # Add sum row
    sums = {exp: sum(cast(List[int], raw_data[exp])) for exp in all_stats.keys()}
    stats_df.loc["sum"] = pd.Series(sums)

    stats_table = stats_df.reset_index()

    # Create title based on whether we're using tokens or characters
    measure = "Chars" if tiktoken is None else "Tokens"
    title = f"Axtree Sizes ({measure})"

    # Format table with headers and title
    headers = ["Statistic"] + list(all_stats.keys())
    print("\n" + title)
    print(tabulate(stats_table.values.tolist(), headers=headers, tablefmt="psql"))

    # If exactly 2 experiments provided, run statistical test
    if len(args.experiments) == 2:
        exp1, exp2 = args.experiments
        stat, pval = stats.mannwhitneyu(raw_data[exp1], raw_data[exp2], alternative="two-sided")
        print(f"\nMann-Whitney U test comparing {exp1} vs {exp2}:")
        print(f"p-value: {pval:.4f}")
        print(f"Statistically {'different' if pval < 0.05 else 'similar'} (α=0.05)")


if __name__ == "__main__":
    main()
