import argparse
from collections import defaultdict

import pandas as pd
from sqlalchemy import func
from tabulate import tabulate  # type: ignore

from src.db.client import Session
from src.db.helpers.defense_experiments import get_defense_experiment_id
from src.db.tables import DefenseHarnessStep, Observation, Trajectory
from src.tasks import vwa_config, wa_config


def task_to_site(task_id: str) -> str:
    return wa_config.task_site_map.get(task_id) or vwa_config.task_site_map.get(task_id) or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="Defense harness experiment name")
    args = parser.parse_args()

    defense_experiment_id, _ = get_defense_experiment_id(args.experiment)
    if defense_experiment_id is None:
        raise SystemExit(f"Defense harness experiment {args.experiment!r} not found")

    # Fetch counts of successes/failures per task_id over defense harness steps (i.e., observations)
    with Session() as session:
        rows = (
            session.query(
                Trajectory.task_id,
                DefenseHarnessStep.success,
                func.count().label("cnt"),
            )
            .join(Observation, Observation.trajectory_id == Trajectory.id)
            .join(DefenseHarnessStep, DefenseHarnessStep.observation_id == Observation.id)
            .filter(DefenseHarnessStep.defense_experiment_id == defense_experiment_id)
            .group_by(Trajectory.task_id, DefenseHarnessStep.success)
            .all()
        )

    # Count failures/successes per site (by observation/defense step)
    site_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"failures": 0, "successes": 0})
    for task_id, success, cnt in rows:
        site = task_to_site(task_id)
        if success:
            site_stats[site]["successes"] += int(cnt)
        else:
            site_stats[site]["failures"] += int(cnt)

    df = pd.DataFrame(
        [(site, stats["failures"], stats["successes"]) for site, stats in site_stats.items()],
        columns=["site", "failures", "successes"],
    ).sort_values(["failures", "successes"], ascending=[False, False])

    # First table: per-site breakdown
    print(tabulate(df.values, headers=list(df.columns), tablefmt="psql"))

    # Second table: totals
    total_failures = int(df["failures"].sum()) if not df.empty else 0
    total_successes = int(df["successes"].sum()) if not df.empty else 0
    totals_df = pd.DataFrame([["TOTAL", total_failures, total_successes]], columns=list(df.columns))
    print()
    print(tabulate(totals_df.values, headers=list(totals_df.columns), tablefmt="psql"))


if __name__ == "__main__":
    main()
