#!/usr/bin/env python3
"""
This script backfills step_number values for all existing observations in the database.
It loops through all experiments, then their trajectories, and assigns step numbers
to observations in order of their creation (based on ID).
"""

import sys

from sqlalchemy import text
from sqlalchemy.orm import joinedload
from tqdm import tqdm

from src.db.client import Session, safe_commit
from src.db.tables import Experiment, Observation, Trajectory
from src.log import log


def backfill_step_numbers(dry_run: bool = True):
    """
    Backfill step_number values for all observations.

    Args:
        dry_run (bool): If True, only print what would be done without making changes.
    """
    with Session() as session:
        # Get total count of experiments for progress bar
        total_experiments = session.query(Experiment).count()

        # Iterate through all experiments
        for experiment in tqdm(session.query(Experiment), total=total_experiments, desc="Processing experiments"):
            log.info(f"Processing experiment {experiment.name} (ID: {experiment.id})")

            # Get all trajectories for this experiment
            trajectories = (
                session.query(Trajectory)
                .filter(Trajectory.experiment_id == experiment.id)
                .options(joinedload(Trajectory.observations))
                .all()
            )

            for trajectory in trajectories:
                # Get observations ordered by ID (creation order)
                sorted_observations = sorted(trajectory.observations, key=lambda o: o.id)

                # Update step numbers
                for step_number, observation in enumerate(sorted_observations):
                    if dry_run:
                        log.info(
                            f"Would set step_number={step_number} for observation {observation.id} "
                            f"in trajectory {trajectory.id}"
                        )
                    else:
                        observation.step_number = step_number

                if not dry_run:
                    safe_commit(session)
                    log.info(f"Updated {len(sorted_observations)} observations for trajectory {trajectory.id}")


def verify_step_numbers():
    """
    Verify that step numbers are correctly assigned:
    1. All observations have step numbers
    2. Step numbers are sequential within each trajectory
    3. No duplicate step numbers within a trajectory
    """
    with Session() as session:
        # Check for any NULL step numbers
        null_count = session.query(Observation).filter(Observation.step_number.is_(None)).count()
        if null_count > 0:
            log.error(f"Found {null_count} observations with NULL step_number")
            return False

        # Check for non-sequential or duplicate step numbers within trajectories
        check_sql = text(
            """
            WITH numbered_obs AS (
                SELECT 
                    id,
                    trajectory_id,
                    step_number,
                    ROW_NUMBER() OVER (PARTITION BY trajectory_id ORDER BY id) - 1 as expected_step
                FROM observations_v2
            )
            SELECT COUNT(*) 
            FROM numbered_obs 
            WHERE step_number != expected_step
        """
        )

        mismatch_count = session.execute(check_sql).scalar()
        if mismatch_count > 0:
            log.error(f"Found {mismatch_count} observations with non-sequential or duplicate step numbers")
            return False

        log.info("All step numbers verified successfully!")
        return True


def main():
    """Main entry point for the script."""
    import argparse

    parser = argparse.ArgumentParser(description="Backfill step numbers for observations")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without making changes")
    args = parser.parse_args()

    if args.dry_run:
        log.info("Running in dry-run mode - no changes will be made")

    try:
        backfill_step_numbers(dry_run=args.dry_run)
        if not args.dry_run:
            if verify_step_numbers():
                log.info("Migration completed successfully!")
            else:
                log.error("Migration verification failed!")
                sys.exit(1)
    except Exception as e:
        log.error(f"Error during migration: {e}")
        raise


if __name__ == "__main__":
    main()
