import argparse
import datetime

import sqlalchemy

from src.db.config import POSTGRES_DATABASE_URL

DEFAULT_DAYS = 3

DELETE_TEMPLATE = """
DELETE FROM experiments
WHERE name LIKE 'test-%' AND created_at < :cutoff{user_clause}
RETURNING id, name
"""

SELECT_TEMPLATE = """
SELECT id, name, username, created_at
FROM experiments
WHERE name LIKE 'test-%' AND created_at < :cutoff{user_clause}
ORDER BY created_at
"""

SELECT_NO_TRAJ = """
SELECT id, name, username, created_at
FROM experiments
WHERE NOT EXISTS (
    SELECT 1 FROM trajectories t WHERE t.experiment_id = experiments.id
)
ORDER BY created_at
"""

DELETE_NO_TRAJ = """
DELETE FROM experiments
WHERE NOT EXISTS (
    SELECT 1 FROM trajectories t WHERE t.experiment_id = experiments.id
)
RETURNING id, name
"""


def prune_records(engine: sqlalchemy.engine.Engine, days: int, username: str | None) -> None:
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    user_clause = " AND username = :username" if username else ""
    params: dict[str, object] = {"cutoff": cutoff}
    if username:
        params["username"] = username

    # Preview experiments to delete
    select_query = sqlalchemy.text(SELECT_TEMPLATE.format(user_clause=user_clause))
    with engine.connect() as conn:
        rows = conn.execute(select_query, params).fetchall()

    if not rows:
        print("No matching experiments found.")
        return

    print("The following experiments will be deleted:")
    for row in rows:
        print(f"{row.id}: {row.name} ({row.username}, {row.created_at})")

    confirm = input("Type 'yes' to confirm deletion: ")
    if confirm.strip().lower() != "yes":
        print("Aborting deletion.")
        return

    # Perform deletion
    delete_query = sqlalchemy.text(DELETE_TEMPLATE.format(user_clause=user_clause))
    with engine.begin() as conn:
        result = conn.execute(delete_query, params)
        for row in result:
            print(f"deleted experiment {row.id}: {row.name}")


def prune_records_no_trajectories(engine: sqlalchemy.engine.Engine) -> None:
    """Delete experiments that have no associated trajectories."""

    select_query = sqlalchemy.text(SELECT_NO_TRAJ)
    with engine.connect() as conn:
        rows = conn.execute(select_query).fetchall()

    if not rows:
        print("No experiments without trajectories found.")
        return

    print("The following experiments lack trajectories and will be deleted:")
    for row in rows:
        print(f"{row.id}: {row.name} ({row.username}, {row.created_at})")

    confirm = input("Type 'yes' to confirm deletion: ")
    if confirm.strip().lower() != "yes":
        print("Aborting deletion.")
        return

    delete_query = sqlalchemy.text(DELETE_NO_TRAJ)
    with engine.begin() as conn:
        result = conn.execute(delete_query)
        for row in result:
            print(f"deleted experiment {row.id}: {row.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help="Delete 'test-*' experiments older than this many days",
    )
    parser.add_argument("--username", type=str, default=None, help="Only delete experiments for this user")
    parser.add_argument(
        "--no-trajectories",
        action="store_true",
        help="Also delete experiments that have no trajectories",
    )
    args = parser.parse_args()

    engine = sqlalchemy.create_engine(POSTGRES_DATABASE_URL, future=True)
    if args.no_trajectories:
        prune_records_no_trajectories(engine)
    prune_records(engine, args.days, args.username)


if __name__ == "__main__":
    main()
