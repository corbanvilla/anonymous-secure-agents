"""
This script prunes the database of observations_v2 by removing the
- chat_messages
- goal_object
- dom_object
- axtree_object
- extra_element_properties

This significantly reduces the size of the database.
"""

import sqlalchemy

from src.db.config import POSTGRES_DATABASE_URL

BATCH_SIZE = 1000

SELECT_QUERY = sqlalchemy.text(
    """
    SELECT id
    FROM observations_v2
    WHERE id > :last_id
    ORDER BY id
    LIMIT :limit
    """
)

# This query removes the specified keys from the JSONB column
REMOVE_QUERY = sqlalchemy.text(
    "UPDATE observations_v2 SET data = data - 'chat_messages' - 'goal_object' - 'dom_object' - 'axtree_object' - 'extra_element_properties' WHERE id = ANY(:ids)"
)


def prune_observations(engine: sqlalchemy.engine.Engine, batch_size: int = BATCH_SIZE) -> None:
    last_id = 0
    while True:
        with engine.connect() as conn:
            ids = [row.id for row in conn.execute(SELECT_QUERY, {"last_id": last_id, "limit": batch_size})]

        if not ids:
            break

        with engine.begin() as conn:
            conn.execute(REMOVE_QUERY, {"ids": ids})

        last_id = ids[-1]
        if len(ids) < batch_size:
            break


def main() -> None:
    engine = sqlalchemy.create_engine(POSTGRES_DATABASE_URL, future=True)
    prune_observations(engine)


if __name__ == "__main__":
    main()
