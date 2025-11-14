import collections

import sqlalchemy

from src.db.config import POSTGRES_DATABASE_URL

BATCH_SIZE = 1000

# Query to fetch key sizes for a batch of observation rows.
# It selects rows greater than :last_id up to :limit rows and
# expands the JSONB data column to measure the size of each key.
QUERY = sqlalchemy.text(
    """
    SELECT b.id, kv.key, pg_column_size(kv.value)::bigint AS size
    FROM (
        SELECT id, data
        FROM observations_v2
        WHERE id > :last_id
        ORDER BY id
        LIMIT :limit
    ) AS b,
    LATERAL jsonb_each(b.data) AS kv(key, value)
    ORDER BY b.id
    """
)


def gather_key_sizes(engine: sqlalchemy.engine.Engine, batch_size: int = BATCH_SIZE) -> dict[str, int]:
    """Iteratively collect total bytes used by each key in ``observations_v2``.

    ``batch_size`` controls how many rows are processed at a time to avoid
    excessive memory use on the database server.
    """
    totals: collections.defaultdict[str, int] = collections.defaultdict(int)
    last_id = 0

    while True:
        with engine.connect() as conn:
            results = conn.execute(QUERY, {"last_id": last_id, "limit": batch_size}).fetchall()

        if not results:
            break

        ids = set()
        for row in results:
            ids.add(row.id)
            totals[row.key] += row.size

        last_id = max(ids)
        if len(ids) < batch_size:
            break

    return dict(totals)


def main() -> None:
    engine = sqlalchemy.create_engine(POSTGRES_DATABASE_URL, future=True)
    key_sizes = gather_key_sizes(engine)
    for key, size in sorted(key_sizes.items(), key=lambda item: item[1], reverse=True):
        print(f"{key}: {size / (1024 * 1024):.2f} MB ({size} bytes)")


if __name__ == "__main__":
    main()
