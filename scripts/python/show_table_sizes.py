import sqlalchemy

from src.db.config import POSTGRES_DATABASE_URL

QUERY = """
SELECT
    s.relname AS table_name,
    ps.n_live_tup AS row_count,
    pg_total_relation_size(s.relid) AS total_bytes,
    pg_size_pretty(pg_total_relation_size(s.relid)) AS total_size,
    pg_relation_size(s.relid) AS table_bytes,
    pg_size_pretty(pg_relation_size(s.relid)) AS table_size,
    pg_total_relation_size(s.relid) - pg_relation_size(s.relid) AS index_bytes,
    pg_size_pretty(pg_total_relation_size(s.relid) - pg_relation_size(s.relid)) AS index_size
FROM pg_catalog.pg_statio_user_tables s
JOIN pg_stat_user_tables ps ON ps.relid = s.relid
ORDER BY pg_total_relation_size(s.relid) DESC;
"""


def main() -> None:
    engine = sqlalchemy.create_engine(POSTGRES_DATABASE_URL)
    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text(QUERY))
        for row in result:
            print(
                f"{row.table_name}: {row.total_size} (rows: {row.row_count}, "
                f"table: {row.table_size}, indexes: {row.index_size})"
            )


if __name__ == "__main__":
    main()
