import argparse

import sqlalchemy

from src.db.config import POSTGRES_DATABASE_URL


def list_tables(engine: sqlalchemy.engine.Engine) -> list[str]:
    query = """
    SELECT tablename
    FROM pg_catalog.pg_tables
    WHERE schemaname = 'public'
    ORDER BY tablename
    """
    with engine.connect() as conn:
        rows = conn.execute(sqlalchemy.text(query)).fetchall()
    return [row.tablename for row in rows]


def reindex_table(engine: sqlalchemy.engine.Engine, table: str) -> None:
    quoted = sqlalchemy.inspect(engine).dialect.identifier_preparer.quote(table)
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text(f"REINDEX TABLE {quoted}"))
        print(f"reindexed {table}")


def vacuum_table(engine: sqlalchemy.engine.Engine, table: str) -> None:
    quoted = sqlalchemy.inspect(engine).dialect.identifier_preparer.quote(table)
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(sqlalchemy.text(f"VACUUM FULL {quoted}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table",
        action="append",
        dest="tables",
        help="Specific table(s) to reindex; if omitted all user tables are reindexed",
    )
    args = parser.parse_args()

    engine = sqlalchemy.create_engine(POSTGRES_DATABASE_URL, future=True)

    tables = args.tables if args.tables else list_tables(engine)
    for table in tables:
        print(f"reindexing {table}")
        reindex_table(engine, table)
        print(f"vacuuming {table}")
        vacuum_table(engine, table)


if __name__ == "__main__":
    main()
