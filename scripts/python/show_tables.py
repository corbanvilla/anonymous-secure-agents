import sqlalchemy

from src.db.config import POSTGRES_DATABASE_URL

QUERY = """
SELECT 
    tablename 
FROM pg_catalog.pg_tables 
WHERE schemaname = 'public'
ORDER BY tablename;
"""


def main() -> None:
    engine = sqlalchemy.create_engine(POSTGRES_DATABASE_URL)
    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text(QUERY))
        print("Tables in database:")
        print("-" * 20)
        for row in result:
            print(row.tablename)


if __name__ == "__main__":
    main()
