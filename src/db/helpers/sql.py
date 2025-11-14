from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import text


def to_sql(query):
    return str(query.statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))


def from_sql(query):
    return text(query)
