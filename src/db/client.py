from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.db.config import POSTGRES_DATABASE_URL
from src.db.functions import FUNCTIONS
from src.db.tables import UNLOGGED_TABLES, Base
from src.db.views import ALL_VIEWS

DATABASE_URL = POSTGRES_DATABASE_URL


# Create an engine
@retry(
    retry=retry_if_exception_type(OperationalError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=0.1, max=2),
)
def safe_commit(session):
    session.commit()


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=20,
    max_overflow=0,
    pool_timeout=30,
    pool_recycle=3600,
    connect_args={
        "options": "-c statement_timeout=60000 -c lock_timeout=15000 -c synchronous_commit=off",
        "connect_timeout": 10,
    },
)

# Session factory
Session = sessionmaker(bind=engine)

# Create functions using a short-lived session
with Session() as init_session:
    for function in FUNCTIONS:
        init_session.execute(text(function.strip()))
    safe_commit(init_session)

# Unlogged tables
with Session() as unlogged_session:
    for table in UNLOGGED_TABLES:
        unlogged_session.execute(text(f"ALTER TABLE {table} SET UNLOGGED;"))
    safe_commit(unlogged_session)

# Views
with Session() as view_session:
    for view in ALL_VIEWS:
        view_session.execute(text(view))
    safe_commit(view_session)


# Create all tables
Base.metadata.create_all(engine)
