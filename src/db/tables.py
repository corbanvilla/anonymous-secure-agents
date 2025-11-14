"""
This module defines the tables used in the database.

It can be used as a helpful reference for the database schema.
"""

from datetime import datetime, timezone
from typing import List

from sqlalchemy import (
    ARRAY,
    Boolean,
    Column,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
)
from sqlalchemy.dialects.postgresql import JSON, JSONB
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, index=True)
    description = Column(String, nullable=True)
    username = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    config = Column(JSONB, nullable=False)
    task_ids: Column[List[str]] = Column(ARRAY(String), index=True)
    is_running = Column(Boolean, default=False)
    trajectories = relationship("Trajectory", back_populates="experiment", cascade="all, delete-orphan")
    hidden = Column(Boolean, default=False)
    favorite = Column(Boolean, default=False)

    __table_args__ = (Index("ix_experiments_task_ids", "task_ids", postgresql_using="gin"),)


class Trajectory(Base):
    __tablename__ = "trajectories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True)
    task_id = Column(String, nullable=False, index=True)
    trajectory = Column(JSONB, nullable=False)
    chat = Column(String, nullable=True)
    success = Column(Boolean, Computed("(trajectory->>'reward')::float > 0", persisted=True))
    experiment = relationship("Experiment", back_populates="trajectories", passive_deletes=True)
    observations = relationship("Observation", back_populates="trajectory", cascade="all, delete-orphan")

    __table_args__ = (Index("uix_experiment_task", "experiment_id", "task_id", unique=True),)


class Observation(Base):
    __tablename__ = "observations_v2"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trajectory_id = Column(Integer, ForeignKey("trajectories.id", ondelete="CASCADE"), nullable=False, index=True)
    data = Column(JSON, nullable=False)
    step_number = Column(Integer, nullable=True)
    trajectory = relationship("Trajectory", back_populates="observations", passive_deletes=True)


class FunctionTiming(Base):
    """Record the duration of LLM API calls."""

    __tablename__ = "function_timings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    function_name = Column(String, nullable=False)
    duration = Column(Float, nullable=False)
    model_name = Column(String, nullable=True)
    defense_name = Column(String, nullable=True)


class ErrorLog(Base):
    """Store errors raised during attack or defense execution."""

    __tablename__ = "error_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String, nullable=True, index=True)
    phase = Column(String, nullable=False)
    message = Column(String, nullable=False)
    traceback = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    username = Column(String)


class TaskDataset(Base):
    """Store named collections of task IDs."""

    __tablename__ = "task_datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, index=True, nullable=False)
    tasks: Column[List[str]] = Column(ARRAY(String), nullable=False)

    __table_args__ = (Index("ix_task_datasets_tasks", "tasks", postgresql_using="gin"),)


class DefenseHarnessExperiment(Base):
    __tablename__ = "defense_harness_experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, index=True)
    description = Column(String, nullable=True)
    username = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    config = Column(JSONB, nullable=False)
    reference_experiment_id = Column(
        Integer, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    skip_observation_ids: Column[List[int]] = Column(ARRAY(Integer), nullable=False)
    experiment = relationship("Experiment")

    __table_args__ = (
        Index("ix_defense_harness_experiments_skip_observation_ids", "skip_observation_ids", postgresql_using="gin"),
        Index("ix_defense_harness_experiments_ref_exp_id", "reference_experiment_id"),
    )


class DefenseHarnessStep(Base):
    __tablename__ = "defense_harness_steps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    observation_id = Column(Integer, ForeignKey("observations_v2.id", ondelete="CASCADE"), nullable=False, index=True)
    full_action = Column(String, nullable=False)
    function = Column(String, nullable=False)
    required_bid = Column(Integer, nullable=False)
    allowed_bids: Column[List[int]] = Column(ARRAY(Integer), nullable=False)
    all_bids: Column[List[int]] = Column(ARRAY(Integer), nullable=False)
    success = Column(Boolean, Computed("required_bid = ANY(allowed_bids)", persisted=True))
    error_message = Column(String, nullable=True)
    llm_logs = Column(JSONB, nullable=True)
    relevant_cap_set = Column(Boolean, nullable=True)
    async_messages_stats = Column(JSONB, nullable=True)

    # Computed columns for token usage and timing
    input_tokens = Column(Integer, Computed("sum_input_tokens(llm_logs)", persisted=True))
    cached_input_tokens = Column(Integer, Computed("sum_cached_input_tokens(llm_logs)", persisted=True))
    output_tokens = Column(Integer, Computed("sum_output_tokens(llm_logs)", persisted=True))
    total_request_time = Column(Numeric, Computed("sum_request_duration(llm_logs)", persisted=True))

    defense_experiment_id = Column(
        Integer, ForeignKey("defense_harness_experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )

    defense_experiment = relationship("DefenseHarnessExperiment")
    observation = relationship("Observation")

    __table_args__ = (Index("ix_defense_harness_steps_obs_id_id", "observation_id", "id"),)


UNLOGGED_TABLES = [FunctionTiming.__tablename__, ErrorLog.__tablename__]
