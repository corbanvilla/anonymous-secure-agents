"""add model and defense columns to function_timing

Revision ID: 3f3fcd1c30e2
Revises: fb1256ed6725
Create Date: 2025-07-01 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "3f3fcd1c30e2"
down_revision: Union[str, None] = "fb1256ed6725"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("function_timings", sa.Column("model_name", sa.String(), nullable=True))
    op.add_column("function_timings", sa.Column("defense_name", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("function_timings", "defense_name")
    op.drop_column("function_timings", "model_name")
