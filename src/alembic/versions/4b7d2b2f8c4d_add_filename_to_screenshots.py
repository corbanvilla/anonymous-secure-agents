"""add filename column to screenshots_v2

Revision ID: 4b7d2b2f8c4d
Revises: 3f3fcd1c30e2
Create Date: 2025-07-04 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "4b7d2b2f8c4d"
down_revision: Union[str, None] = "3f3fcd1c30e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("screenshots_v2", sa.Column("filename", sa.String(), nullable=True))
    op.alter_column("screenshots_v2", "data", nullable=True)


def downgrade() -> None:
    op.alter_column("screenshots_v2", "data", nullable=False)
    op.drop_column("screenshots_v2", "filename")
