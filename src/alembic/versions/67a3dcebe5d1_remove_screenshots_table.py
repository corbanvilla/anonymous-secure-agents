"""remove screenshots_v2 table

Revision ID: 67a3dcebe5d1
Revises: 4b7d2b2f8c4d
Create Date: 2025-07-05 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "67a3dcebe5d1"
down_revision: Union[str, None] = "4b7d2b2f8c4d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table("screenshots_v2")


def downgrade() -> None:
    op.create_table(
        "screenshots_v2",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "observation_id",
            sa.Integer(),
            sa.ForeignKey("observations_v2.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("data", sa.LargeBinary(), nullable=True),
        sa.Column("filename", sa.String(), nullable=True),
    )
    op.create_index(
        op.f("ix_screenshots_v2_observation_id"),
        "screenshots_v2",
        ["observation_id"],
        unique=False,
    )
