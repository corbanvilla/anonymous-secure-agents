"""add environment logs column"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "1bf331206322"
down_revision: Union[str, None] = "fb1256ed6725"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("trajectories", sa.Column("environment_logs", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("trajectories", "environment_logs")
