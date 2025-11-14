"""empty message

Revision ID: f21dfa794c89
Revises: 1bf331206322, 3f3fcd1c30e2
Create Date: 2025-07-03 15:29:02.621683

"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "f21dfa794c89"
down_revision: Union[str, None] = ("1bf331206322", "3f3fcd1c30e2")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
