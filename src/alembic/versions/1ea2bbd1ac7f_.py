"""empty message

Revision ID: 1ea2bbd1ac7f
Revises: 67a3dcebe5d1, cd58d6cf1910
Create Date: 2025-07-06 10:32:39.200265

"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "1ea2bbd1ac7f"
down_revision: Union[str, None] = ("67a3dcebe5d1", "cd58d6cf1910")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
