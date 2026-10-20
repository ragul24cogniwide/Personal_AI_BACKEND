"""tool_permissions_last_used_at

Revision ID: 0005_tool_permissions_last_used
Revises: 0004_memories_vector_index
Create Date: 2026-10-07 12:45:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0005_tool_permissions_last_used"
down_revision: Union[str, None] = "0004_memories_vector_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add last_used_at timestamp column to tool_permissions
    op.execute("ALTER TABLE tool_permissions ADD COLUMN IF NOT EXISTS last_used_at TIMESTAMP;")
    # Add is_sensitive flag column to tool_permissions
    op.execute("ALTER TABLE tool_permissions ADD COLUMN IF NOT EXISTS is_sensitive BOOLEAN DEFAULT FALSE;")


def downgrade() -> None:
    op.execute("ALTER TABLE tool_permissions DROP COLUMN IF EXISTS last_used_at;")
    op.execute("ALTER TABLE tool_permissions DROP COLUMN IF EXISTS is_sensitive;")
