"""add_user_id_to_conversation_messages

Revision ID: 0003_conv_messages_user_id
Revises: 0002_user_onboarding
Create Date: 2026-10-06 11:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0003_conv_messages_user_id"
down_revision: Union[str, None] = "0002_user_onboarding"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add user_id column as nullable initially for smooth migration
    op.add_column(
        "conversation_messages",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    # 2. Backfill existing messages from their parent conversation's user_id
    op.execute(
        """
        UPDATE conversation_messages cm
        SET user_id = c.user_id
        FROM conversations c
        WHERE cm.conversation_id = c.id
        """
    )
    # 3. Alter user_id to NOT NULL
    op.alter_column("conversation_messages", "user_id", nullable=False)
    # 4. Add foreign key constraint to users table
    op.create_foreign_key(
        "fk_conversation_messages_user_id_users",
        "conversation_messages",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    # 5. Add index on user_id
    op.create_index("ix_conversation_messages_user_id", "conversation_messages", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_conversation_messages_user_id", table_name="conversation_messages")
    op.drop_constraint("fk_conversation_messages_user_id_users", "conversation_messages", type_="foreignkey")
    op.drop_column("conversation_messages", "user_id")
