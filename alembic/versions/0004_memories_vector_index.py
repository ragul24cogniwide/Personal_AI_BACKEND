"""memories_vector_index

Revision ID: 0004_memories_vector_index
Revises: 0003_conv_messages_user_id
Create Date: 2026-10-06 14:57:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0004_memories_vector_index"
down_revision: Union[str, None] = "0003_conv_messages_user_id"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Ensure vector extension is present
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    # Add HNSW cosine index for pgvector fast approximate nearest neighbor search
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_memories_embedding_hnsw "
        "ON memories USING hnsw (embedding vector_cosine_ops);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_memories_embedding_hnsw;")
