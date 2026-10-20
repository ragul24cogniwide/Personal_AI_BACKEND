"""add_user_onboarding_fields

Revision ID: 0002_user_onboarding
Revises: 0001_initial_schema
Create Date: 2026-10-04 18:50:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0002_user_onboarding"
down_revision: Union[str, None] = "0001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. users table
    op.add_column(
        "users",
        sa.Column("is_onboarded", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    )
    op.add_column(
        "users",
        sa.Column("onboarding_completed_at", sa.DateTime(timezone=True), nullable=True),
    )

    # 2. user_profiles table
    op.add_column(
        "user_profiles",
        sa.Column("interests", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False),
    )
    op.add_column(
        "user_profiles",
        sa.Column("hobbies", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False),
    )
    op.add_column(
        "user_profiles",
        sa.Column("fitness_interests", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False),
    )
    op.add_column(
        "user_profiles",
        sa.Column("learning_interests", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False),
    )
    op.add_column(
        "user_profiles",
        sa.Column("work_interests", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False),
    )
    op.add_column(
        "user_profiles",
        sa.Column("daily_routine", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
    )


def downgrade() -> None:
    op.drop_column("user_profiles", "daily_routine")
    op.drop_column("user_profiles", "work_interests")
    op.drop_column("user_profiles", "learning_interests")
    op.drop_column("user_profiles", "fitness_interests")
    op.drop_column("user_profiles", "hobbies")
    op.drop_column("user_profiles", "interests")
    op.drop_column("users", "onboarding_completed_at")
    op.drop_column("users", "is_onboarded")
