"""
Onboarding Endpoints: complete conversational onboarding and status tracking.
Stores structured profile, goals, preferences, and marks the user as onboarded.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import get_current_user, get_db
from app.models.user import User, UserGoal, UserPreference, UserProfile
from app.schemas.onboarding import (
    OnboardingCompleteRequest,
    OnboardingCompleteResponse,
)

router = APIRouter()


@router.post(
    "/complete",
    response_model=OnboardingCompleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Complete conversational onboarding and save structured profile",
)
async def complete_onboarding(
    data: OnboardingCompleteRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Finalize conversational onboarding:
    - Update UserProfile with structured personal, occupational, and interest details
    - Update UserPreference with tone and timezone
    - Insert initial UserGoals
    - Set is_onboarded = True and record timestamp
    """
    result = await db.execute(
        select(User)
        .options(
            selectinload(User.profile),
            selectinload(User.preferences),
            selectinload(User.assistant_settings),
        )
        .where(User.id == current_user.id)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # 1. Update User Profile
    if not user.profile:
        user.profile = UserProfile(user_id=user.id)
        db.add(user.profile)

    if data.name:
        user.profile.name = data.name.strip()
    if data.preferred_name:
        user.profile.preferred_name = data.preferred_name.strip()
    if data.occupation:
        user.profile.occupation = data.occupation.strip()
    if data.communication_style:
        user.profile.communication_style = data.communication_style.strip()

    if data.interests:
        user.profile.interests = data.interests
    if data.hobbies:
        user.profile.hobbies = data.hobbies
    if data.fitness_interests:
        user.profile.fitness_interests = data.fitness_interests
    if data.learning_interests:
        user.profile.learning_interests = data.learning_interests
    if data.work_interests:
        user.profile.work_interests = data.work_interests
    if data.daily_routine:
        user.profile.daily_routine = data.daily_routine

    # 2. Update User Preferences
    if not user.preferences:
        user.preferences = UserPreference(user_id=user.id)
        db.add(user.preferences)

    if data.assistant_tone:
        user.preferences.assistant_tone = data.assistant_tone.strip()
    if data.timezone:
        user.preferences.timezone = data.timezone.strip()

    # 3. Store User Goals
    if data.goals:
        for goal_input in data.goals:
            goal = UserGoal(
                user_id=user.id,
                category=goal_input.category,
                title=goal_input.title.strip(),
                description=goal_input.description,
                priority=goal_input.priority,
                status="in_progress",
            )
            db.add(goal)

    # 4. Mark user as onboarded
    user.is_onboarded = True
    user.onboarding_completed_at = datetime.now(timezone.utc)

    await db.commit()

    # Re-fetch eager loaded user to return full model
    refreshed = await db.execute(
        select(User)
        .options(
            selectinload(User.profile),
            selectinload(User.preferences),
            selectinload(User.assistant_settings),
        )
        .where(User.id == user.id)
    )
    user_final = refreshed.scalar_one()

    return OnboardingCompleteResponse(
        success=True,
        message="Onboarding completed successfully. Welcome to your personal AI assistant!",
        user=user_final,
    )


@router.get(
    "/status",
    summary="Get user onboarding status",
)
async def get_onboarding_status(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Check if the current user has completed conversational onboarding."""
    return {
        "is_onboarded": current_user.is_onboarded,
        "onboarding_completed_at": current_user.onboarding_completed_at,
        "name": current_user.profile.name if current_user.profile else None,
        "preferred_name": current_user.profile.preferred_name if current_user.profile else None,
        "occupation": current_user.profile.occupation if current_user.profile else None,
    }
