"""
User Preferences & Proactivity Settings API Endpoints.
Allows the user to view and control their preferences:
- Proactivity Level: 'low', 'medium', 'high'
- Assistant Tone, Verbosity, Theme, Timezone
- Active User Goals & Habits
"""

from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import get_current_user, get_db
from app.models.user import User, UserGoal, UserHabit, UserPreference
from app.schemas.preference import (
    ProactivityUpdateRequest,
    UserGoalCreate,
    UserGoalRead,
    UserHabitCreate,
    UserHabitRead,
    UserPreferenceRead,
    UserPreferenceUpdate,
)

preferences_router = APIRouter()
goals_router = APIRouter()
habits_router = APIRouter()
router = preferences_router  # Alias for backward-compatibility


@preferences_router.get(
    "",
    response_model=UserPreferenceRead,
    summary="Get user preferences including proactivity setting",
)
@preferences_router.get(
    "/",
    response_model=UserPreferenceRead,
    include_in_schema=False,
)
async def get_preferences(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Retrieves the active user preferences or initializes defaults."""
    stmt = select(UserPreference).where(UserPreference.user_id == current_user.id)
    pref = (await db.execute(stmt)).scalar_one_or_none()

    if not pref:
        pref = UserPreference(user_id=current_user.id, proactivity="medium")
        db.add(pref)
        await db.commit()
        await db.refresh(pref)

    return pref


@preferences_router.patch(
    "",
    response_model=UserPreferenceRead,
    summary="Update user preferences (proactivity, tone, verbosity)",
)
@preferences_router.patch(
    "/",
    response_model=UserPreferenceRead,
    include_in_schema=False,
)
async def update_preferences(
    data: UserPreferenceUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Updates user preferences in PostgreSQL.
    Configurable proactivity setting: 'low', 'medium', 'high'.
    """
    stmt = select(UserPreference).where(UserPreference.user_id == current_user.id)
    pref = (await db.execute(stmt)).scalar_one_or_none()

    if not pref:
        pref = UserPreference(user_id=current_user.id)
        db.add(pref)

    if data.proactivity is not None:
        if data.proactivity not in ("low", "medium", "high"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Proactivity must be one of: 'low', 'medium', 'high'",
            )
        pref.proactivity = data.proactivity

    if data.assistant_tone is not None:
        pref.assistant_tone = data.assistant_tone.strip()
    if data.verbosity is not None:
        pref.verbosity = data.verbosity.strip()
    if data.theme is not None:
        pref.theme = data.theme.strip()
    if data.language is not None:
        pref.language = data.language.strip()
    if data.voice is not None:
        pref.voice = data.voice.strip()
    if data.timezone is not None:
        pref.timezone = data.timezone.strip()

    await db.commit()
    await db.refresh(pref)
    return pref


@preferences_router.put(
    "/proactivity",
    response_model=UserPreferenceRead,
    summary="Update assistant proactivity level directly",
)
async def set_proactivity(
    data: ProactivityUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Sets proactivity level ('low', 'medium', 'high')."""
    if data.proactivity not in ("low", "medium", "high"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Proactivity must be one of: 'low', 'medium', 'high'",
        )

    stmt = select(UserPreference).where(UserPreference.user_id == current_user.id)
    pref = (await db.execute(stmt)).scalar_one_or_none()
    if not pref:
        pref = UserPreference(user_id=current_user.id)
        db.add(pref)

    pref.proactivity = data.proactivity
    await db.commit()
    await db.refresh(pref)
    return pref


# --- Goals Management ---

@goals_router.get(
    "",
    response_model=List[UserGoalRead],
    summary="List active user goals",
)
@goals_router.get(
    "/",
    response_model=List[UserGoalRead],
    include_in_schema=False,
)
async def list_goals(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Returns active user goals."""
    stmt = (
        select(UserGoal)
        .where(UserGoal.user_id == current_user.id)
        .order_by(UserGoal.created_at.desc())
    )
    res = await db.execute(stmt)
    return res.scalars().all()


@goals_router.post(
    "",
    response_model=UserGoalRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user goal",
)
@goals_router.post(
    "/",
    response_model=UserGoalRead,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
async def create_goal(
    data: UserGoalCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Adds a personal goal for the assistant to track and suggest around."""
    goal = UserGoal(
        user_id=current_user.id,
        category=data.category.lower().strip(),
        title=data.title.strip(),
        description=data.description.strip() if data.description else None,
        priority=data.priority.lower().strip(),
        status=data.status.lower().strip(),
    )
    db.add(goal)
    await db.commit()
    await db.refresh(goal)
    return goal


@goals_router.delete(
    "/{goal_id}",
    summary="Delete a user goal",
)
async def delete_goal(
    goal_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Deletes a user goal by ID."""
    stmt = select(UserGoal).where(
        UserGoal.id == goal_id,
        UserGoal.user_id == current_user.id,
    )
    goal = (await db.execute(stmt)).scalar_one_or_none()
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    await db.delete(goal)
    await db.commit()
    return {"message": "Goal deleted successfully"}


# --- Habits Management ---

@habits_router.get(
    "",
    response_model=List[UserHabitRead],
    summary="List user habits",
)
@habits_router.get(
    "/",
    response_model=List[UserHabitRead],
    include_in_schema=False,
)
async def list_habits(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Returns user habits."""
    stmt = (
        select(UserHabit)
        .where(UserHabit.user_id == current_user.id)
        .order_by(UserHabit.created_at.desc())
    )
    res = await db.execute(stmt)
    return res.scalars().all()


@habits_router.post(
    "",
    response_model=UserHabitRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user habit",
)
@habits_router.post(
    "/",
    response_model=UserHabitRead,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
async def create_habit(
    data: UserHabitCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Adds a habit for the assistant to consider when suggesting plans."""
    habit = UserHabit(
        user_id=current_user.id,
        category=data.category.lower().strip(),
        name=data.name.strip(),
        frequency=data.frequency.lower().strip(),
        description=data.description.strip() if data.description else None,
    )
    db.add(habit)
    await db.commit()
    await db.refresh(habit)
    return habit


@habits_router.delete(
    "/{habit_id}",
    summary="Delete a user habit",
)
async def delete_habit(
    habit_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Deletes a user habit by ID."""
    stmt = select(UserHabit).where(
        UserHabit.id == habit_id,
        UserHabit.user_id == current_user.id,
    )
    habit = (await db.execute(stmt)).scalar_one_or_none()
    if not habit:
        raise HTTPException(status_code=404, detail="Habit not found")
    await db.delete(habit)
    await db.commit()
    return {"message": "Habit deleted successfully"}


# Also nest goals and habits inside preferences_router (/preferences/goals, /preferences/habits)
preferences_router.include_router(goals_router, prefix="/goals", tags=["Goals"])
preferences_router.include_router(habits_router, prefix="/habits", tags=["Habits"])

