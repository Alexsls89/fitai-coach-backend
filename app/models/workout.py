from datetime import datetime, timezone
from typing import Optional
from sqlmodel import Field, SQLModel


class BodyAssessment(SQLModel, table=True):
    __tablename__ = "body_assessments"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id")
    weight_kg: Optional[float] = None
    body_fat_pct: Optional[float] = None
    muscle_mass_kg: Optional[float] = None
    ai_analysis: Optional[str] = None
    image_url: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WorkoutPlan(SQLModel, table=True):
    __tablename__ = "workout_plans"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id")
    title: str
    goal: Optional[str] = None
    schedule_json: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))