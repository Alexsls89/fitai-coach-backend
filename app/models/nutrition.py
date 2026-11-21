from datetime import datetime, timezone
from typing import Optional
from sqlmodel import Field, SQLModel


class NutritionLog(SQLModel, table=True):
    __tablename__ = "nutrition_logs"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id")
    meal_description: Optional[str] = None
    calories: Optional[float] = None
    protein_g: Optional[float] = None
    carbs_g: Optional[float] = None
    fat_g: Optional[float] = None
    ai_feedback: Optional[str] = None
    image_url: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))