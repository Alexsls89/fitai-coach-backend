from typing import Optional
from pydantic import BaseModel


class WorkoutPromptRequest(BaseModel):
    user_id: int
    goal: str
    experience_level: str = "intermédio"
    days_per_week: int = 4
    notes: Optional[str] = None


class AIResponse(BaseModel):
    reply: str


class BodyVisionResponse(BaseModel):
    assessment_id: int
    estimated_body_fat_pct: Optional[float] = None
    analysis: str


class NutriVisionResponse(BaseModel):
    log_id: int
    meal_summary: str
    estimated_calories: Optional[float] = None
    protein_g: Optional[float] = None
    carbs_g: Optional[float] = None
    fat_g: Optional[float] = None
    feedback: str