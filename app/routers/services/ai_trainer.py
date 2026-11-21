from fastapi import APIRouter, Depends
from sqlmodel import Session
from app.core.database import get_session
from app.models.workout import WorkoutPlan
from app.schemas.ai_schema import AIResponse, WorkoutPromptRequest
from app.services.gemini_service import generate_workout_plan

router = APIRouter(prefix="/ai-trainer", tags=["Módulo A - Treinador IA"])


@router.post("/workout-plan", response_model=AIResponse)
def create_workout_plan(
    payload: WorkoutPromptRequest,
    session: Session = Depends(get_session),
):
    plan_text = generate_workout_plan(
        goal=payload.goal,
        level=payload.experience_level,
        days_per_week=payload.days_per_week,
        notes=payload.notes,
    )
    workout = WorkoutPlan(
        user_id=payload.user_id,
        title=f"Plano {payload.goal} ({payload.days_per_week}x/semana)",
        goal=payload.goal,
        schedule_json=plan_text,
    )
    session.add(workout)
    session.commit()
    session.refresh(workout)
    return AIResponse(reply=plan_text)