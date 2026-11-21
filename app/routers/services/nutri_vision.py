from typing import Optional
from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlmodel import Session
from app.core.database import get_session
from app.models.nutrition import NutritionLog
from app.schemas.ai_schema import NutriVisionResponse
from app.services.vision_service import analyze_meal_image

router = APIRouter(prefix="/nutri-vision", tags=["Módulo C - Nutricionista IA"])


@router.post("/analyze", response_model=NutriVisionResponse)
async def analyze_meal(
    user_id: int = Form(...),
    meal_description: Optional[str] = Form(None),
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    image_bytes = await file.read()
    feedback_text = analyze_meal_image(image_bytes=image_bytes, description=meal_description)

    log = NutritionLog(
        user_id=user_id,
        meal_description=meal_description or file.filename,
        ai_feedback=feedback_text,
        image_url=file.filename,
    )
    session.add(log)
    session.commit()
    session.refresh(log)

    return NutriVisionResponse(
        log_id=log.id,
        meal_summary=log.meal_description or "Refeição analisada",
        feedback=feedback_text,
    )