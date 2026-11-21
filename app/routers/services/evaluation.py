from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from sqlmodel import Session
from app.database import get_session
from app.models.user import User
from app.dependencies.auth import get_current_user
from app.services.gemini_service import analisar_imagem_com_retry

router = APIRouter(prefix="/api/v1/evaluations", tags=["Evaluations"])

@router.post("/analyze")
async def create_physical_evaluation(
    file: UploadFile = File(...),
    objetivo: str = Form("Hipertrofia e definição"),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    # 1. Trava de Uso Gratuito (Bloqueio antes de gastar tokens do Gemini)
    if not current_user.is_premium and current_user.free_evaluations_used >= 1:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Você já utilizou sua 1ª avaliação física gratuita. Assine o FitAI Premium para avaliações ilimitadas."
        )

    # 2. Validação de tamanho no servidor (garante que o app comprimiu corretamente)
    image_bytes = await file.read()
    if len(image_bytes) > 500 * 1024:  # Trava de segurança de 500 KB
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Imagem muito pesada. Atualize o aplicativo para comprimir a foto automaticamente."
        )

    prompt = (
        f"Atue como treinador físico e especialista em biomecânica do FitAI & Coach. "
        f"Analise a composição corporal desta imagem considerando o objetivo: '{objetivo}'. "
        f"Forneça: 1) Pontos fortes musculares, 2) Pontos a desenvolver, "
        f"3) Sugestão de divisão de treino e 4) Observações posturais."
    )

    # 3. Chamada ao Gemini 1.5 Flash com timeout 60s e retry (2s -> 4s -> 8s)
    resultado_ia = await analisar_imagem_com_retry(
        image_bytes=image_bytes,
        mime_type=file.content_type or "image/jpeg",
        prompt=prompt,
    )

    # 4. Incrementa o contador apenas após sucesso da análise
    if not current_user.is_premium:
        current_user.free_evaluations_used += 1
        session.add(current_user)
        session.commit()
        session.refresh(current_user)

    return {
        "status": "success",
        "analise": resultado_ia,
        "free_evaluations_used": current_user.free_evaluations_used,
        "is_premium": current_user.is_premium,
    }