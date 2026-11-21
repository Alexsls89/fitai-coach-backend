import time
from fastapi import HTTPException
from google import genai
from google.genai.errors import APIError
from app.core.config import settings

GEMINI_MODEL_NAME = "gemini-1.5-flash"


def _get_client() -> genai.Client:
    api_key = (settings.GEMINI_API_KEY or "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY não configurada no arquivo .env.",
        )
    return genai.Client(api_key=api_key, http_options={"timeout": 60.0})


def generate_workout_plan(
    goal: str,
    level: str = "intermediário",
    days_per_week: int = 4,
    notes: str | None = None,
) -> str:
    prompt = (
        f"Atue como Head Coach de Musculação do FitAI & Coach. "
        f"Monte um plano de treino completo para objetivo: '{goal}', "
        f"nível: '{level}', frequência: {days_per_week}x por semana. "
        f"Observações adicionais: {notes or 'Nenhuma'}. "
        f"Estruture em tabelas Markdown claras com Exercício, Séries, Repetições, Descanso e Dica Técnica."
    )

    backoff_delays = [2, 4, 8]
    last_error = "Falha desconhecida."

    with _get_client() as client:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL_NAME,
                    contents=prompt,
                )
                text = (getattr(response, "text", None) or "").strip()
                if text:
                    return text
                last_error = "Resposta vazia da IA."
            except APIError as e:
                code = getattr(e, "code", None)
                last_error = f"HTTP {code}: {str(e)}"
                if code in (429, 503, 529) and attempt < 2:
                    time.sleep(backoff_delays[attempt])
                    continue
                raise HTTPException(status_code=502, detail=f"Erro na API do Gemini: {last_error}")
            except Exception as e:
                last_error = str(e)
                if attempt < 2:
                    time.sleep(backoff_delays[attempt])
                    continue
                raise HTTPException(status_code=504, detail=f"Timeout ou falha de rede: {last_error}")

    raise HTTPException(status_code=503, detail=f"Serviço indisponível após 3 tentativas ({last_error})")