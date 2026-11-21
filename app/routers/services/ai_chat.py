import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai.errors import APIError
from app.core.config import settings

router = APIRouter()

MODEL_ID = "gemini-1.5-flash"


class PromptRequest(BaseModel):
    prompt: str


@router.post("/api/ai/gerar-resposta")
def gerar_resposta_ia(payload: PromptRequest):
    backoff_delays = [2, 4, 8]
    last_error = ""

    try:
        with genai.Client(api_key=settings.GEMINI_API_KEY, http_options={"timeout": 60.0}) as client:
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model=MODEL_ID,
                        contents=payload.prompt,
                    )
                    return {"status": "sucesso", "resposta": response.text}
                except APIError as e:
                    code = getattr(e, "code", None)
                    last_error = str(e)
                    if code in (429, 503, 529) and attempt < 2:
                        time.sleep(backoff_delays[attempt])
                        continue
                    raise HTTPException(
                        status_code=502,
                        detail=f"Erro na API do Gemini: {last_error}",
                    )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Falha de conexão com o serviço de IA. Detalhes: {str(e)}",
        )