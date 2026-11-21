import base64
import json
import logging
import os
import re
import time
import traceback
from io import BytesIO

import httpx
from fastapi import HTTPException
from google import genai
from google.genai import types
from google.genai.errors import APIError
from PIL import Image

from app.core.config import settings

for noisy_logger in ("tenacity", "httpx", "httpcore", "google_genai"):
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)

logger = logging.getLogger("fitai.vision")

PREFERRED_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-flash-latest",
]


def _get_gemini_api_keys() -> list[str]:
    raw_candidates: list[str] = []

    for attr_name in ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEYS"):
        val = getattr(settings, attr_name, None) or os.getenv(attr_name, "")
        if val:
            if isinstance(val, (list, tuple)):
                raw_candidates.extend(str(item) for item in val)
            else:
                raw_candidates.extend(part for part in str(val).split(","))

    keys: list[str] = []
    for raw in raw_candidates:
        cleaned = raw.strip().strip('"').strip("'")
        if cleaned and cleaned not in keys:
            keys.append(cleaned)

    if not keys:
        raise RuntimeError("Nenhuma GEMINI_API_KEY configurada no arquivo .env.")
    return keys


def _get_httpx_client() -> httpx.Client:
    transport = httpx.HTTPTransport(http2=False, retries=2)
    return httpx.Client(
        transport=transport,
        timeout=httpx.Timeout(120.0, connect=30.0, read=120.0, write=30.0),
        follow_redirects=True,
        limits=httpx.Limits(
            max_connections=1,
            max_keepalive_connections=0,
            keepalive_expiry=0,
        ),
    )


def get_genai_client(api_key: str) -> genai.Client:
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            api_version="v1beta",
            httpx_client=_get_httpx_client(),
        ),
    )


def _close_client_safely(client: genai.Client | None) -> None:
    if client is None:
        return
    try:
        client._api_client._httpx_client.close()
    except Exception:
        pass
    try:
        client.close()
    except Exception:
        pass


def _is_invalid_api_key_error(error_text: str) -> bool:
    lowered = error_text.lower()
    return (
        "api key not valid" in lowered
        or "api_key_invalid" in lowered
        or "invalid api key" in lowered
        or "api key expired" in lowered
    )


def _prepare_image(image_bytes: bytes, max_size: tuple[int, int] = (1024, 1024)) -> Image.Image:
    try:
        img = Image.open(BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail(max_size)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        buffer.seek(0)
        return Image.open(buffer)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Imagem inválida ou corrompida ({type(exc).__name__}): {exc}",
        ) from exc


def _image_to_base64_jpeg(pil_img: Image.Image) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _clean_json_text(raw_text: str) -> str:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _call_openai_gpt4o_fallback(
    prompt: str,
    first_image: Image.Image | None = None,
    images: list[Image.Image] | None = None,
) -> str:
    openai_key = (
        getattr(settings, "OPENAI_API_KEY", None)
        or os.getenv("OPENAI_API_KEY", "")
    ).strip()
    if not openai_key:
        raise RuntimeError(
            "Fallback OpenAI indisponível: OPENAI_API_KEY não configurada no .env."
        )

    target_images = (
        images if images else ([first_image] if first_image is not None else [])
    )
    if not target_images:
        raise RuntimeError("Nenhuma imagem enviada para o fallback OpenAI gpt-4o.")

    content_blocks: list[dict] = [{"type": "text", "text": prompt}]
    for idx, img in enumerate(target_images, start=1):
        b64 = _image_to_base64_jpeg(img)
        content_blocks.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "auto" if idx == 1 else "low",
                },
            }
        )

    print(
        f"[FitAI Vision] Acionando fallback automático OpenAI gpt-4o "
        f"({len(target_images)} foto(s))..."
    )

    with _get_httpx_client() as http_client:
        resp = http_client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openai_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": content_blocks,
                    }
                ],
                "max_tokens": 3500,
                "temperature": 0.35,
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"OpenAI gpt-4o falhou (HTTP {resp.status_code}): {resp.text[:300]}"
            )

        data = resp.json()
        text = (data["choices"][0]["message"]["content"] or "").strip()
        if not text:
            raise RuntimeError("OpenAI gpt-4o retornou resposta vazia.")
        print(f"✅ Sucesso com fallback OpenAI gpt-4o ({len(target_images)} foto(s))!")
        return _clean_json_text(text)


def _build_body_prompt(
    angles_count: int,
    weight_kg: float | None = None,
    height_cm: float | None = None,
    goal: str = "Hipertrofia e Definição",
    level: str = "Intermediário",
    days_per_week: int = 4,
    restrictions: str = "Nenhuma restrição informada",
    uses_mounjaro: bool = False,
    mounjaro_dose: str | None = None,
) -> str:
    angle_labels = ["Frente", "Costas", "Perfil Esquerdo", "Perfil Direito"]
    angles_str = ", ".join(angle_labels[:angles_count]) if angles_count > 0 else "Frente"
    weight_info = f"{weight_kg:.1f} kg" if weight_kg else "Estimar visualmente"
    height_info = f"{height_cm:.0f} cm" if height_cm else "Não informada"

    imc_val_str = "25.0"
    if weight_kg and height_cm and height_cm > 0:
        h_m = height_cm / 100.0
        imc_val_str = f"{weight_kg / (h_m * h_m):.1f}"

    mounjaro_note = ""
    if uses_mounjaro:
        dose_str = (mounjaro_dose or "dose semanal informada").strip()
        mounjaro_note = (
            f"O usuário utiliza Mounjaro (Tirzepatida) na dose {dose_str}. "
            "Ajuste o protocolo para alta densidade proteica (2,1 a 2,4 g/kg), refeições menores, "
            "treino focado em tensão mecânica anti-catabólica e suplementação manipulada de suporte (Psyllium, Magnésio, Complexo B, Enzimas digestivas, Creatina)."
        )

    restrictions_note = ""
    if restrictions and restrictions.strip().lower() not in ("", "nenhuma restrição informada", "nenhuma"):
        restrictions_note = (
            f"RESTRIÇÕES / DORES ARTICULARES DO ALUNO: {restrictions}. "
            "Adapte TODOS os exercícios da planilha para proteger essas articulações e evitar sobrecarga lesiva."
        )

    return (
        "Você é uma junta técnica: Avaliador Biomecânico, Head Coach de Hipertrofia e Nutricionista Esportivo.\n"
        f"Aluno: Ângulos={angles_str} | Peso={weight_info} | Altura={height_info} | IMC={imc_val_str} | "
        f"Objetivo={goal} | Nível={level} | Frequência={days_per_week}x/sem.\n"
        f"{restrictions_note} {mounjaro_note}\n\n"
        "INSTRUÇÕES DE ANÁLISE CORPORAL 360°:\n"
        "- Avalie o corpo inteiro cruzando todas as fotos enviadas (Frente, Costas e Perfis).\n"
        "- No array `muscles`, avalie SEPARADAMENTE tanto os membros superiores quanto cada região dos membros inferiores "
        "(Quadríceps, Posterior de Coxa / Femoral, Glúteos e Panturrilhas), atribuindo nota real de 1 a 10 "
        "(notas >= 8 indicam ponto forte; notas <= 7 indicam foco de hipertrofia).\n\n"
        "REGRA OBRIGATÓRIA DE SAÍDA:\n"
        "- PROIBIDO gerar tabelas Markdown (`| Coluna |`).\n"
        "- Retorne APENAS um objeto JSON válido (sem blocos ```json e sem texto fora do JSON) seguindo EXATAMENTE esta estrutura:\n"
        "{\n"
        '  "shape_score": 87,\n'
        '  "kpis": {\n'
        f'    "imc": "{imc_val_str}",\n'
        '    "imc_status": "Bom Equilíbrio / Massa Magra",\n'
        '    "bf_percent": "16-18%",\n'
        '    "massa_magra_kg": "74,5 kg",\n'
        '    "massa_gorda_kg": "14,9 kg"\n'
        "  },\n"
        '  "muscles": [\n'
        '    {"name": "Peitoral", "score": 8, "percent": 80, "summary": "Bom volume esternal, evoluir porção clavicular", "details": "Diagnóstico técnico detalhado do peitoral superior e médio."},\n'
        '    {"name": "Dorsais", "score": 7, "percent": 70, "summary": "Largura intermediária em V-Taper", "details": "Diagnóstico técnico detalhado de largura e espessura dorsal."},\n'
        '    {"name": "Ombros", "score": 7, "percent": 70, "summary": "Bom deltoide anterior, evoluir lateral e posterior", "details": "Diagnóstico técnico detalhado das três cabeças do deltoide."},\n'
        '    {"name": "Braços", "score": 8, "percent": 80, "summary": "Boa proporção entre bíceps e tríceps", "details": "Diagnóstico técnico detalhado de bíceps, tríceps e antebraço."},\n'
        '    {"name": "Abdômen / Core", "score": 7, "percent": 70, "summary": "Linha de cintura controlada, evoluir definição infra", "details": "Diagnóstico técnico detalhado da parede abdominal e oblíquos."},\n'
        '    {"name": "Quadríceps", "score": 8, "percent": 80, "summary": "Boa base de vasto lateral e reto femoral", "details": "Diagnóstico técnico detalhado de volume e cortes frontais da coxa."},\n'
        '    {"name": "Posterior de Coxa / Femoral", "score": 6, "percent": 60, "summary": "Ponto de atenção: ganhar densidade nos isquiotibiais", "details": "Diagnóstico técnico detalhado do perfil posterior da coxa e transição glúteo-femoral."},\n'
        '    {"name": "Glúteos", "score": 7, "percent": 70, "summary": "Simetria adequada, evoluir projeção superior", "details": "Diagnóstico técnico detalhado de glúteo máximo e médio."},\n'
        '    {"name": "Panturrilhas", "score": 6, "percent": 60, "summary": "Ponto de atenção: aumentar volume de gastrocnêmio e sóleo", "details": "Diagnóstico técnico detalhado da proporção entre panturrilha e coxa."}\n'
        "  ],\n"
        '  "posture": {\n'
        '    "summary": "Leve rotação interna de ombros e alinhamento pélvico estável",\n'
        '    "bullets": [\n'
        '      "Fortalecer trapézio médio/inferior e manguito rotador",\n'
        '      "Mobilidade torácica antes dos treinos de empurrar",\n'
        '      "Ativação de transverso abdominal e glúteo máximo"\n'
        "    ]\n"
        "  },\n"
        '  "workouts": [\n'
        "    {\n"
        '      "day_code": "A",\n'
        '      "title": "Peito + Ombro Lateral + Tríceps",\n'
        '      "duration_min": 65,\n'
        '      "kcal": 480,\n'
        '      "exercises": [\n'
        '        {"name": "Supino Inclinado com Halteres", "sets": "4 Séries", "reps": "8-10 Reps", "rir": "RIR 1-2", "cadence": "3010", "rest_sec": 90, "focus": "Porção clavicular"}\n'
        "      ]\n"
        "    }\n"
        "  ],\n"
        '  "nutrition": {\n'
        '    "daily_targets": {\n'
        '      "kcal": "2.480 kcal",\n'
        '      "protein": "200g",\n'
        '      "protein_g_kg": "2,20 g/kg",\n'
        '      "carbs": "260g",\n'
        '      "fat": "68g",\n'
        '      "fiber": "38g",\n'
        '      "water": "4,0 L"\n'
        "    },\n"
        '    "meals": [\n'
        "      {\n"
        '        "time": "07:00",\n'
        '        "title": "Refeição 01 - Café Metabólico",\n'
        '        "kcal": 450,\n'
        '        "protein_g": 34,\n'
        '        "carbs_g": 45,\n'
        '        "fat_g": 12,\n'
        '        "foods": ["3 Ovos Inteiros", "40g Aveia", "1 Banana", "Café sem açúcar"]\n'
        "      }\n"
        "    ],\n"
        '    "supplements": [\n'
        '      {"name": "Creatina Monohidratada", "dose": "5g / dia", "timing": "Pós-treino", "goal": "Força e volumização celular"}\n'
        "    ],\n"
        '    "compounded_formula": {\n'
        '      "name": "Psyllium + Magnésio Dimalato + Zinco Quelato",\n'
        '      "dose": "1 sachê / cápsula manipulados",\n'
        '      "timing": "Antes de dormir",\n'
        '      "notes": "Suporte intestinal, sono profundo e recuperação hormonal. Validar com médico/nutricionista."\n'
        "    },\n"
        '    "shopping_list": ["Peito de frango", "Ovos inteiros", "Aveia em flocos", "Batata doce", "Arroz integral", "Azeite extra virgem"]\n'
        "  }\n"
        "}\n"
        f"Gere exatamente {days_per_week} divisões de treino em `workouts` (com 5 a 6 exercícios cada, priorizando os pontos fracos encontrados) e 4 a 5 refeições completas em `nutrition.meals`."
    )


def _execute_multimodal_with_fresh_clients(
    images: list[Image.Image],
    prompt: str,
) -> str:
    last_gemini_error = "Nenhum modelo Gemini respondeu."
    api_keys = _get_gemini_api_keys()

    for key_idx, api_key in enumerate(api_keys, start=1):
        masked_key = f"...{api_key[-6:]}" if len(api_key) >= 6 else "***"
        key_invalid = False

        for model_id in PREFERRED_MODELS:
            if key_invalid:
                break

            for attempt in range(2):
                client: genai.Client | None = None
                try:
                    client = get_genai_client(api_key)
                    print(
                        f"[FitAI Vision] Chave {key_idx}/{len(api_keys)} ({masked_key}) | "
                        f"Chamando {model_id} (v1beta)..."
                    )
                    response = client.models.generate_content(
                        model=model_id,
                        contents=[*images, prompt],
                    )
                    text = (getattr(response, "text", None) or "").strip()
                    if text:
                        print(f"✅ Sucesso com {model_id} (Chave {key_idx}/{len(api_keys)})!")
                        return _clean_json_text(text)
                    last_gemini_error = f"{model_id} retornou resposta vazia."
                    print(f"[FitAI Vision] Falha -> {last_gemini_error}")
                    break
                except APIError as exc:
                    code = getattr(exc, "code", None)
                    msg = getattr(exc, "message", None) or str(exc)
                    last_gemini_error = f"Gemini {model_id} [Chave {key_idx}] (HTTP {code}): {msg}"
                    print(f"[FitAI Vision] Falha -> {last_gemini_error}")

                    if _is_invalid_api_key_error(msg):
                        key_invalid = True
                    # Pula imediatamente para o próximo modelo em 404, 429, 503 ou 529
                    break
                except Exception as exc:
                    err_str = str(exc)
                    last_gemini_error = f"Gemini {model_id} [Chave {key_idx}] ({type(exc).__name__}): {err_str}"
                    print(f"[FitAI Vision] Falha -> {last_gemini_error}")

                    if _is_invalid_api_key_error(err_str):
                        key_invalid = True
                        break

                    if ("10054" in err_str or isinstance(exc, httpx.ReadError)) and attempt < 1:
                        time.sleep(1.0)
                        continue
                    break
                finally:
                    _close_client_safely(client)

    try:
        return _call_openai_gpt4o_fallback(prompt=prompt, images=images)
    except Exception as fallback_exc:
        traceback.print_exc()
        raise RuntimeError(f"{last_gemini_error} | {fallback_exc}") from None


def analyze_body_images(
    images: list[Image.Image],
    weight_kg: float | None = None,
    height_cm: float | None = None,
    goal: str = "Hipertrofia e Definição",
    level: str = "Intermediário",
    days_per_week: int = 4,
    restrictions: str = "Nenhuma restrição informada",
    uses_mounjaro: bool = False,
    mounjaro_dose: str | None = None,
) -> dict[str, str]:
    prompt = _build_body_prompt(
        angles_count=len(images),
        weight_kg=weight_kg,
        height_cm=height_cm,
        goal=goal,
        level=level,
        days_per_week=days_per_week,
        restrictions=restrictions,
        uses_mounjaro=uses_mounjaro,
        mounjaro_dose=mounjaro_dose,
    )
    json_text = _execute_multimodal_with_fresh_clients(images=images, prompt=prompt)
    return {
        "analysis_text": json_text,
        "analysis_markdown": json_text,
    }


def analyze_meal_image(
    image_bytes: bytes,
    meal_description: str | None = None,
    description: str | None = None,
) -> str:
    image = _prepare_image(image_bytes, max_size=(1024, 1024))
    desc = meal_description or description
    desc_info = f"Contexto: {desc}." if desc else ""

    prompt = (
        f"Nutricionista Esportivo. {desc_info} Analise esta refeição e responda APENAS em JSON válido "
        "(sem tabelas Markdown):\n"
        '{"meal_name": "Prato Escaneado", "kcal": 520, "protein_g": 45, "carbs_g": 50, "fat_g": 12, '
        '"foods": ["Alimento 1", "Alimento 2"], "tip": "Ajuste rápido recomendado"}'
    )

    return _execute_multimodal_with_fresh_clients(images=[image], prompt=prompt)


analyze_food_image = analyze_meal_image