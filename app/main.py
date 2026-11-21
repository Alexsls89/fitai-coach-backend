import asyncio
import hashlib
import importlib.util
import json
import os
import random
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import Field, Session, SQLModel, create_engine, select


def _load_vision_module():
    app_dir = Path(__file__).resolve().parent
    for py_file in app_dir.rglob("*.py"):
        if py_file.name == "main.py":
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "def analyze_body_images" in content:
            rel_parts = py_file.relative_to(app_dir.parent).with_suffix("").parts
            mod_name = ".".join(rel_parts)
            spec = importlib.util.spec_from_file_location(mod_name, py_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
    raise RuntimeError(
        f"Nenhum arquivo com 'def analyze_body_images' encontrado em {app_dir}"
    )


vision_service = _load_vision_module()

# Banco de Dados (SQLite local automático agora -> PostgreSQL na nuvem via DATABASE_URL)
_raw_db_url = (
    getattr(getattr(vision_service, "settings", None), "DATABASE_URL", None)
    or os.getenv("DATABASE_URL", "")
    or "sqlite:///./fitai_coach.db"
).strip()

if _raw_db_url.startswith("postgres://"):
    _raw_db_url = _raw_db_url.replace("postgres://", "postgresql://", 1)

_connect_args = {"check_same_thread": False} if _raw_db_url.startswith("sqlite") else {}
engine = create_engine(_raw_db_url, echo=False, connect_args=_connect_args)


# ==============================================================================
# MODELOS DO BANCO DE DADOS (USUÁRIOS + AVALIAÇÕES)
# ==============================================================================


class UserAccount(SQLModel, table=True):
    __tablename__ = "user_accounts"

    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    password_hash: str
    full_name: str
    role: str = Field(default="student")  # "student" ou "coach"
    coach_code: Optional[str] = Field(default=None, index=True)  # Ex: COACH-742 (para coach)
    linked_coach_code: Optional[str] = Field(default=None, index=True)  # Coach vinculado ao aluno
    specialty: Optional[str] = Field(default=None)  # "personal", "nutricionista", "ambos"
    professional_registry: Optional[str] = Field(default=None)  # CREF / CRN
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BodyEvaluation(SQLModel, table=True):
    __tablename__ = "body_evaluations"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_key: str = Field(index=True)
    user_name: str = Field(default="Aluno FitAI")
    weight_kg: float = Field(default=89.4)
    height_cm: Optional[float] = Field(default=None)
    bf_percent: str = Field(default="15,0%")
    imc: str = Field(default="25,0")
    lean_mass_kg: str = Field(default="75,0 kg")
    fat_mass_kg: str = Field(default="14,0 kg")
    shape_score: int = Field(default=85)
    goal: str = Field(default="Hipertrofia & Definição")
    level: str = Field(default="Intermediário")
    days_per_week: int = Field(default=4)
    comparative_summary: str = Field(default="")
    muscle_scores_json: str = Field(default="{}")
    protocol_json: str = Field(default="{}")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def _create_db_tables() -> None:
    SQLModel.metadata.create_all(engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _create_db_tables()
    print(f"[FitAI DB] Banco conectado e tabelas prontas ({_raw_db_url.split('://')[0]}).")
    yield


app = FastAPI(title="FIT AI & COACH PERFORMANCE LAB API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# UTILITÁRIOS DE SEGURANÇA E CÓDIGO DO COACH
# ==============================================================================


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
    )
    return f"{salt}${dk.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, hex_hash = stored_hash.split("$", 1)
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
        )
        return secrets.compare_digest(dk.hex(), hex_hash)
    except Exception:
        return False


def _generate_unique_coach_code(session: Session) -> str:
    for _ in range(40):
        candidate = f"COACH-{random.randint(100, 999)}"
        exists = session.exec(
            select(UserAccount).where(UserAccount.coach_code == candidate)
        ).first()
        if not exists:
            return candidate
    return f"COACH-{random.randint(1000, 9999)}"


def _serialize_user(user: UserAccount, coach_name: str | None = None) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role,
        "coach_code": user.coach_code,
        "linked_coach_code": user.linked_coach_code,
        "linked_coach_name": coach_name,
        "specialty": user.specialty,
        "professional_registry": user.professional_registry,
    }


# ==============================================================================
# ROTAS DE AUTENTICAÇÃO & VÍNCULO COACH ↔ ALUNO
# ==============================================================================


class RegisterPayload(BaseModel):
    full_name: str
    email: str
    password: str
    role: str = "student"  # "student" ou "coach"
    coach_code: Optional[str] = None  # Código opcional digitado pelo aluno
    specialty: Optional[str] = None  # Para coach: personal / nutricionista / ambos
    professional_registry: Optional[str] = None  # Para coach: CREF / CRN


class LoginPayload(BaseModel):
    email: str
    password: str


class LinkCoachPayload(BaseModel):
    student_email: str
    coach_code: str


import re


def _validate_professional_registry(specialty: str, registry: str) -> str:
    clean_reg = registry.strip().upper()
    if not clean_reg:
        raise HTTPException(
            status_code=400,
            detail="Informe seu registro profissional (CREF ou CRN) para criar uma conta de Coach.",
        )

    cref_pattern = re.compile(r"^(CREF[\s\-]?)?\d{4,7}[\s\-]?[GP]/[A-Z]{2}$")
    crn_pattern = re.compile(r"^CRN[\s\-]?\d{1,2}[\s\-/]\d{3,7}(/[A-Z]{1,2})?$")

    spec_lower = specialty.strip().lower()
    parts = [p.strip() for p in re.split(r"[,;+]", clean_reg) if p.strip()]

    if "personal" in spec_lower and "nutri" not in spec_lower:
        if not cref_pattern.match(clean_reg):
            raise HTTPException(
                status_code=400,
                detail="CREF inválido. Use o formato oficial (ex.: CREF 012345-G/PE).",
            )
    elif "nutri" in spec_lower and "personal" not in spec_lower:
        if not crn_pattern.match(clean_reg):
            raise HTTPException(
                status_code=400,
                detail="CRN inválido. Use o formato oficial (ex.: CRN-6 12345).",
            )
    else:
        all_valid = len(parts) > 0 and all(
            cref_pattern.match(p) or crn_pattern.match(p) for p in parts
        )
        if not all_valid:
            raise HTTPException(
                status_code=400,
                detail="Registro inválido. Informe CREF (ex.: CREF 012345-G/PE) ou CRN (ex.: CRN-6 12345).",
            )

    return clean_reg


@app.post("/auth/register")
def register_user(payload: RegisterPayload):
    clean_email = payload.email.strip().lower()
    clean_name = payload.full_name.strip()
    clean_role = payload.role.strip().lower()

    if not clean_name or len(clean_name) < 2:
        raise HTTPException(status_code=400, detail="Informe seu nome completo.")
    if "@" not in clean_email or "." not in clean_email:
        raise HTTPException(status_code=400, detail="Informe um e-mail válido.")
    if len(payload.password.strip()) < 4:
        raise HTTPException(
            status_code=400, detail="A senha deve ter pelo menos 4 caracteres."
        )
    if clean_role not in ("student", "coach"):
        clean_role = "student"

    with Session(engine) as session:
        existing = session.exec(
            select(UserAccount).where(UserAccount.email == clean_email)
        ).first()
        if existing:
            raise HTTPException(
                status_code=409,
                detail="Este e-mail já possui cadastro. Use a opção Entrar.",
            )

        generated_coach_code: str | None = None
        linked_coach: str | None = None
        linked_coach_name: str | None = None
        clean_specialty: str | None = None
        validated_registry: str | None = None

        if clean_role == "coach":
            clean_specialty = (payload.specialty or "").strip() or "Consultoria Completa"
            validated_registry = _validate_professional_registry(
                clean_specialty, payload.professional_registry or ""
            )

            duplicate_registry = session.exec(
                select(UserAccount).where(
                    UserAccount.role == "coach",
                    UserAccount.professional_registry == validated_registry,
                )
            ).first()
            if duplicate_registry:
                raise HTTPException(
                    status_code=409,
                    detail="Este registro profissional (CREF/CRN) já está vinculado a outra conta.",
                )

            generated_coach_code = _generate_unique_coach_code(session)
        else:
            raw_code = (payload.coach_code or "").strip().upper()
            if raw_code:
                coach_user = session.exec(
                    select(UserAccount).where(
                        UserAccount.role == "coach",
                        UserAccount.coach_code == raw_code,
                    )
                ).first()
                if not coach_user:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Código '{raw_code}' não encontrado. Verifique com seu Coach ou deixe em branco.",
                    )
                linked_coach = coach_user.coach_code
                linked_coach_name = coach_user.full_name

        new_user = UserAccount(
            email=clean_email,
            password_hash=_hash_password(payload.password.strip()),
            full_name=clean_name,
            role=clean_role,
            coach_code=generated_coach_code,
            linked_coach_code=linked_coach,
            specialty=clean_specialty,
            professional_registry=validated_registry,
        )
        session.add(new_user)
        session.commit()
        session.refresh(new_user)

        return {
            "message": "Conta criada com sucesso!",
            "user": _serialize_user(new_user, coach_name=linked_coach_name),
        }


@app.post("/auth/login")
def login_user(payload: LoginPayload):
    clean_email = payload.email.strip().lower()
    with Session(engine) as session:
        user = session.exec(
            select(UserAccount).where(UserAccount.email == clean_email)
        ).first()
        if not user or not _verify_password(payload.password.strip(), user.password_hash):
            raise HTTPException(
                status_code=401,
                detail="E-mail ou senha incorretos. Verifique seus dados ou crie uma conta.",
            )

        linked_coach_name: str | None = None
        if user.linked_coach_code:
            coach_user = session.exec(
                select(UserAccount).where(
                    UserAccount.coach_code == user.linked_coach_code
                )
            ).first()
            if coach_user:
                linked_coach_name = coach_user.full_name

        return {
            "message": "Login realizado com sucesso!",
            "user": _serialize_user(user, coach_name=linked_coach_name),
        }


@app.post("/student/link-coach")
def link_student_to_coach(payload: LinkCoachPayload):
    clean_email = payload.student_email.strip().lower()
    raw_code = payload.coach_code.strip().upper()

    with Session(engine) as session:
        student = session.exec(
            select(UserAccount).where(UserAccount.email == clean_email)
        ).first()
        if not student:
            raise HTTPException(status_code=404, detail="Aluno não encontrado.")

        coach = session.exec(
            select(UserAccount).where(
                UserAccount.role == "coach",
                UserAccount.coach_code == raw_code,
            )
        ).first()
        if not coach:
            raise HTTPException(
                status_code=404,
                detail=f"Nenhum Coach encontrado com o código '{raw_code}'.",
            )

        student.linked_coach_code = coach.coach_code
        session.add(student)
        session.commit()
        session.refresh(student)

        return {
            "message": f"Vinculado com sucesso ao profissional {coach.full_name}!",
            "user": _serialize_user(student, coach_name=coach.full_name),
        }


# ==============================================================================
# HISTÓRICO DE AVALIAÇÕES E VISÃO DO COACH
# ==============================================================================


def _format_history_lines(records: list[BodyEvaluation]) -> str:
    if not records:
        return ""
    lines: list[str] = []
    for rec in records:
        date_str = rec.created_at.strftime("%d/%m/%Y")
        try:
            scores = json.loads(rec.muscle_scores_json or "{}")
            scores_txt = ", ".join(f"{k}: {v}/10" for k, v in scores.items())
        except Exception:
            scores_txt = ""
        lines.append(
            f"- Data {date_str} | Peso: {rec.weight_kg:.1f}kg | BF: {rec.bf_percent} | "
            f"Massa Magra: {rec.lean_mass_kg} | Massa Gorda: {rec.fat_mass_kg} | "
            f"Score: {rec.shape_score}/100 | Notas: [{scores_txt}] | Parecer: {rec.comparative_summary}"
        )
    return "\n".join(lines)


def _extract_and_save_evaluation(
    user_key: str,
    user_name: str,
    weight_kg: float,
    height_cm: float | None,
    goal: str,
    level: str,
    days_per_week: int,
    raw_json_text: str,
) -> BodyEvaluation:
    decoded: dict = {}
    start_idx = raw_json_text.find("{")
    end_idx = raw_json_text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            decoded = json.loads(raw_json_text[start_idx : end_idx + 1])
        except Exception:
            decoded = {}

    kpis = decoded.get("kpis") if isinstance(decoded.get("kpis"), dict) else {}
    shape_score = int(decoded.get("shape_score") or 85)
    bf_percent = str(kpis.get("bf_percent") or "15,0%")
    imc_val = str(kpis.get("imc") or "25,0")
    lean_mass = str(kpis.get("massa_magra_kg") or f"{weight_kg * 0.84:.1f} kg")
    fat_mass = str(kpis.get("massa_gorda_kg") or f"{weight_kg * 0.16:.1f} kg")
    comp_summary = str(
        decoded.get("comparative_analysis")
        or "Avaliação corporal 360° registrada como referência do ciclo."
    )

    muscle_scores: dict[str, int] = {}
    if isinstance(decoded.get("muscles"), list):
        for m in decoded["muscles"]:
            if isinstance(m, dict) and m.get("name"):
                try:
                    muscle_scores[str(m["name"])] = int(m.get("score") or 7)
                except (ValueError, TypeError):
                    muscle_scores[str(m["name"])] = 7

    record = BodyEvaluation(
        user_key=user_key,
        user_name=user_name,
        weight_kg=weight_kg,
        height_cm=height_cm,
        bf_percent=bf_percent,
        imc=imc_val,
        lean_mass_kg=lean_mass,
        fat_mass_kg=fat_mass,
        shape_score=shape_score,
        goal=goal,
        level=level,
        days_per_week=days_per_week,
        comparative_summary=comp_summary,
        muscle_scores_json=json.dumps(muscle_scores, ensure_ascii=False),
        protocol_json=json.dumps(decoded, ensure_ascii=False) if decoded else raw_json_text,
    )

    with Session(engine) as session:
        session.add(record)
        session.commit()
        session.refresh(record)

    return record


@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "FIT AI & COACH PERFORMANCE LAB",
        "database": _raw_db_url.split("://")[0],
    }


@app.get("/body-vision/history")
@app.get("/api/v1/body-vision/history")
def get_body_history(
    user_email: str = Query(default="aluno_padrao"),
):
    user_key = user_email.strip().lower() or "aluno_padrao"
    with Session(engine) as session:
        stmt = (
            select(BodyEvaluation)
            .where(BodyEvaluation.user_key == user_key)
            .order_by(BodyEvaluation.created_at.asc())
        )
        records = list(session.exec(stmt).all())

    history_items = []
    for rec in records:
        try:
            scores = json.loads(rec.muscle_scores_json or "{}")
        except Exception:
            scores = {}
        history_items.append(
            {
                "id": rec.id,
                "date": rec.created_at.isoformat(),
                "weight_kg": rec.weight_kg,
                "height_cm": rec.height_cm,
                "bf_percent": rec.bf_percent,
                "imc": rec.imc,
                "lean_mass_kg": rec.lean_mass_kg,
                "fat_mass_kg": rec.fat_mass_kg,
                "shape_score": rec.shape_score,
                "comparative_summary": rec.comparative_summary,
                "muscle_scores": scores,
            }
        )

    latest_protocol = None
    if records and records[-1].protocol_json:
        try:
            latest_protocol = json.loads(records[-1].protocol_json)
        except Exception:
            latest_protocol = None

    return {
        "user_key": user_key,
        "count": len(history_items),
        "evaluations": history_items,
        "latest_protocol": latest_protocol,
    }


@app.get("/coach/students")
def list_coach_students(coach_code: Optional[str] = Query(default=None)):
    clean_code = (coach_code or "").strip().upper()
    with Session(engine) as session:
        if clean_code:
            linked_users = list(
                session.exec(
                    select(UserAccount).where(
                        UserAccount.role == "student",
                        UserAccount.linked_coach_code == clean_code,
                    )
                ).all()
            )
            students_out = []
            for st_user in linked_users:
                evals = list(
                    session.exec(
                        select(BodyEvaluation)
                        .where(BodyEvaluation.user_key == st_user.email)
                        .order_by(BodyEvaluation.created_at.desc())
                    ).all()
                )
                latest = evals[0] if evals else None
                students_out.append(
                    {
                        "user_key": st_user.email,
                        "user_name": st_user.full_name,
                        "latest_weight_kg": latest.weight_kg if latest else 80.0,
                        "latest_bf_percent": latest.bf_percent if latest else "15,0%",
                        "latest_shape_score": latest.shape_score if latest else 80,
                        "goal": latest.goal if latest else "Hipertrofia & Definição",
                        "level": latest.level if latest else "Intermediário",
                        "days_per_week": latest.days_per_week if latest else 4,
                        "last_evaluation_at": latest.created_at.isoformat()
                        if latest
                        else st_user.created_at.isoformat(),
                        "evaluations_count": len(evals),
                    }
                )
            return {"coach_code": clean_code, "students": students_out}

        stmt = select(BodyEvaluation).order_by(BodyEvaluation.created_at.desc())
        all_evals = list(session.exec(stmt).all())

    by_user: dict[str, dict] = {}
    for ev in all_evals:
        if ev.user_key not in by_user:
            by_user[ev.user_key] = {
                "user_key": ev.user_key,
                "user_name": ev.user_name,
                "latest_weight_kg": ev.weight_kg,
                "latest_bf_percent": ev.bf_percent,
                "latest_shape_score": ev.shape_score,
                "goal": ev.goal,
                "level": ev.level,
                "days_per_week": ev.days_per_week,
                "last_evaluation_at": ev.created_at.isoformat(),
                "evaluations_count": 1,
            }
        else:
            by_user[ev.user_key]["evaluations_count"] += 1

    return {"students": list(by_user.values())}


# ==============================================================================
# ROTAS DE IA MULTIMODAL (SHAPE 360°, TREINO E NUTRIÇÃO)
# ==============================================================================


@app.post("/body-vision/analyze")
@app.post("/api/v1/body-vision/analyze")
@app.post("/analyze-body-360")
async def analyze_body_360(request: Request):
    form = await request.form()
    print(f"[BodyVision] Campos recebidos: {[(k, type(v).__name__) for k, v in form.multi_items()]}")

    pil_images = []
    for key, value in form.multi_items():
        if hasattr(value, "read"):
            data = await value.read()
            if data:
                pil_images.append(vision_service._prepare_image(data))

    if not pil_images:
        raise HTTPException(
            status_code=400,
            detail="Nenhuma imagem válida foi recebida pelo servidor. Verifique o envio da foto.",
        )

    user_key = (
        str(form.get("user_email") or form.get("user_id") or "aluno_padrao")
        .strip()
        .lower()
    )
    user_name = str(form.get("user_name") or "Aluno FitAI").strip()

    raw_weight = form.get("weight_kg")
    weight_kg = 89.4
    if raw_weight not in (None, ""):
        try:
            weight_kg = float(str(raw_weight).replace(",", "."))
        except (ValueError, TypeError):
            weight_kg = 89.4

    raw_height = form.get("height_cm")
    height_cm = None
    if raw_height not in (None, ""):
        try:
            height_cm = float(str(raw_height).replace(",", "."))
        except (ValueError, TypeError):
            height_cm = None

    raw_days = form.get("days_per_week")
    try:
        days_per_week = int(raw_days) if raw_days not in (None, "") else 4
    except (ValueError, TypeError):
        days_per_week = 4

    goal = str(form.get("goal") or "Hipertrofia e Definição")
    level = str(form.get("level") or "Intermediário")
    restrictions = str(form.get("restrictions") or "Nenhuma restrição informada")
    uses_mounjaro = str(form.get("uses_mounjaro") or "").lower() in ("true", "1", "sim")
    mounjaro_dose = str(form.get("mounjaro_dose") or "").strip() or None

    with Session(engine) as session:
        stmt = (
            select(BodyEvaluation)
            .where(BodyEvaluation.user_key == user_key)
            .order_by(BodyEvaluation.created_at.desc())
            .limit(5)
        )
        past_records = list(reversed(session.exec(stmt).all()))

    db_history_text = _format_history_lines(past_records)
    client_history_text = str(form.get("previous_evaluations") or "").strip()
    combined_history = db_history_text or client_history_text

    def _run_analysis_with_memory() -> dict:
        prompt = vision_service._build_body_prompt(
            angles_count=len(pil_images),
            weight_kg=weight_kg,
            height_cm=height_cm,
            goal=goal,
            level=level,
            days_per_week=days_per_week,
            restrictions=restrictions,
            uses_mounjaro=uses_mounjaro,
            mounjaro_dose=mounjaro_dose,
        )
        prompt += (
            "\n\nCAMPOS ADICIONAIS OBRIGATÓRIOS NO JSON RAIZ:\n"
            '- "v_taper_analysis": "Análise técnica da proporção ombro x cintura e densidade muscular."\n'
            '- "training_strategy": "Resumo estratégico da periodização de treino gerada."\n'
            '- "nutrition_strategy": "Resumo estratégico do plano alimentar e distribuição de macros."\n'
        )
        if combined_history:
            prompt += (
                "\nHISTÓRICO REAL DE AVALIAÇÕES ANTERIORES DESTE ALUNO:\n"
                f"{combined_history}\n"
                '- "comparative_analysis": "Comparação detalhada entre as avaliações passadas e a atual: o que melhorou (peso, BF%, pontos fracos) e o que ainda precisa evoluir."\n'
            )
        else:
            prompt += (
                '- "comparative_analysis": "Primeira avaliação oficial salva no histórico do aluno. Servirá de linha de base para medir evolução de peso, BF% e simetria nos próximos 30 dias."\n'
            )

        raw_json = vision_service._execute_multimodal_with_fresh_clients(
            images=pil_images,
            prompt=prompt,
        )
        saved_rec = _extract_and_save_evaluation(
            user_key=user_key,
            user_name=user_name,
            weight_kg=weight_kg,
            height_cm=height_cm,
            goal=goal,
            level=level,
            days_per_week=days_per_week,
            raw_json_text=raw_json,
        )
        return {
            "analysis_text": raw_json,
            "analysis_markdown": raw_json,
            "evaluation_id": saved_rec.id,
            "history_count": len(past_records) + 1,
        }

    try:
        return await asyncio.to_thread(_run_analysis_with_memory)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/generate-workout")
async def generate_workout(request: Request):
    form = await request.form()
    goal = str(form.get("goal") or "Hipertrofia & Definição")
    level = str(form.get("level") or "Intermediário")
    restrictions = str(form.get("restrictions") or "Nenhuma restrição")

    raw_days = form.get("days_per_week")
    try:
        days_per_week = int(raw_days) if raw_days not in (None, "") else 4
    except (ValueError, TypeError):
        days_per_week = 4

    prompt = f"""
Você é um Head Coach de Treinamento de Força e Biomecânica.
Gere uma PLANILHA DE TREINO PERSONALIZADA ({days_per_week} divisões: A, B, C...) para:
- Objetivo: {goal}
- Nível: {level}
- Frequência: {days_per_week}x/semana
- Restrições / Lesões / Dores a evitar: {restrictions}

REGRAS OBRIGATÓRIAS:
1. Adapte TODOS os exercícios para respeitar 100% as restrições ({restrictions}).
2. Cada exercício deve conter: name, muscle_group, sets, reps, load_kg, rir (ex: "RIR 1-2"), cadence (ex: "3010"), rest_sec (inteiro) e focus (instrução técnica detalhada de ângulo, postura e execução).
3. Responda EXCLUSIVAMENTE em JSON válido (sem blocos ```json):

{{
  "strategy_summary": "Explicação técnica de como esta planilha foi montada respeitando nível ({level}), objetivo ({goal}) e restrições ({restrictions}).",
  "workouts": [
    {{
      "day_code": "A",
      "title": "Nome da Divisão A",
      "exercises": [
        {{
          "name": "Nome do Exercício",
          "muscle_group": "GRUPAMENTO ALVO",
          "sets": "4 Séries",
          "reps": "8-10 Reps",
          "load_kg": "Carga progressiva",
          "rir": "RIR 1-2",
          "cadence": "3010",
          "rest_sec": 90,
          "focus": "Instrução técnica detalhada de ângulo, postura e controle."
        }}
      ]
    }}
  ]
}}
"""

    try:
        workout_text = await asyncio.to_thread(
            vision_service._execute_multimodal_with_fresh_clients,
            images=[],
            prompt=prompt,
        )
        return {"workout_json": workout_text}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/analyze-meal")
@app.post("/nutrition/analyze-meal")
@app.post("/api/v1/nutrition/analyze-meal")
async def analyze_meal(request: Request):
    form = await request.form()
    image_bytes = None

    for key, value in form.multi_items():
        if hasattr(value, "read"):
            data = await value.read()
            if data:
                image_bytes = data
                break

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Nenhuma imagem de refeição enviada.")

    goal = str(form.get("goal") or "Hipertrofia e Definição")
    description = form.get("description")

    try:
        text = await asyncio.to_thread(
            vision_service.analyze_meal_image,
            image_bytes=image_bytes,
            meal_description=str(description) if description else goal,
        )
        return {"raw_analysis": text, "analysis_markdown": text}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc