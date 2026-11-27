import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import Session, select

router = APIRouter(tags=["Coach Sync"])


def _parse_float(raw: Any, default: float = 0.0) -> float:
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    m = re.search(r"-?\d+(?:[.,]\d+)?", str(raw))
    if not m:
        return default
    try:
        return float(m.group(0).replace(",", "."))
    except ValueError:
        return default


def _parse_int(raw: Any, default: int = 0) -> int:
    return int(round(_parse_float(raw, float(default))))


def _extract_protocol_parts(proto: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(proto, dict):
        return {
            "workouts": [],
            "meals": [],
            "training_strategy": "",
            "nutrition_strategy": "",
            "comparative_summary": "",
            "coach_observation": "",
        }

    workout_node = proto.get("workout") if isinstance(proto.get("workout"), dict) else {}
    nutrition_node = proto.get("nutrition") if isinstance(proto.get("nutrition"), dict) else {}

    raw_days = (
        workout_node.get("days")
        or workout_node.get("workouts")
        or proto.get("workouts")
        or proto.get("workout_days")
        or []
    )
    workouts_out: List[Dict[str, Any]] = []
    if isinstance(raw_days, list):
        for idx, d in enumerate(raw_days):
            if not isinstance(d, dict):
                continue
            code = str(d.get("code") or d.get("day_code") or chr(65 + (idx % 26)))
            title = str(d.get("title") or d.get("focus") or f"Treino {code}")
            raw_exs = d.get("exercises") or d.get("items") or []
            exs_out = []
            if isinstance(raw_exs, list):
                for ex in raw_exs:
                    if not isinstance(ex, dict):
                        continue
                    exs_out.append({
                        "name": str(ex.get("name") or ex.get("exercise") or "Exercício"),
                        "muscle_group": str(ex.get("muscle_group") or ex.get("muscleGroup") or "Geral"),
                        "sets": str(ex.get("sets") or "3"),
                        "reps": str(ex.get("reps") or "10-12"),
                        "load_kg": str(ex.get("load_kg") or ex.get("loadKg") or ex.get("load") or "Moderada"),
                        "rir": str(ex.get("rir") or "2"),
                        "cadence": str(ex.get("cadence") or "2020"),
                        "rest_sec": _parse_int(ex.get("rest_sec") or ex.get("restSec") or 60, 60),
                        "notes": str(ex.get("notes") or ""),
                        "completed": bool(ex.get("completed", False)),
                    })
            workouts_out.append({"code": code, "title": title, "exercises": exs_out})

    raw_meals = (
        nutrition_node.get("meals")
        or proto.get("meals")
        or []
    )
    meals_out: List[Dict[str, Any]] = []
    if isinstance(raw_meals, list):
        for idx, m in enumerate(raw_meals):
            if not isinstance(m, dict):
                continue
            foods_raw = m.get("foods") or m.get("items") or []
            foods = [str(f).strip() for f in foods_raw if str(f).strip()] if isinstance(foods_raw, list) else []
            meals_out.append({
                "time": str(m.get("time") or f"{8 + idx * 3:02d}:00"),
                "title": str(m.get("title") or m.get("name") or f"Refeição {idx + 1}"),
                "kcal": _parse_int(m.get("kcal") or m.get("calories") or 0),
                "protein_g": _parse_int(m.get("protein_g") or m.get("protein") or 0),
                "carbs_g": _parse_int(m.get("carbs_g") or m.get("carbs") or 0),
                "fat_g": _parse_int(m.get("fat_g") or m.get("fat") or 0),
                "foods": foods,
            })

    return {
        "workouts": workouts_out,
        "meals": meals_out,
        "training_strategy": str(workout_node.get("strategy") or proto.get("training_strategy") or ""),
        "nutrition_strategy": str(nutrition_node.get("strategy") or proto.get("nutrition_strategy") or ""),
        "comparative_summary": str(proto.get("comparative_analysis") or proto.get("comparative_summary") or ""),
        "coach_observation": str(proto.get("coach_observation") or ""),
    }


def build_enriched_coach_students(
    session: Session,
    UserAccount: Any,
    BodyEvaluation: Any,
    DailyLog: Any,
    clean_code: str,
) -> List[Dict[str, Any]]:
    stmt = select(UserAccount).where(UserAccount.role == "student")
    if clean_code:
        stmt = stmt.where(func.upper(UserAccount.linked_coach_code) == clean_code)

    linked_users = list(session.exec(stmt).all())
    students_out: List[Dict[str, Any]] = []
    now_utc = datetime.now(timezone.utc)

    for st_user in linked_users:
        email_norm = (st_user.email or "").strip().lower()
        evals = list(
            session.exec(
                select(BodyEvaluation)
                .where(func.lower(BodyEvaluation.user_key) == email_norm)
                .order_by(BodyEvaluation.created_at.desc())
            ).all()
        )
        logs = list(
            session.exec(
                select(DailyLog)
                .where(func.lower(DailyLog.user_email) == email_norm)
                .order_by(DailyLog.log_date.desc())
            ).all()
        )

        latest = evals[0] if evals else None
        oldest = evals[-1] if evals else None

        proto_dict: Dict[str, Any] = {}
        if latest and getattr(latest, "protocol_json", None):
            try:
                parsed = json.loads(latest.protocol_json)
                if isinstance(parsed, dict):
                    proto_dict = parsed
            except Exception:
                proto_dict = {}

        extracted = _extract_protocol_parts(proto_dict)
        for ev in evals:
            try:
                ep = json.loads(getattr(ev, "protocol_json", None) or "{}")
            except Exception:
                ep = {}
            if isinstance(ep, dict):
                ex = _extract_protocol_parts(ep)
                extracted["workouts"] = extracted["workouts"] or ex["workouts"]
                extracted["meals"] = extracted["meals"] or ex["meals"]
                extracted["comparative_summary"] = extracted["comparative_summary"] or ex["comparative_summary"]
                extracted["training_strategy"] = extracted["training_strategy"] or ex["training_strategy"]
                extracted["nutrition_strategy"] = extracted["nutrition_strategy"] or ex["nutrition_strategy"]

        score_val = next((int(ev.shape_score) for ev in evals if getattr(ev, "shape_score", 0)), 0)

        # Consistência e dias inativos reais
        cutoff_14d = (now_utc - timedelta(days=14)).strftime("%Y-%m-%d")
        active_recent_days = len({lg.log_date for lg in logs if (lg.log_date or "") >= cutoff_14d})
        if active_recent_days > 0:
            consistency_pct = min(100, int(round((active_recent_days / 14.0) * 100)))
        elif evals:
            consistency_pct = min(100, max(35, len(evals) * 25))
        else:
            consistency_pct = 0

        last_activity_dt = latest.created_at if latest else st_user.created_at
        if logs and logs[0].updated_at:
            if logs[0].updated_at.tzinfo is None:
                last_activity_dt = logs[0].updated_at.replace(tzinfo=timezone.utc)
            else:
                last_activity_dt = logs[0].updated_at
        elif last_activity_dt and last_activity_dt.tzinfo is None:
            last_activity_dt = last_activity_dt.replace(tzinfo=timezone.utc)

        days_inactive = max(0, (now_utc - last_activity_dt).days) if last_activity_dt else 0

        # Curva cronológica de evolução
        evolution_points = []
        history_log = []
        for ev in reversed(evals):
            dt_label = ev.created_at.strftime("%d/%m") if ev.created_at else "--/--"
            bf_num = _parse_float(ev.bf_percent, 0.0)
            evolution_points.append({
                "label": dt_label,
                "weight_kg": round(float(ev.weight_kg or 0.0), 1),
                "bf_percent": round(bf_num, 1),
            })
            history_log.append(
                f"{dt_label} • Avaliação 360°: {float(ev.weight_kg or 0.0):.1f} kg | BF {bf_num:.1f}% | Score {ev.shape_score}/100"
            )

        for lg in logs[:5]:
            history_log.append(
                f"{lg.log_date} • Diário: {lg.total_kcal} kcal | Prot {lg.total_protein_g}g | Treino {'OK' if lg.workout_completed else 'Pendente'}"
            )

        current_w = float(latest.weight_kg) if latest else 0.0
        initial_w = float(oldest.weight_kg) if oldest else current_w
        bf_val = _parse_float(latest.bf_percent, 0.0) if latest else 0.0
        score_val = next((int(ev.shape_score) for ev in evals if getattr(ev, "shape_score", 0)), 0)
        comp_summary = (
            (latest.comparative_summary if latest and latest.comparative_summary else "")
            or extracted["comparative_summary"]
        )

        students_out.append({
            # Chaves normalizadas para o Flutter AppStore
            "id": email_norm,
            "email": email_norm,
            "name": st_user.full_name or "Aluno",
            "coach_code": st_user.linked_coach_code or clean_code,
            "goal": (latest.goal if latest and latest.goal else "Hipertrofia & Definição"),
            "level": (latest.level if latest and latest.level else "Intermediário"),
            "days_per_week": int(latest.days_per_week) if latest else 4,
            "weight_kg": round(current_w, 1),
            "initial_weight_kg": round(initial_w, 1),
            "bf_percent": round(bf_val, 1),
            "shape_score": score_val,
            "consistency_percent": consistency_pct,
            "days_inactive": days_inactive,
            "comparative_summary": comp_summary,
            "training_strategy": extracted["training_strategy"],
            "nutrition_strategy": extracted["nutrition_strategy"],
            "coach_observation": extracted["coach_observation"],
            "workouts": extracted["workouts"],
            "meals": extracted["meals"],
            "evolution": evolution_points,
            "history_log": history_log,
            # Chaves legadas preservadas
            "user_key": email_norm,
            "user_name": st_user.full_name or "Aluno",
            "latest_weight_kg": round(current_w, 1),
            "latest_bf_percent": f"{bf_val:.1f}%" if latest else "",
            "latest_shape_score": score_val,
            "last_evaluation_at": latest.created_at.isoformat() if latest else st_user.created_at.isoformat(),
            "evaluations_count": len(evals),
        })

    return students_out


class CoachStudentUpdatePayload(BaseModel):
    student_email: str
    student_name: Optional[str] = None
    weight_kg: Optional[float] = None
    bf_percent: Optional[float] = None
    coach_observation: Optional[str] = None
    workouts: Optional[List[Dict[str, Any]]] = None
    meals: Optional[List[Dict[str, Any]]] = None


@router.post("/coach/students/update")
def update_coach_student_protocol(payload: CoachStudentUpdatePayload):
    from app.main import BodyEvaluation, engine

    email_norm = payload.student_email.strip().lower()
    if not email_norm:
        raise HTTPException(status_code=400, detail="E-mail do aluno obrigatório.")

    with Session(engine) as session:
        latest = session.exec(
            select(BodyEvaluation)
            .where(func.lower(BodyEvaluation.user_key) == email_norm)
            .order_by(BodyEvaluation.created_at.desc())
        ).first()

        if latest is None:
            latest = BodyEvaluation(
                user_key=email_norm,
                user_name=payload.student_name or "Aluno",
                weight_kg=payload.weight_kg or 0.0,
                bf_percent=f"{payload.bf_percent:.1f}%" if payload.bf_percent else "",
                shape_score=0,
                goal="Hipertrofia & Definição",
                level="Intermediário",
                days_per_week=4,
                protocol_json="{}",
            )

        if payload.weight_kg is not None and payload.weight_kg > 0:
            latest.weight_kg = float(payload.weight_kg)
        if payload.bf_percent is not None and payload.bf_percent > 0:
            latest.bf_percent = f"{payload.bf_percent:.1f}%"

        proto: Dict[str, Any] = {}
        try:
            parsed = json.loads(latest.protocol_json or "{}")
            if isinstance(parsed, dict):
                proto = parsed
        except Exception:
            proto = {}

        if payload.coach_observation is not None:
            proto["coach_observation"] = payload.coach_observation.strip()

        if payload.workouts is not None:
            proto["workouts"] = payload.workouts
            if not isinstance(proto.get("workout"), dict):
                proto["workout"] = {}
            proto["workout"]["days"] = payload.workouts

        if payload.meals is not None:
            proto["meals"] = payload.meals
            if not isinstance(proto.get("nutrition"), dict):
                proto["nutrition"] = {}
            proto["nutrition"]["meals"] = payload.meals

        latest.protocol_json = json.dumps(proto, ensure_ascii=False)
        session.add(latest)
        session.commit()
        session.refresh(latest)
        return {"status": "ok", "student_email": email_norm}