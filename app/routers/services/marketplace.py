from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from app.core.database import get_session
from app.models.marketplace import MarketplaceSession
from app.models.user import User
from app.schemas.marketplace_schema import SessionCreate, SessionRead

router = APIRouter(prefix="/marketplace", tags=["Módulo D - Marketplace"])


@router.post("/users", response_model=User)
def create_user(name: str, email: str, role: str = "student", session: Session = Depends(get_session)):
    user = User(name=name, email=email, role=role)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


@router.post("/sessions", response_model=SessionRead)
def create_session(payload: SessionCreate, session: Session = Depends(get_session)):
    new_session = MarketplaceSession(**payload.model_dump())
    session.add(new_session)
    session.commit()
    session.refresh(new_session)
    return new_session


@router.get("/sessions", response_model=List[SessionRead])
def list_sessions(session: Session = Depends(get_session)):
    return session.exec(select(MarketplaceSession)).all()


@router.post("/sessions/{session_id}/book", response_model=SessionRead)
def book_session(session_id: int, student_id: int, session: Session = Depends(get_session)):
    db_session = session.get(MarketplaceSession, session_id)
    if not db_session:
        raise HTTPException(status_code=404, detail="Aula não encontrada")
    if db_session.status != "open":
        raise HTTPException(status_code=400, detail="Esta aula já não está disponível")

    db_session.student_id = student_id
    db_session.status = "booked"
    session.add(db_session)
    session.commit()
    session.refresh(db_session)
    return db_session