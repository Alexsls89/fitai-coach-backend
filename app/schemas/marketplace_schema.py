from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class SessionCreate(BaseModel):
    coach_id: int
    title: str
    description: Optional[str] = None
    price: float
    scheduled_at: Optional[datetime] = None


class SessionRead(BaseModel):
    id: int
    coach_id: int
    student_id: Optional[int] = None
    title: str
    description: Optional[str] = None
    price: float
    scheduled_at: Optional[datetime] = None
    status: str