from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel


class MarketplaceSession(SQLModel, table=True):
    __tablename__ = "marketplace_sessions"

    id: Optional[int] = Field(default=None, primary_key=True)
    coach_id: int = Field(foreign_key="users.id")
    student_id: Optional[int] = Field(default=None, foreign_key="users.id")
    title: str
    description: Optional[str] = None
    price: float = Field(default=0.0)
    scheduled_at: Optional[datetime] = None
    status: str = Field(default="open")  # open | booked | completed | cancelled