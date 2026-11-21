from sqlmodel import SQLModel, Session, create_engine
from app.core.config import settings

connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(settings.DATABASE_URL, echo=True, connect_args=connect_args)


def create_db_and_tables():
    from app import models  # Regista todas as tabelas antes de criar
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session