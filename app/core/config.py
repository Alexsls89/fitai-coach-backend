import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)


class Settings:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "FitAI & Coach API")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./fitai.db")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key")


settings = Settings()