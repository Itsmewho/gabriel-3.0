import os
from dotenv import load_dotenv
from typing import Optional
from utils.helpers import get_env_int, get_env_str


load_dotenv()


# --- Redis
REDIS_HOST: str = get_env_str("REDIS_HOST")
REDIS_PORT: int = get_env_int("REDIS_PORT")
REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
REDIS_DB: int = get_env_int("REDIS_DB", default=0)

# --- SQL
CONNECTION_STRING: str = get_env_str("DB_CONNECTION")
DATABASE_URL: str = get_env_str("DATABASE_URL")

# --- Email
SMTP_HOST: str = get_env_str("SMTP_HOST")
SMTP_PORT: str = get_env_str("SMTP_PORT")
SMTP_USER: str = get_env_str("SMTP_USER")
SMTP_PASS: str = get_env_str("SMTP_PASS")


# --- MT5
ACCOUNT: str = os.getenv("ACCOUNT_NUM", "")
ACCOUNT_PASS: str = os.getenv("ACCOUNT_PASS", "")
META_SERVER: str = os.getenv("META_SERVER", "")


# --- URL
BASE_URL: str = os.getenv("BASE_URL", "http://localhost:5500")

# --- HTTP
HTTPONLY: bool = os.getenv("HTTPONLY", "False").lower() == "true"
SESSION_EXPIRE_SECONDS = int(os.getenv("SESSION_EXPIRE_SECONDS", "60")) * 60


# --- API CALL TIMES

SLEEPTIME: int = 60000
MODELSLEEPTIME: int = 60000  # 1min?
