import random
import smtplib
from email.mime.text import MIMEText
from passlib.context import CryptContext
import jwt
from fastapi import HTTPException
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import logging
from app.db import get_db

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()
pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("SECRET_KEY", "secret")
ALGORITHM = "HS256"
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")


def hash_password(password: str):
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str):
    return pwd_context.verify(plain, hashed)


def generate_verify_code():
    return str(random.randint(100000, 999999))


def send_email(to_email: str, code: str):
    msg = MIMEText(f"Your verification code: {code}")
    msg["Subject"] = "Verify Code"
    msg["From"] = SMTP_USER
    msg["To"] = to_email
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, to_email, msg.as_string())


from typing import Optional


def create_jwt(user_id: int, expires_delta: Optional[timedelta] = None) -> str:
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            hours=1
        )  # Giữ mặc định 1 giờ nếu không truyền expires_delta
    payload = {"sub": str(user_id), "exp": expire}
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    logger.debug(f"Created JWT for user_id {user_id}: {token}")
    return token


def decode_jwt(token: str):
    try:
        logger.debug(f"Decoding token: {token}")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        logger.debug(f"Decoded payload: {payload}")
        return payload
    except jwt.ExpiredSignatureError as e:
        logger.error(f"Token has expired: {str(e)}")
        raise HTTPException(401, f"Token has expired: {str(e)}")
    except jwt.InvalidTokenError as e:
        logger.error(f"Invalid token: {str(e)}")
        raise HTTPException(401, f"Invalid token format or signature: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error decoding token: {str(e)}")
        raise HTTPException(401, f"Unexpected error: {str(e)}")
