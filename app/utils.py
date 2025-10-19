import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from passlib.context import CryptContext
import jwt
from fastapi import HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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

security = HTTPBearer()

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str):
    return pwd_context.verify(plain, hashed)

def generate_verify_code():
    return str(random.randint(100000, 999999))

def send_email(to_email: str, code: str):
    # Create a multipart message
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Your Verification Code"
    msg["From"] = SMTP_USER
    msg["To"] = to_email

    # Plain text version for fallback
    text = f"Your verification code is: {code}\n\nThis code is valid for 10 minutes.\nIf you did not request this code, please ignore this email."

    # HTML version for professional look
    html = f"""
    <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="text-align: center; padding: 20px; background-color: #f8f8f8; border-radius: 8px;">
                <h2 style="color: #2c3e50;">Verification Code</h2>
                <p style="font-size: 16px;">Hello,</p>
                <p style="font-size: 16px;">Thank you for using our service. Please use the following code to verify your account:</p>
                <div style="background-color: #3498db; color: white; font-size: 24px; font-weight: bold; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    {code}
                </div>
                <p style="font-size: 14px;">This code is valid for <strong>10 minutes</strong>.</p>
                <p style="font-size: 14px;">If you did not request this code, please ignore this email or contact our support team.</p>
                <hr style="border: 1px solid #eee; margin: 20px 0;">
                <p style="font-size: 12px; color: #777;">
                    &copy; {datetime.now().year} Your Company Name. All rights reserved.<br>
                    For support, contact us at <a href="mailto:support@yourcompany.com" style="color: #3498db;">support@yourcompany.com</a>
                </p>
            </div>
        </body>
    </html>
    """

    # Attach both text and HTML versions
    part1 = MIMEText(text, "plain")
    part2 = MIMEText(html, "html")
    msg.attach(part1)
    msg.attach(part2)

    # Send the email
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, to_email, msg.as_string())

def create_jwt(user_id: int, expires_delta: Optional[timedelta] = None) -> str:
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=1)
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

def verify_jwt(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Xác thực JWT token từ header Authorization: Bearer <token>.
    Trả về user_id nếu token hợp lệ.
    """
    try:
        token = credentials.credentials
        payload = decode_jwt(token)
        user_id = int(payload.get("sub"))
        if user_id is None:
            logger.error("No user_id in token payload")
            raise HTTPException(status_code=401, detail="Invalid token: no user_id")
        return user_id
    except Exception as e:
        logger.error(f"JWT verification failed: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
