import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from passlib.context import CryptContext
import jwt
from fastapi import HTTPException, Depends, Request, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import logging
import secrets

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


def create_reset_token(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    logger.debug(f"Created reset token for user_id {user_id}: {token}")
    return token


def send_email(to_email: str, code: str, template_type: str = "verification"):
    # Create a multipart message
    msg = MIMEMultipart("alternative")
    msg["Subject"] = (
        "Your Verification Code"
        if template_type == "verification"
        else "Reset Your Password"
    )
    msg["From"] = SMTP_USER
    msg["To"] = to_email

    if template_type == "verification":
        # Plain text version for verification
        text = f"Your verification code is: {code}\n\nThis code is valid for 10 minutes.\nIf you did not request this code, please ignore this email."

        # HTML version for verification
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
                        &copy; {datetime.now().year} FourTAI. All rights reserved.<br>
                        For support, contact us at <a href="mailto:vtrung836@gmail.com.com" style="color: #3498db;">vtrung836@gmail.com.com</a>
                    </p>
                </div>
            </body>
        </html>
        """
    else:  # reset_password
        # Plain text version for reset password
        text = f"Click the following link to reset your password:\nhttps://living-tortoise-polite.ngrok-free.app/reset-password?token={code}\n\nThis link is valid for 1 hour.\nIf you did not request a password reset, please ignore this email."

        # HTML version for reset password
        html = f"""
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="text-align: center; padding: 20px; background-color: #f8f8f8; border-radius: 8px;">
                    <h2 style="color: #2c3e50;">Reset Your Password</h2>
                    <p style="font-size: 16px;">Hello,</p>
                    <p style="font-size: 16px;">We received a request to reset your password. Please click the button below to set a new password:</p>
                    <a href="https://living-tortoise-polite.ngrok-free.app/reset-password?token={code}" style="display: inline-block; background-color: #e74c3c; color: white; font-size: 16px; font-weight: bold; padding: 15px 30px; border-radius: 5px; margin: 20px 0; text-decoration: none;">
                        Reset Password
                    </a>
                    <p style="font-size: 14px;">This link is valid for <strong>1 hour</strong>.</p>
                    <p style="font-size: 14px;">If you did not request a password reset, please ignore this email or contact our support team.</p>
                    <hr style="border: 1px solid #eee; margin: 20px 0;">
                    <p style="font-size: 12px; color: #777;">
                        &copy; {datetime.now().year} FourT AI. All rights reserved.<br>
                        For support, contact us at <a href="mailto:vtrung836@gmail.com.com" style="color: #3498db;">vtrung836@gmail.com.com</a>
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


async def verify_jwt(request: Request):
    """
    Xác thực JWT token từ header Authorization hoặc cookie.
    Trả về user_id nếu token hợp lệ.
    """
    token = None

    # Ưu tiên lấy từ header Authorization
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]

    # Nếu không có trong header, thử lấy từ cookie
    if not token:
        cookie_token = request.cookies.get("access_token")
        if cookie_token and cookie_token.startswith("Bearer "):
            token = cookie_token[7:]  # Bỏ "Bearer "

    if not token:
        logger.error("No token provided in header or cookie")
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        payload = decode_jwt(token)
        user_id = int(payload.get("sub"))
        if user_id is None:
            logger.error("No user_id in token payload")
            raise HTTPException(status_code=401, detail="Invalid token: no user_id")
        return user_id
    except Exception as e:
        logger.error(f"JWT verification failed: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
