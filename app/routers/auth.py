from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.db import get_db
from app.models import User
from app.schemas import UserRegister, UserLogin, VerifyCode, ResetPassword
from app.utils import (
    hash_password,
    verify_password,
    generate_verify_code,
    send_email,
    create_jwt,
    verify_jwt,
    create_reset_token
)
from typing import Dict, List
from datetime import timedelta, datetime
from pydantic import BaseModel
import logging
import secrets

logger = logging.getLogger(__name__)

router = APIRouter()
verify_codes: Dict[str, str] = {}  # Lưu mã xác minh email
auth_codes: Dict[str, dict] = {}  # Lưu authorization code: {code: {device_id, user_id, expires}}
reset_tokens: Dict[str, dict] = {}  # Lưu reset token: {token: {user_id, expires}}

# Schema cho forgetpw
class ForgetPasswordRequest(BaseModel):
    email: str

@router.post("/register")
def register(user: UserRegister, db: Session = Depends(get_db)):
    existing = (
        db.query(User)
        .filter((User.username == user.username) | (User.email == user.email))
        .first()
    )
    if existing:
        raise HTTPException(400, "User exists")
    hashed = hash_password(user.password)
    new_user = User(
        username=user.username,
        email=user.email,
        password_hash=hashed,
        gender=user.gender  # Lưu gender từ UserRegister
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    code = generate_verify_code()
    verify_codes[user.email] = code
    send_email(user.email, code, template_type="verification")
    logger.debug(f"Registered user: {user.username}, email: {user.email}, code: {code}, gender: {user.gender}")
    return {"message": "Registered, check email for code", "user_id": new_user.id}

@router.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter((User.username == user.username_or_email) | (User.email == user.username_or_email)).first()
    if not db_user:
        logger.error(f"User not found: {user.username_or_email}")
        raise HTTPException(400, "User not found")
    db.refresh(db_user)
    try:
        if not verify_password(user.password, db_user.password_hash):
            logger.error(f"Invalid password for user: {user.username_or_email}")
            raise HTTPException(400, "Invalid password")
    except Exception as e:
        logger.error(f"Password verification failed for user {user.username_or_email}: {str(e)}")
        raise HTTPException(400, f"Password verification failed: {str(e)}")

    input_device_id = user.device_id.strip().lower()
    if db_user.verified_devices is None:
        verified_devices: List[str] = []
    else:
        verified_devices = db_user.verified_devices

    verified_set = set(d.strip().lower() for d in verified_devices)
    logger.debug(f"Verified devices for user_id {db_user.id} (Set): {verified_set}")
    logger.debug(f"Device_id sent (Normalized): {input_device_id}")

    if input_device_id in verified_set:
        token = create_jwt(db_user.id, expires_delta=timedelta(days=7))
        logger.debug(f"Login successful for user_id {db_user.id}, token issued: {token}")
        return {"token": token}

    code = generate_verify_code()
    verify_codes[db_user.email] = code
    send_email(db_user.email, code, template_type="verification")
    logger.debug(f"Verification code sent to {db_user.email} for user_id {db_user.id}: {code}")
    return {"message": "Verify needed", "user_id": db_user.id}

@router.post("/verify")
def verify(verify: VerifyCode, user_id: int = Query(...), db: Session = Depends(get_db)):
    logger.debug(f"Verify request: user_id={user_id}, code={verify.code}, device_id: {verify.device_id}")
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        logger.error(f"User not found for user_id: {user_id}")
        raise HTTPException(status_code=404, detail="User not found")

    db.refresh(db_user)
    if db_user.email not in verify_codes:
        logger.error(f"No verification code found for email: {db_user.email}")
        raise HTTPException(status_code=400, detail="No verification code found for this user")
    if verify_codes[db_user.email] != verify.code:
        logger.error(f"Invalid verification code for user_id {user_id}, email: {db_user.email}, provided: {verify.code}")
        raise HTTPException(status_code=400, detail="Invalid verification code")

    device_id_to_add = verify.device_id.strip().lower()
    if db_user.verified_devices is None:
        db_user.verified_devices = []

    verified_set = set(d.strip().lower() for d in db_user.verified_devices)
    if device_id_to_add not in verified_set:
        db_user.verified_devices.append(device_id_to_add)
        db.commit()
        logger.debug(f"Added device_id {device_id_to_add} to verified_devices for user_id {user_id}")
    else:
        logger.debug(f"Device_id {device_id_to_add} already verified for user_id {user_id}")

    if db_user.email in verify_codes:
        del verify_codes[db_user.email]
        logger.debug(f"Removed verification code for email: {db_user.email}")

    for auth_code, data in auth_codes.items():
        if data["device_id"] == device_id_to_add and data["user_id"] is None:
            data["user_id"] = db_user.id
            logger.debug(f"Updated auth_code {auth_code} with user_id {db_user.id}")

    token = create_jwt(db_user.id, expires_delta=timedelta(days=7))
    logger.debug(f"Verification successful for user_id {user_id}, device_id: {device_id_to_add}, token issued: {token}")
    return {"token": token}

@router.get("/get-token")
def get_token(user_id: int = Query(...), device_id: str = Query(...), db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        logger.error(f"User not found for user_id: {user_id}")
        raise HTTPException(status_code=404, detail="User not found")

    if db_user.verified_devices is None:
        verified_devices: List[str] = []
    else:
        verified_devices = db_user.verified_devices

    verified_set = set(d.strip().lower() for d in verified_devices)
    normalized_device_id = device_id.strip().lower()

    if normalized_device_id not in verified_set:
        logger.error(f"Device not verified for user_id: {user_id}, device_id: {device_id}")
        raise HTTPException(status_code=400, detail="Device not verified")

    token = create_jwt(db_user.id, expires_delta=timedelta(days=7))
    logger.debug(f"Token retrieved for user_id: {user_id}, device_id: {normalized_device_id}, token: {token}")
    return {"token": token}

@router.get("/validate-token")
def validate_token(user_id: int = Depends(verify_jwt)):
    return {"message": "Token is valid", "user_id": user_id}

@router.get("/authorize")
def authorize(device_id: str = Query(...), redirect_uri: str = Query(...), state: str = Query(...), db: Session = Depends(get_db)):
    """
    Endpoint để tạo URL đăng nhập cho OAuth flow.
    """
    auth_code = secrets.token_urlsafe(32)
    auth_codes[auth_code] = {
        "device_id": device_id,
        "user_id": None,  # Chưa xác thực
        "expires": datetime.utcnow() + timedelta(minutes=10),  # Code hết hạn sau 10 phút
        "state": state
    }
    login_url = f"https://living-tortoise-polite.ngrok-free.app/?code={auth_code}&state={state}&redirect_uri={redirect_uri}"
    logger.debug(f"Generated login_url: {login_url}")
    return {"login_url": login_url}

@router.post("/token")
def exchange_token(code: str = Query(...), state: str = Query(...), db: Session = Depends(get_db)):
    """
    Đổi authorization code lấy JWT token.
    """
    logger.debug(f"Exchange token request: code={code}, state={state}")
    auth_data = auth_codes.get(code)
    if not auth_data or auth_data["state"] != state or auth_data["expires"] < datetime.utcnow():
        logger.error(f"Invalid or expired auth code: {code}")
        raise HTTPException(status_code=400, detail="Invalid or expired code")

    user_id = auth_data.get("user_id")
    if not user_id:
        logger.error(f"No user associated with code: {code}")
        raise HTTPException(status_code=400, detail="User not authenticated")

    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        logger.error(f"User not found for user_id: {user_id}")
        raise HTTPException(status_code=404, detail="User not found")

    device_id = auth_data["device_id"].strip().lower()
    if db_user.verified_devices is None:
        verified_devices: List[str] = []
    else:
        verified_devices = db_user.verified_devices

    verified_set = set(d.strip().lower() for d in verified_devices)
    if device_id not in verified_set:
        logger.error(f"Device not verified for user_id: {user_id}, device_id: {device_id}")
        raise HTTPException(status_code=400, detail="Device not verified")

    token = create_jwt(user_id, expires_delta=timedelta(days=7))
    del auth_codes[code]  # Xóa code sau khi sử dụng
    logger.debug(f"Token issued for user_id: {user_id}, device_id: {device_id}, token: {token}")
    return {"token": token}

@router.post("/forgetpw")
def forget_password(request: ForgetPasswordRequest, db: Session = Depends(get_db)):
    """
    Endpoint để yêu cầu đặt lại mật khẩu và gửi email chứa link reset.
    """
    logger.debug(f"Forget password request for email: {request.email}")
    db_user = db.query(User).filter(User.email == request.email).first()
    if not db_user:
        logger.error(f"User not found for email: {request.email}")
        raise HTTPException(status_code=404, detail="User not found")

    reset_token = create_reset_token(db_user.id)
    reset_tokens[reset_token] = {
        "user_id": db_user.id,
        "expires": datetime.utcnow() + timedelta(hours=1)
    }
    send_email(db_user.email, reset_token, template_type="reset_password")
    logger.debug(f"Reset password link sent to {db_user.email} for user_id {db_user.id}")
    return {"message": "Reset password link sent to your email"}

@router.post("/reset-password")
def reset_password(reset: ResetPassword, db: Session = Depends(get_db)):
    """
    Endpoint để đặt lại mật khẩu bằng reset token.
    """
    logger.debug(f"Reset password request for token: {reset.reset_token}")
    token_data = reset_tokens.get(reset.reset_token)
    if not token_data or token_data["expires"] < datetime.utcnow():
        logger.error(f"Invalid or expired reset token: {reset.reset_token}")
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    user_id = token_data["user_id"]
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        logger.error(f"User not found for user_id: {user_id}")
        raise HTTPException(status_code=404, detail="User not found")

    hashed_password = hash_password(reset.new_password)
    db_user.password_hash = hashed_password
    db.commit()
    db.refresh(db_user)

    del reset_tokens[reset.reset_token]
    logger.debug(f"Password reset successful for user_id: {user_id}, token removed")
    return {"message": "Password reset successfully"}
