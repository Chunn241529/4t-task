from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.db import get_db
from app.models import User
from app.schemas import UserRegister, UserLogin, VerifyCode
from app.utils import (
    hash_password,
    verify_password,
    generate_verify_code,
    send_email,
    create_jwt,
)
from typing import Dict, List
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
verify_codes: Dict[str, str] = {}  # Temp storage, thay bằng DB sau


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
    new_user = User(username=user.username, email=user.email, password_hash=hashed)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    code = generate_verify_code()
    verify_codes[user.email] = code
    send_email(user.email, code)
    logger.debug(f"Registered user: {user.username}, email: {user.email}, code: {code}")
    return {"message": "Registered, check email for code"}


@router.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter((User.username == user.username_or_email) | (User.email == user.username_or_email)).first()
    if not db_user:
        logger.error(f"User not found: {user.username_or_email}")
        raise HTTPException(400, "User not found")
    db.refresh(db_user)  # Đảm bảo load data mới nhất từ DB
    try:
        if not verify_password(user.password, db_user.password_hash):
            logger.error(f"Invalid password for user: {user.username_or_email}")
            raise HTTPException(400, "Invalid password")
    except Exception as e:
        logger.error(f"Password verification failed for user {user.username_or_email}: {str(e)}")
        raise HTTPException(400, f"Password verification failed: {str(e)}")

    # === LOGIC MỚI CHO DEVICE ID ===
    # 1. Chuẩn hóa device_id đầu vào
    input_device_id = user.device_id.strip().lower()

    # 2. Xử lý verified_devices nếu None
    if db_user.verified_devices is None:
        verified_devices: List[str] = []
    else:
        verified_devices = db_user.verified_devices

    # 3. Chuẩn hóa danh sách đã xác thực sang set để tìm kiếm hiệu quả
    verified_set = set(d.strip().lower() for d in verified_devices)

    logger.debug(f"Verified devices for user_id {db_user.id} (Set): {verified_set}")
    logger.debug(f"Device_id sent (Normalized): {input_device_id}")

    if input_device_id in verified_set:
        # THÀNH CÔNG: Thiết bị đã được xác thực
        token = create_jwt(db_user.id, expires_delta=timedelta(days=7))  # Token sống 7 ngày
        logger.debug(f"Login successful for user_id {db_user.id}, token issued: {token}")
        return {"token": token}

    # GỬI MÃ: Thiết bị chưa được xác thực
    code = generate_verify_code()
    verify_codes[db_user.email] = code
    send_email(db_user.email, code)
    logger.debug(f"Verification code sent to {db_user.email} for user_id {db_user.id}: {code}")
    return {"message": "Verify needed", "user_id": db_user.id}


@router.post("/verify")
def verify(verify: VerifyCode, user_id: int = Query(...), db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        logger.error(f"User not found for user_id: {user_id}")
        raise HTTPException(400, "User not found")
    db.refresh(db_user)  # Đảm bảo load data mới nhất
    if verify_codes.get(db_user.email) != verify.code:
        logger.error(f"Invalid verification code for user_id {user_id}, email: {db_user.email}")
        raise HTTPException(400, "Invalid code")

    # === LOGIC MỚI CHO DEVICE ID ===
    # 1. Chuẩn hóa device_id đầu vào để lưu
    device_id_to_add = verify.device_id.strip().lower()

    # 2. Xử lý verified_devices nếu None
    if db_user.verified_devices is None:
        db_user.verified_devices = []

    # 3. Chuẩn hóa danh sách đã xác thực (tạo set để kiểm tra)
    verified_set = set(d.strip().lower() for d in db_user.verified_devices)

    if device_id_to_add not in verified_set:
        # Thêm device_id đã chuẩn hóa vào danh sách
        db_user.verified_devices.append(device_id_to_add)
        db.commit()
        logger.debug(f"Added device_id {device_id_to_add} to verified_devices for user_id {user_id}")
    else:
        logger.debug(f"Device_id {device_id_to_add} already verified for user_id {user_id}")

    if db_user.email in verify_codes:
        del verify_codes[db_user.email]

    token = create_jwt(db_user.id, expires_delta=timedelta(days=7))  # Token sống 7 ngày
    logger.debug(f"Verification successful for user_id {user_id}, device_id: {device_id_to_add}, token issued: {token}")
    return {"token": token}
