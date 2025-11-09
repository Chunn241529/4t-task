from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy.orm import Session
from app.db import get_db
from app.services.device_detection import DeviceDetectionService
from app.services.device_service import DeviceService
from app.models import User
from app.schemas import UserRegister, UserLogin, VerifyCode, ResetPassword
from app.utils import (
    hash_password,
    verify_password,
    generate_verify_code,
    send_email,
    create_jwt,
    verify_jwt,
    create_reset_token,
)
from typing import Dict, List
from datetime import timedelta, datetime
from pydantic import BaseModel
import logging
import secrets

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])
verify_codes: Dict[str, str] = {}  # Lưu mã xác minh email
auth_codes: Dict[str, dict] = (
    {}
)  # Lưu authorization code: {code: {device_id, user_id, expires}}
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
        gender=user.gender
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    code = generate_verify_code()
    
    # FIX: Cũng lưu dưới dạng dictionary cho consistency
    verify_codes[user.email] = {
        "code": code,
        "device_id": None,  # Đăng ký chưa có device
        "device_info": {}
    }
    
    send_email(user.email, code, template_type="verification")
    logger.debug(f"Registered user: {user.username}, email: {user.email}, code: {code}, gender: {user.gender}")
    return {"message": "Registered, check email for code", "user_id": new_user.id}


@router.post("/login")
def login(
    user: UserLogin, 
    request: Request,
    db: Session = Depends(get_db), 
    response: Response = None
):
    # Tìm user
    db_user = db.query(User).filter(
        (User.username == user.username_or_email) | 
        (User.email == user.username_or_email)
    ).first()
    
    if not db_user:
        logger.error(f"User not found: {user.username_or_email}")
        raise HTTPException(400, "User not found")

    # Verify password
    try:
        if not verify_password(user.password, db_user.password_hash):
            logger.error(f"Invalid password for user: {user.username_or_email}")
            raise HTTPException(400, "Invalid password")
    except Exception as e:
        logger.error(f"Password verification failed: {str(e)}")
        raise HTTPException(400, "Password verification failed")

    # Tự động generate device fingerprint
    device_id = DeviceDetectionService.generate_device_fingerprint(request)
    device_info = DeviceDetectionService.get_device_info(request)
    
    logger.info(f"Auto-detected device: {device_id} for user: {db_user.id}")
    logger.info(f"Device info: {device_info}")

    # Check device verification
    is_verified = DeviceService.is_device_verified(db, db_user.id, device_id)
    
    if is_verified:
        # Device is verified, create token
        token = create_jwt(db_user.id, expires_delta=timedelta(days=7))
        
        # Set cookie
        if response:
            response.set_cookie(
                key="access_token",
                value=f"Bearer {token}",
                httponly=True,
                max_age=7*24*60*60,
                secure=True,
                samesite="lax"
            )
        
        logger.info(f"Login successful for user_id {db_user.id}, device: {device_id}")
        return {
            "message": "Login successful", 
            "token": token,
            "user_id": db_user.id
        }
    else:
        # Device needs verification
        code = generate_verify_code()
        # FIX: Đảm bảo luôn lưu dưới dạng dictionary
        verify_codes[db_user.email] = {
            "code": code,
            "device_id": device_id,
            "device_info": device_info
        }
        
        send_email(db_user.email, code, template_type="verification")
        
        logger.info(f"Verification required for user_id {db_user.id}, device: {device_id}")
        return {
            "message": "Device verification required", 
            "user_id": db_user.id,
            "email": db_user.email
        }

@router.post("/verify")
def verify(
    verify: VerifyCode, 
    user_id: int = Query(...), 
    request: Request = None,
    db: Session = Depends(get_db), 
    response: Response = None
):
    logger.debug(f"Verify request: user_id={user_id}, code={verify.code}")
    
    # Validate user
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        logger.error(f"User not found for user_id: {user_id}")
        raise HTTPException(status_code=404, detail="User not found")

    # Validate verification code
    if db_user.email not in verify_codes:
        logger.error(f"No verification code found for email: {db_user.email}")
        raise HTTPException(status_code=400, detail="No verification code found for this user")
    
    stored_data = verify_codes[db_user.email]
    
    # FIX: Kiểm tra kiểu dữ liệu của stored_data
    if isinstance(stored_data, dict):
        # Trường hợp mới: stored_data là dictionary
        if stored_data.get("code") != verify.code:
            logger.error(f"Invalid verification code for user_id {user_id}, provided: {verify.code}")
            raise HTTPException(status_code=400, detail="Invalid verification code")
        
        device_id = stored_data.get("device_id")
        device_info = stored_data.get("device_info", {})
    else:
        # Trường hợp cũ: stored_data là string (backward compatibility)
        if stored_data != verify.code:
            logger.error(f"Invalid verification code for user_id {user_id}, provided: {verify.code}")
            raise HTTPException(status_code=400, detail="Invalid verification code")
        
        # Tạo device_id từ request nếu có
        if request:
            device_id = DeviceDetectionService.generate_device_fingerprint(request)
            device_info = DeviceDetectionService.get_device_info(request)
        else:
            # Fallback: tạo random device_id
            import uuid
            device_id = f"fallback_{uuid.uuid4().hex}"
            device_info = {}

    # Add device to verified devices
    success = DeviceService.add_verified_device(db, user_id, device_id, device_info)
    if not success:
        logger.error(f"Failed to add device {device_id} for user {user_id}")
        raise HTTPException(status_code=500, detail="Failed to verify device")

    # Clean up
    del verify_codes[db_user.email]

    # Create JWT token
    token = create_jwt(db_user.id, expires_delta=timedelta(days=7))
    
    # Set HTTP-only cookie
    if response:
        response.set_cookie(
            key="access_token",
            value=f"Bearer {token}",
            httponly=True,
            max_age=7*24*60*60,
            secure=True,
            samesite="lax"
        )
    
    logger.info(f"Verification successful for user_id {user_id}, device_id: {device_id}")
    return {
        "message": "Device verified successfully", 
        "token": token,
        "user_id": user_id
    }


@router.get("/get-token")
def get_token(
    user_id: int = Query(...),
    device_id: str = Query(...),
    db: Session = Depends(get_db),
):
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
        logger.error(
            f"Device not verified for user_id: {user_id}, device_id: {device_id}"
        )
        raise HTTPException(status_code=400, detail="Device not verified")

    token = create_jwt(db_user.id, expires_delta=timedelta(days=7))
    logger.debug(
        f"Token retrieved for user_id: {user_id}, device_id: {normalized_device_id}, token: {token}"
    )
    return {"token": token}


@router.get("/validate-token")
def validate_token(user_id: int = Depends(verify_jwt)):
    return {"message": "Token is valid", "user_id": user_id}


@router.get("/authorize")
def authorize(
    device_id: str = Query(...),
    redirect_uri: str = Query(...),
    state: str = Query(...),
    db: Session = Depends(get_db),
):
    """
    Endpoint để tạo URL đăng nhập cho OAuth flow.
    """
    auth_code = secrets.token_urlsafe(32)
    auth_codes[auth_code] = {
        "device_id": device_id,
        "user_id": None,  # Chưa xác thực
        "expires": datetime.utcnow()
        + timedelta(minutes=10),  # Code hết hạn sau 10 phút
        "state": state,
    }
    login_url = f"https://living-tortoise-polite.ngrok-free.app/?code={auth_code}&state={state}&redirect_uri={redirect_uri}"
    logger.debug(f"Generated login_url: {login_url}")
    return {"login_url": login_url}


@router.post("/token")
def exchange_token(
    code: str = Query(...), state: str = Query(...), db: Session = Depends(get_db)
):
    """
    Đổi authorization code lấy JWT token.
    """
    logger.debug(f"Exchange token request: code={code}, state={state}")
    auth_data = auth_codes.get(code)
    if (
        not auth_data
        or auth_data["state"] != state
        or auth_data["expires"] < datetime.utcnow()
    ):
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
        logger.error(
            f"Device not verified for user_id: {user_id}, device_id: {device_id}"
        )
        raise HTTPException(status_code=400, detail="Device not verified")

    token = create_jwt(user_id, expires_delta=timedelta(days=7))
    del auth_codes[code]  # Xóa code sau khi sử dụng
    logger.debug(
        f"Token issued for user_id: {user_id}, device_id: {device_id}, token: {token}"
    )
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
        "expires": datetime.utcnow() + timedelta(hours=1),
    }
    send_email(db_user.email, reset_token, template_type="reset_password")
    logger.debug(
        f"Reset password link sent to {db_user.email} for user_id {db_user.id}"
    )
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


@router.get("/devices")
def get_user_devices(
    request: Request,  # Thêm request
    user_id: int = Depends(verify_jwt), 
    db: Session = Depends(get_db)
):
    """Lấy danh sách devices đã verify của user"""
    devices = DeviceService.get_verified_devices(db, user_id)
    
    # Thêm thông tin về device hiện tại
    current_device_id = DeviceDetectionService.generate_device_fingerprint(request)
    
    for device in devices:
        device["is_current"] = device.get("device_id") == current_device_id
    
    return {
        "verified_devices": devices,
        "current_device_id": current_device_id
    }

@router.delete("/devices/{device_id}")
def remove_device(
    device_id: str,
    user_id: int = Depends(verify_jwt), 
    db: Session = Depends(get_db)
):
    """Xóa một device khỏi danh sách verified"""
    success = DeviceService.remove_verified_device(db, user_id, device_id)
    
    if success:
        return {"message": f"Device {device_id} removed successfully"}
    else:
        raise HTTPException(404, f"Device {device_id} not found")
