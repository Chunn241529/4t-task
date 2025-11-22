import logging
from typing import Dict, Optional
from sqlalchemy.orm import Session
from fastapi import Request
from app.models import User
from app.utils import verify_password, create_jwt, generate_verify_code, send_email
from app.services.device_detection import DeviceDetectionService
from app.services.device_service import DeviceService
from datetime import timedelta

logger = logging.getLogger(__name__)


class AuthService:
    @staticmethod
    async def login_user(
        username_or_email: str,
        password: str,
        request: Request,
        db: Session,
        device_id: Optional[str] = None,
    ) -> Dict:
        """
        Xử lý đăng nhập user với device detection tự động
        """
        try:
            # Tìm user
            db_user = (
                db.query(User)
                .filter(
                    (User.username == username_or_email)
                    | (User.email == username_or_email)
                )
                .first()
            )

            if not db_user:
                logger.error(f"User not found: {username_or_email}")
                return {"error": "User not found", "status": 400}

            # Verify password
            try:
                if not verify_password(password, db_user.password_hash):
                    logger.error(f"Invalid password for user: {username_or_email}")
                    return {"error": "Invalid password", "status": 400}
            except Exception as e:
                logger.error(f"Password verification failed: {str(e)}")
                return {"error": "Password verification failed", "status": 400}

            # Device identification strategy
            device_info = {}

            if device_id:
                # Case 1: Client provided device_id (Preferred)
                logger.info(f"Using client-provided device_id: {device_id}")
                try:
                    device_info = DeviceDetectionService.get_device_info(request)
                except:
                    pass
            else:
                # Case 2: Server-side detection (Fallback)
                try:
                    device_id = DeviceDetectionService.generate_device_fingerprint(
                        request
                    )
                    device_info = DeviceDetectionService.get_device_info(request)

                    logger.info(
                        f"Auto-detected device: {device_id} for user: {db_user.id}"
                    )
                    logger.info(f"Device info: {device_info}")
                except Exception as e:
                    logger.error(f"Error detecting device: {str(e)}")
                    # Fallback: tạo random device_id
                    import uuid

                    device_id = f"fallback_{uuid.uuid4().hex}"
                    device_info = {"error": "Device detection failed"}

            # Check device verification
            is_verified = DeviceService.is_device_verified(db, db_user.id, device_id)

            if is_verified:
                # Device is verified, create token
                token = create_jwt(db_user.id, expires_delta=timedelta(days=7))

                logger.info(
                    f"Login successful for user_id {db_user.id}, device: {device_id}"
                )
                return {
                    "message": "Login successful",
                    "token": token,
                    "user_id": db_user.id,
                    "status": 200,
                }
            else:
                # Device needs verification
                code = generate_verify_code()
                from app.routers.auth import verify_codes  # Import từ auth router

                verify_codes[db_user.email] = {
                    "code": code,
                    "device_id": device_id,
                    "device_info": device_info,
                }

                send_email(db_user.email, code, template_type="verification")

                logger.info(
                    f"Verification required for user_id {db_user.id}, device: {device_id}"
                )
                return {
                    "message": "Device verification required",
                    "user_id": db_user.id,
                    "email": db_user.email,
                    "status": 200,
                }

        except Exception as e:
            logger.error(f"Login service error: {str(e)}")
            return {"error": "Login failed", "status": 500}

    @staticmethod
    async def verify_user(
        user_id: int, code: str, request: Request, db: Session
    ) -> Dict:
        """
        Xử lý verify user với device detection tự động
        """
        try:
            from app.routers.auth import verify_codes  # Import từ auth router

            # Validate user
            db_user = db.query(User).filter(User.id == user_id).first()
            if not db_user:
                logger.error(f"User not found for user_id: {user_id}")
                return {"error": "User not found", "status": 404}

            # Validate verification code
            if db_user.email not in verify_codes:
                logger.error(f"No verification code found for email: {db_user.email}")
                return {
                    "error": "No verification code found for this user",
                    "status": 400,
                }

            stored_data = verify_codes[db_user.email]

            # Xử lý device_id
            device_id = None
            device_info = {}

            if isinstance(stored_data, dict):
                # Trường hợp mới: stored_data là dictionary
                if stored_data.get("code") != code:
                    logger.error(
                        f"Invalid verification code for user_id {user_id}, provided: {code}"
                    )
                    return {"error": "Invalid verification code", "status": 400}

                device_id = stored_data.get("device_id")
                device_info = stored_data.get("device_info", {})
                logger.debug(f"Got device_id from stored_data: {device_id}")
            else:
                # Trường hợp cũ: stored_data là string
                if stored_data != code:
                    logger.error(
                        f"Invalid verification code for user_id {user_id}, provided: {code}"
                    )
                    return {"error": "Invalid verification code", "status": 400}

                logger.debug(
                    "Using string stored_data, device_id will be generated from request"
                )

            # FIX QUAN TRỌNG: Nếu device_id là None, tạo từ request
            if device_id is None:
                logger.debug("device_id is None, generating from request...")
                if request:
                    try:
                        device_id = DeviceDetectionService.generate_device_fingerprint(
                            request
                        )
                        device_info = DeviceDetectionService.get_device_info(request)
                        logger.debug(f"Generated device_id from request: {device_id}")
                    except Exception as e:
                        logger.error(
                            f"Error generating device_id from request: {str(e)}"
                        )
                        # Fallback
                        import uuid

                        device_id = f"fallback_{uuid.uuid4().hex}"
                        device_info = {"error": "Device detection failed"}
                else:
                    # Fallback cuối cùng
                    import uuid

                    device_id = f"fallback_{uuid.uuid4().hex}"
                    device_info = {"error": "No request available"}
                    logger.debug(f"Using fallback device_id: {device_id}")

            # Add device to verified devices
            success = DeviceService.add_verified_device(
                db, user_id, device_id, device_info
            )
            if not success:
                logger.error(f"Failed to add device {device_id} for user {user_id}")
                return {"error": "Failed to verify device", "status": 500}

            # Clean up
            del verify_codes[db_user.email]

            # Create JWT token
            token = create_jwt(db_user.id, expires_delta=timedelta(days=7))

            logger.info(
                f"Verification successful for user_id {user_id}, device_id: {device_id}"
            )
            return {
                "message": "Device verified successfully",
                "token": token,
                "user_id": user_id,
                "status": 200,
            }

        except Exception as e:
            logger.error(f"Verification service error: {str(e)}")
            return {"error": "Verification failed", "status": 500}
