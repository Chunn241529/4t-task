import logging
from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from app.models import User
from datetime import datetime, timezone
import uuid

logger = logging.getLogger(__name__)

class DeviceService:
    @staticmethod
    def _normalize_device_id(device_id: str) -> str:
        """Chuẩn hóa device_id và đảm bảo không bao giờ trả về None"""
        if device_id is None:
            logger.error("device_id is None in _normalize_device_id")
            return f"error_fallback_{uuid.uuid4().hex}"
        
        if not isinstance(device_id, str):
            logger.warning(f"device_id is not string: {type(device_id)}, converting to string")
            device_id = str(device_id)
            
        return device_id.strip().lower()

    @staticmethod
    def add_verified_device(db: Session, user_id: int, device_id: str, device_info: Optional[Dict] = None) -> bool:
        """Thêm device vào danh sách verified devices với thông tin device"""
        try:
            logger.debug(f"Adding verified device: user_id={user_id}, device_id={device_id}, device_info={device_info}")
            
            # FIX: Kiểm tra device_id không được None
            if device_id is None:
                logger.error("device_id is None, cannot add device")
                return False
                
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                logger.error(f"User {user_id} not found")
                return False
            
            # Chuẩn hóa device_id
            normalized_device_id = DeviceService._normalize_device_id(device_id)
            logger.debug(f"Normalized device_id: {normalized_device_id}")
            
            # Khởi tạo nếu chưa có
            if user.verified_devices is None:
                user.verified_devices = []
                logger.debug(f"Initialized verified_devices for user {user_id}")
            
            # DEBUG: Log current verified_devices
            logger.debug(f"Current verified_devices for user {user_id}: {user.verified_devices}")
            
            # Kiểm tra device đã tồn tại chưa
            existing_devices = []
            for device in user.verified_devices:
                if isinstance(device, dict):
                    existing_devices.append(device.get('device_id'))
                else:
                    existing_devices.append(device)
            
            existing_devices = [DeviceService._normalize_device_id(d) for d in existing_devices if d is not None]
            logger.debug(f"Existing devices (normalized): {existing_devices}")
            
            if normalized_device_id in existing_devices:
                logger.debug(f"Device {normalized_device_id} already verified for user {user_id}")
                return True
            
            # Tạo device entry với thông tin chi tiết
            current_time = datetime.now(timezone.utc).isoformat()
            
            device_entry = {
                "device_id": normalized_device_id,
                "verified_at": current_time,
                **(device_info or {})  # Đảm bảo device_info không phải None
            }
            
            logger.debug(f"Adding device entry: {device_entry}")
            user.verified_devices.append(device_entry)
            
            db.commit()
            db.refresh(user)
            
            logger.info(f"Added device {normalized_device_id} to verified devices for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding verified device: {str(e)}")
            logger.error(f"Full error details:", exc_info=True)
            db.rollback()
            return False

    @staticmethod
    def is_device_verified(db: Session, user_id: int, device_id: str) -> bool:
        """Kiểm tra device đã được verify chưa"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user or not user.verified_devices:
                return False
            
            # Chuẩn hóa device_id đầu vào
            normalized_device_id = DeviceService._normalize_device_id(device_id)
            
            # Extract device_id từ các entry (có thể là string hoặc dict)
            existing_devices = []
            for device in user.verified_devices:
                if isinstance(device, dict):
                    existing_devices.append(device.get('device_id'))
                else:
                    existing_devices.append(device)
            
            existing_devices = [DeviceService._normalize_device_id(d) for d in existing_devices if d is not None]
            
            return normalized_device_id in existing_devices
            
        except Exception as e:
            logger.error(f"Error checking device verification: {str(e)}")
            return False

    @staticmethod
    def get_verified_devices(db: Session, user_id: int) -> List[Dict]:
        """Lấy danh sách verified devices với thông tin đầy đủ"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user or not user.verified_devices:
                return []
            
            # Chuẩn hóa định dạng trả về
            devices = []
            for device in user.verified_devices:
                if isinstance(device, dict):
                    devices.append(device)
                else:
                    devices.append({
                        "device_id": device,
                        "verified_at": None,
                        "browser": "Unknown",
                        "os": "Unknown",
                        "type": "Unknown"
                    })
            
            return devices
            
        except Exception as e:
            logger.error(f"Error getting verified devices: {str(e)}")
            return []

    @staticmethod
    def remove_verified_device(db: Session, user_id: int, device_id: str) -> bool:
        """Xóa device khỏi danh sách verified"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user or not user.verified_devices:
                return False
            
            normalized_device_id = DeviceService._normalize_device_id(device_id)
            original_count = len(user.verified_devices)
            
            # Lọc bỏ device
            user.verified_devices = [
                d for d in user.verified_devices 
                if (isinstance(d, dict) and DeviceService._normalize_device_id(d.get('device_id')) != normalized_device_id) or
                   (not isinstance(d, dict) and DeviceService._normalize_device_id(d) != normalized_device_id)
            ]
            
            if len(user.verified_devices) < original_count:
                db.commit()
                db.refresh(user)
                logger.info(f"Removed device {device_id} from user {user_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error removing verified device: {str(e)}")
            db.rollback()
            return False
