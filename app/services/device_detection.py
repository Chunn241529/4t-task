import hashlib
import uuid
from typing import Dict, Optional
from fastapi import Request
import user_agents
import logging

logger = logging.getLogger(__name__)


class DeviceDetectionService:
    @staticmethod
    def generate_device_fingerprint(request: Request) -> str:
        """
        Tạo device fingerprint duy nhất từ thông tin request
        """
        try:
            # Lấy các thông tin từ request
            user_agent = request.headers.get("User-Agent", "")
            accept_language = request.headers.get("Accept-Language", "")
            accept_encoding = request.headers.get("Accept-Encoding", "")
            client_ip = request.client.host if request.client else "unknown"

            # Parse User-Agent để lấy thông tin chi tiết
            ua = user_agents.parse(user_agent)

            # Tạo fingerprint string từ các thông tin
            fingerprint_data = {
                "browser_family": ua.browser.family,
                "browser_version": ua.browser.version_string,
                "os_family": ua.os.family,
                "os_version": ua.os.version_string,
                "device_family": ua.device.family,
                "is_mobile": ua.is_mobile,
                "is_tablet": ua.is_tablet,
                "is_pc": ua.is_pc,
                "accept_language": accept_language,
                "accept_encoding": accept_encoding,
                "ip_prefix": (
                    client_ip.rsplit(".", 1)[0] if "." in client_ip else client_ip
                ),
            }

            # Tạo hash từ fingerprint data
            fingerprint_str = str(sorted(fingerprint_data.items()))
            device_id = hashlib.sha256(fingerprint_str.encode()).hexdigest()[:32]

            logger.debug(f"Generated device fingerprint: {device_id}")
            return device_id

        except Exception as e:
            logger.error(f"Error generating device fingerprint: {str(e)}")
            # Fallback: tạo random device ID
            return f"fallback_{uuid.uuid4().hex}"

    @staticmethod
    def get_device_info(request: Request) -> Dict:
        """
        Lấy thông tin chi tiết về device
        """
        try:
            user_agent = request.headers.get("User-Agent", "")
            ua = user_agents.parse(user_agent)

            return {
                "browser": f"{ua.browser.family} {ua.browser.version_string}",
                "os": f"{ua.os.family} {ua.os.version_string}",
                "device": ua.device.family,
                "type": (
                    "mobile"
                    if ua.is_mobile
                    else "tablet" if ua.is_tablet else "desktop"
                ),
                "user_agent": user_agent[:200],  # Giới hạn độ dài
            }
        except Exception as e:
            logger.error(f"Error getting device info: {str(e)}")
            return {
                "browser": "Unknown",
                "os": "Unknown",
                "device": "Unknown",
                "type": "Unknown",
                "error": str(e),
            }
