import os
import platform

# API Base URL
API_BASE_URL: str = os.getenv(
    "API_URL", "https://living-tortoise-polite.ngrok-free.app"
)


# Hàm lấy thư mục cấu hình phù hợp theo hệ điều hành
def get_config_dir() -> str:
    system = platform.system()
    home = os.path.expanduser("~")  # Lấy thư mục home của người dùng

    if system == "Windows":
        # Trên Windows, sử dụng AppData\Local hoặc AppData\Roaming
        config_dir = os.getenv("LOCALAPPDATA", os.path.join(home, "AppData", "Local"))
        return os.path.join(config_dir, "FourT_task")
    else:
        # Trên Linux/macOS, sử dụng ~/.config/
        return os.path.join(home, ".config", "FourT_task")


# Đường dẫn tới file token
TOKEN_FILE_PATH: str = os.path.join(get_config_dir(), ".FourT_task_token")
