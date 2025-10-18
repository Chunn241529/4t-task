import os

API_BASE_URL: str = os.getenv("API_URL", "http://127.0.0.1:8000")
TOKEN_FILE_PATH: str = os.path.expanduser("~/.4t_task_token")
