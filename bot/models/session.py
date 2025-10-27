from datetime import datetime
from typing import List, Dict, Optional

class UserSession:
    def __init__(self):
        self.history: List[Dict] = []
        self.auth_token: Optional[str] = None
        self.conversation_id: Optional[str] = None
        self.last_activity: datetime = datetime.now()
    
    def update_activity(self):
        self.last_activity = datetime.now()
    
    def is_expired(self, timeout_minutes: int = 60) -> bool:
        return (datetime.now() - self.last_activity).total_seconds() > timeout_minutes * 60
    
    async def add_to_history(self, message: str, role: str = "user"):
        self.history.append({"role": role, "content": message})
        self.update_activity()
    
    async def retrieve_context(self, k: int = 3) -> str:
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history[-k:]])
    
    def clear(self):
        self.history.clear()
        self.conversation_id = None
        self.update_activity()
