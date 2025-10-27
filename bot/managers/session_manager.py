import asyncio
import logging
from typing import Dict
from models.session import UserSession
from config.settings import settings

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self):
        self.sessions: Dict[int, UserSession] = {}
    
    def get_session(self, user_id: int) -> UserSession:
        if user_id not in self.sessions:
            self.sessions[user_id] = UserSession()
        return self.sessions[user_id]
    
    def delete_session(self, user_id: int):
        if user_id in self.sessions:
            del self.sessions[user_id]
    
    async def cleanup_expired_sessions(self):
        """Clean up expired user sessions periodically"""
        while True:
            await asyncio.sleep(300)  # Check every 5 minutes
            expired_users = [
                user_id for user_id, session in self.sessions.items()
                if session.is_expired(settings.SESSION_TIMEOUT_MINUTES)
            ]
            
            for user_id in expired_users:
                del self.sessions[user_id]
                logger.info(f"Cleaned up expired session for user {user_id}")

# Global session manager instance
session_manager = SessionManager()
