import logging
from telegram import Update
from telegram.ext import CallbackContext
from managers.session_manager import session_manager
from managers.typing_manager import typing_manager

logger = logging.getLogger(__name__)

class BaseHandler:
    """Base handler with common functionality"""
    
    @staticmethod
    async def error_handler(update: Update, context: CallbackContext):
        """Handle unexpected errors and notify the user."""
        logger.error(f"Update {update} caused error: {context.error}", exc_info=True)
        
        if update and update.message:
            try:
                await update.message.reply_text(
                    "❌ Có lỗi xảy ra. Vui lòng thử lại sau hoặc sử dụng /help để biết thêm thông tin."
                )
            except Exception as e:
                logger.error(f"Could not send error message: {e}")
    
    @staticmethod
    def get_user_session(update: Update):
        """Get or create user session."""
        user_id = update.message.from_user.id
        return session_manager.get_session(user_id)
    
    @staticmethod
    async def ensure_authenticated(update: Update, session) -> bool:
        """Check if user is authenticated."""
        if not session.auth_token:
            await update.message.reply_text(
                "🔐 **Cần xác thực**\n\n"
                "Vui lòng sử dụng `/token <your_token>` để xác thực trước khi trò chuyện.",
                parse_mode="MARKDOWN"
            )
            return False
        return True
    
    @staticmethod
    async def with_typing(chat_id, coro):
        """Context manager for typing indicator."""
        await typing_manager.start_typing(coro.__self__.application.bot, chat_id)
        try:
            result = await coro
            return result
        finally:
            await typing_manager.stop_typing(chat_id)
