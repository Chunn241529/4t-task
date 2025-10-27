import asyncio
import logging
from telegram import Update
from telegram.ext import CallbackContext
from telegram.constants import ParseMode
from handlers.base import BaseHandler

logger = logging.getLogger(__name__)

class CommandHandlers(BaseHandler):
    
    @staticmethod
    async def start_command(update: Update, context: CallbackContext):
        """Handle /start command: Welcome user."""
        welcome_text = """
🤖 **Chào mừng bạn đến với AI Assistant!**

Để bắt đầu, hãy:
1. Sử dụng `/login` để lấy thông tin đăng nhập
2. Sử dụng `/token <your_token>` để xác thực
3. Bắt đầu trò chuyện!

**Các lệnh có sẵn:**
/start - Hiển thị thông tin này
/login - Lấy URL đăng nhập
/token - Xác thực token
/reset - Reset hội thoại hiện tại
/help - Hiển thị trợ giúp
/status - Kiểm tra trạng thái phiên
        """
        await update.message.reply_text(welcome_text, parse_mode=ParseMode.MARKDOWN)
    
    @staticmethod
    async def help_command(update: Update, context: CallbackContext):
        """Handle /help command."""
        help_text = """
🆘 **Trợ giúp**

**Hỗ trợ file:**
- 📄 PDF, DOCX, TXT
- 📊 CSV, Excel
- 🖼️ Hình ảnh (JPEG, PNG)

**Cách sử dụng:**
1. Đăng nhập và lấy token với `/login`
2. Xác thực với `/token <your_token>`
3. Gửi tin nhắn hoặc file để trò chuyện
4. Dùng `/reset` để bắt đầu hội thoại mới

**Lưu ý:**
- Phiên sẽ tự động hết hạn sau 60 phút không hoạt động
- Có thể gửi nhiều file trong cùng hội thoại
        """
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)
    
    @staticmethod
    async def login_command(update: Update, context: CallbackContext):
        """Handle /login command: Send login URL to user."""
        from config.settings import settings
        
        login_url = settings.FASTAPI_BACKEND_URL
        await update.message.reply_text(
            f"🔐 **Đăng nhập**\n\n"
            f"Vui lòng truy cập {login_url} để đăng nhập và lấy token.\n\n"
            f"Sau đó, sử dụng lệnh:\n`/token <your_token>`\n\n"
            f"Để xác thực và bắt đầu sử dụng bot.",
            parse_mode=ParseMode.MARKDOWN
        )
    
    @staticmethod
    async def token_command(update: Update, context: CallbackContext):
        """Handle /token command: Store user's authentication token."""
        from managers.session_manager import session_manager
        
        user_id = update.message.from_user.id
        
        if not context.args:
            await update.message.reply_text(
                "❌ Vui lòng cung cấp token:\n\n"
                "`/token your_token_here`",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        token = context.args[0].strip()
        if len(token) < 10:
            await update.message.reply_text("❌ Token không hợp lệ. Vui lòng kiểm tra lại.")
            return

        session = session_manager.get_session(user_id)
        session.auth_token = token
        session.update_activity()
        
        await update.message.reply_text(
            "✅ **Đã xác thực token thành công!**\n\n"
            "Bạn có thể bắt đầu trò chuyện ngay bây giờ!",
            parse_mode=ParseMode.MARKDOWN
        )
    
    @staticmethod
    async def status_command(update: Update, context: CallbackContext):
        """Handle /status command: Show current session status."""
        from managers.session_manager import session_manager
        
        user_id = update.message.from_user.id
        session = session_manager.get_session(user_id)
        
        if not session.auth_token:
            status_text = "❌ **Chưa xác thực**\nSử dụng `/token` để xác thực."
        else:
            conversation_status = "✅ Có" if session.conversation_id else "❌ Không"
            message_count = len(session.history)
            last_active = session.last_activity.strftime("%H:%M:%S %d/%m/%Y")
            
            status_text = f"""
📊 **Trạng thái phiên**

🔐 **Xác thực:** ✅ Đã xác thực
💬 **Hội thoại:** {conversation_status}
📝 **Số tin nhắn:** {message_count}
⏰ **Hoạt động cuối:** {last_active}
            """.strip()
        
        await update.message.reply_text(status_text, parse_mode=ParseMode.MARKDOWN)
    
    @staticmethod
    async def reset_command(update: Update, context: CallbackContext):
        """Handle /reset command: Reset conversation COMPLETELY."""
        from managers.session_manager import session_manager
        from managers.typing_manager import typing_manager
        from services.api_client import api_client
        
        chat_id = update.message.chat_id
        
        async def reset_operation():
            user_id = update.message.from_user.id
            session = CommandHandlers.get_user_session(update)
            
            if not await CommandHandlers.ensure_authenticated(update, session):
                return None

            # Thêm delay để typing hiển thị
            await asyncio.sleep(1)
            
            try:
                # Delete conversation from backend if exists
                if session.conversation_id:
                    await api_client.delete_conversation(session.auth_token, session.conversation_id)
                
                # COMPLETELY reset session
                old_auth_token = session.auth_token
                session_manager.delete_session(user_id)
                
                # Create brand new session
                new_session = session_manager.get_session(user_id)
                new_session.auth_token = old_auth_token
                new_session.conversation_id = None
                new_session.history.clear()
                new_session.update_activity()
                
                return "🔄 **Đã reset phiên hội thoại hoàn toàn!**\n\nBắt đầu hội thoại mới sạch sẽ ngay bây giờ!"
                
            except Exception as e:
                logger.error(f"Error in reset: {e}")
                # Fallback
                session_manager.delete_session(user_id)
                new_session = session_manager.get_session(user_id)
                new_session.auth_token = session.auth_token
                
                return "🔄 **Đã reset phiên hội thoại!**\n\nBắt đầu hội thoại mới!"
        
        # Sử dụng typing manager cho command
        try:
            result_message = await typing_manager.with_typing(
                context.bot,
                chat_id,
                reset_operation()
            )
            if result_message:
                await update.message.reply_text(result_message, parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            logger.error(f"Error in reset command: {e}")
            await update.message.reply_text(
                "❌ Có lỗi xảy ra khi reset. Vui lòng thử lại.",
                parse_mode=ParseMode.MARKDOWN
            )
