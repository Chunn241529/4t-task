import logging
import asyncio
from telegram import Update
from telegram.ext import CallbackContext
from telegram.constants import ParseMode
from handlers.base import BaseHandler

logger = logging.getLogger(__name__)

class MessageHandlers(BaseHandler):
    
    @staticmethod
    async def handle_message(update: Update, context: CallbackContext):
        """Handle all incoming messages and files."""
        from managers.typing_manager import typing_manager
        from services.api_client import api_client
        
        session = MessageHandlers.get_user_session(update)
        chat_id = update.message.chat_id
        bot = context.bot
        
        # Check authentication
        if not await MessageHandlers.ensure_authenticated(update, session):
            return

        session.update_activity()

        # Extract message content and files
        query, file_data, file_name = await MessageHandlers._extract_message_content(update)
        
        # Add to history
        if query:
            await session.add_to_history(query)
        elif file_data:
            await session.add_to_history(f"(File: {file_name})")

        # SỬ DỤNG TYPING MANAGER - GIỐNG CÁCH CLIENT TUI XỬ LÝ STREAMING
        try:
            # Bắt đầu typing và duy trì trong suốt quá trình xử lý
            response_content = await typing_manager.with_typing(
                bot, 
                chat_id,
                MessageHandlers._process_streaming_message(session, query, file_data, file_name, chat_id, bot)
            )
            
            # Send final response
            if response_content:
                await MessageHandlers._send_response(update, response_content)
                await session.add_to_history(response_content, role="assistant")
                
        except Exception as e:
            await MessageHandlers._handle_error(update, e, session)
    
    @staticmethod
    async def _process_streaming_message(session, query: str, file_data: bytes, file_name: str, chat_id: int, bot) -> str:
        """Process message with streaming - inspired by TUI client approach"""
        from services.api_client import api_client
        
        # Tạo conversation mới nếu cần (giống client TUI)
        if not session.conversation_id:
            session.conversation_id = await api_client.create_conversation(session.auth_token)
            logger.info(f"Created NEW conversation {session.conversation_id}")
            session.history.clear()

        # Stream response - tích lũy nội dung giống client TUI
        accumulated_content = ""
        
        try:
            async for data_chunk in api_client.chat_stream(
                session.auth_token,
                session.conversation_id,
                query or f"File: {file_name}",
                file_data,
                file_name
            ):
                # Xử lý các loại data chunk giống client TUI
                if "conversation_id" in data_chunk:
                    session.conversation_id = data_chunk["conversation_id"]
                    continue
                elif data_chunk.get("done"):
                    break
                elif data_chunk.get("error"):
                    raise Exception(f"Backend error: {data_chunk['error']}")
                elif data_chunk.get("tool_calls"):
                    # Có thể thêm xử lý tool calls ở đây nếu cần
                    logger.debug(f"Tool calls detected: {data_chunk['tool_calls']}")
                    continue
                elif data_chunk.get("content"):
                    decoded_content = data_chunk["content"].encode().decode("utf-8", errors="replace")
                    accumulated_content += decoded_content
                    
                    # Có thể gửi partial updates nếu muốn (giống client TUI update real-time)
                    # Nhưng với Telegram có thể gửi toàn bộ ở cuối cho đơn giản
                    
        except Exception as e:
            logger.error(f"Error in chat stream: {e}")
            raise
        
        return accumulated_content
    
    @staticmethod
    async def _extract_message_content(update: Update):
        """Extract text and file data from update."""
        query = update.message.text.strip() if update.message.text else ""
        file_data = None
        file_name = None
        
        if update.message.document:
            file = await update.message.document.get_file()
            file_name = update.message.document.file_name
            file_data = await file.download_as_bytearray()
            # Thông báo đang xử lý file - giống client TUI
            await update.message.reply_text(f"📄 Đang xử lý file: {file_name}...")
            
        elif update.message.photo:
            file = await update.message.photo[-1].get_file()
            file_name = "photo.jpg"
            file_data = await file.download_as_bytearray()
            await update.message.reply_text("🖼️ Đang xử lý hình ảnh...")
        
        return query, file_data, file_name
    
    @staticmethod
    async def _send_response(update: Update, content: str):
        """Send formatted response to user."""
        from utils.formatters import format_response_content, split_long_message
        
        if not content:
            await update.message.reply_text("🤖 Bot không có phản hồi.")
            return
        
        formatted_content = format_response_content(content)
        chunks = split_long_message(formatted_content)
        
        for chunk in chunks:
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN_V2)
    
    @staticmethod
    async def _handle_error(update: Update, error: Exception, session):
        """Handle errors during message processing."""
        logger.error(f"Error in handle_message: {error}")
        
        error_msg = str(error).lower()
        
        if '401' in error_msg or '403' in error_msg:
            await update.message.reply_text(
                "❌ **Token không hợp lệ hoặc đã hết hạn**\n\nVui lòng sử dụng `/token` để xác thực lại.",
                parse_mode=ParseMode.MARKDOWN
            )
            session.auth_token = None
        elif 'timeout' in error_msg:
            await update.message.reply_text(
                "⏰ **Yêu cầu hết thời gian**\n\nVui lòng thử lại.",
                parse_mode=ParseMode.MARKDOWN
            )
        elif 'connection' in error_msg or 'connect' in error_msg:
            await update.message.reply_text(
                "🔌 **Không thể kết nối đến server**\n\nVui lòng thử lại sau.",
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(
                "❌ **Có lỗi xảy ra**\n\nVui lòng thử lại sau hoặc sử dụng /help để biết thêm thông tin.",
                parse_mode=ParseMode.MARKDOWN
            )
