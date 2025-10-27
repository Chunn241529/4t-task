import asyncio
import logging
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from config.settings import settings
from handlers import CommandHandlers, MessageHandlers, BaseHandler
from managers.session_manager import session_manager

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def setup_handlers(application: Application):
    """Set up all handlers for the bot."""
    # Command handlers
    application.add_handler(CommandHandler('start', CommandHandlers.start_command))
    application.add_handler(CommandHandler('help', CommandHandlers.help_command))
    application.add_handler(CommandHandler('login', CommandHandlers.login_command))
    application.add_handler(CommandHandler('token', CommandHandlers.token_command))
    application.add_handler(CommandHandler('status', CommandHandlers.status_command))
    application.add_handler(CommandHandler('reset', CommandHandlers.reset_command))
    
    # Message handlers
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, 
        MessageHandlers.handle_message
    ))
    application.add_handler(MessageHandler(
        filters.Document.ALL | filters.PHOTO, 
        MessageHandlers.handle_message
    ))
    
    # Error handler
    application.add_error_handler(BaseHandler.error_handler)

def main():
    """Main function to start the bot."""
    # Create application
    application = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()
    
    # Setup handlers
    setup_handlers(application)
    
    # Start session cleanup task
    asyncio.get_event_loop().create_task(session_manager.cleanup_expired_sessions())
    
    logger.info("Bot is starting...")
    application.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()
