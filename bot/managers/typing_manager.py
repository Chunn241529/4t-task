import asyncio
import logging
from typing import Dict
from telegram import Bot
from telegram.constants import ChatAction
from telegram.error import TelegramError, Forbidden

logger = logging.getLogger(__name__)

class TypingManager:
    def __init__(self):
        self.active_tasks: Dict[int, asyncio.Task] = {}
        self.stop_events: Dict[int, asyncio.Event] = {}
    
    async def start_typing(self, bot: Bot, chat_id: int):
        """Start typing indicator with continuous renewal"""
        if chat_id in self.active_tasks:
            return  # Already typing
        
        # Create stop event for this chat
        stop_event = asyncio.Event()
        self.stop_events[chat_id] = stop_event
        
        # Start typing task
        self.active_tasks[chat_id] = asyncio.create_task(
            self._typing_loop(bot, chat_id, stop_event)
        )
        logger.debug(f"Started typing indicator for chat {chat_id}")
    
    async def stop_typing(self, chat_id: int):
        """Stop typing indicator"""
        if chat_id in self.stop_events:
            self.stop_events[chat_id].set()
        
        if chat_id in self.active_tasks:
            try:
                self.active_tasks[chat_id].cancel()
                await self.active_tasks[chat_id]
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error stopping typing for {chat_id}: {e}")
            finally:
                # Clean up
                if chat_id in self.active_tasks:
                    del self.active_tasks[chat_id]
                if chat_id in self.stop_events:
                    del self.stop_events[chat_id]
            
            logger.debug(f"Stopped typing indicator for chat {chat_id}")
    
    async def _typing_loop(self, bot: Bot, chat_id: int, stop_event: asyncio.Event):
        """Continuous typing loop that renews every 4 seconds"""
        try:
            while not stop_event.is_set():
                try:
                    await bot.send_chat_action(
                        chat_id=chat_id, 
                        action=ChatAction.TYPING
                    )
                    logger.debug(f"Sent typing action to chat {chat_id}")
                    
                    # Wait for 4 seconds or until stopped
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=4.0)
                    except asyncio.TimeoutError:
                        continue  # Continue to next iteration for renewal
                        
                except Forbidden as e:
                    logger.warning(f"Bot blocked by user {chat_id}: {e}")
                    break
                except TelegramError as e:
                    logger.error(f"Telegram error in typing loop for {chat_id}: {e}")
                    await asyncio.sleep(1)  # Wait before retry
                except Exception as e:
                    logger.error(f"Unexpected error in typing loop for {chat_id}: {e}")
                    break
                    
        except asyncio.CancelledError:
            logger.debug(f"Typing loop for chat {chat_id} was cancelled")
        except Exception as e:
            logger.error(f"Critical error in typing loop for {chat_id}: {e}")
    
    async def with_typing(self, bot: Bot, chat_id: int, coro):
        """
        Context manager for typing indicator.
        Ensures typing shows during entire coroutine execution.
        """
        await self.start_typing(bot, chat_id)
        try:
            result = await coro
            return result
        finally:
            await self.stop_typing(chat_id)

# Global typing manager instance
typing_manager = TypingManager()
