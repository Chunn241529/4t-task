import re
from telegram.constants import ParseMode

def escape_markdown_v2(text: str) -> str:
    """Escape special characters for MarkdownV2"""
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

def format_response_content(content: str) -> str:
    """Format response content for Telegram with proper Markdown"""
    if not content:
        return "🤖 Bot không có phản hồi."
    
    # Escape content first
    escaped_content = escape_markdown_v2(content)
    
    # Check if content contains code blocks
    if '```' in content:
        # Preserve existing code blocks
        return escaped_content
    elif any(keyword in content.lower() for keyword in [
        'def ', 'import ', 'class ', 'function ', 'print(', 
        '<?php', '<html', 'var ', 'const ', 'let ', '#include'
    ]):
        # Format as code if programming keywords detected
        return f"```\n{escaped_content}\n```"
    else:
        # Regular text with proper escaping
        return escaped_content

def split_long_message(message: str, max_length: int = 4000) -> list:
    """Split long messages into chunks that fit Telegram's limits."""
    if len(message) <= max_length:
        return [message]
    
    # For code blocks, use simple split
    if '```' in message:
        return [message[i:i+max_length] for i in range(0, len(message), max_length)]
    
    # For regular text, split by paragraphs
    chunks = []
    current_chunk = ""
    
    paragraphs = message.split('\n\n')
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
