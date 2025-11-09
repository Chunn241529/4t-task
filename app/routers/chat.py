from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Body
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.db import get_db
from app.models import ChatMessage as ModelChatMessage, Conversation as ModelConversation, User
from app.schemas import ChatMessageIn
from app.routers.task import get_current_user
from app.services.chat_service import ChatService
from app.services.file_service import FileService
import logging
from datetime import datetime
from typing import Optional, Union

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/send", response_class=StreamingResponse)
def chat(
    message: ChatMessageIn = Body(...),
    file: Optional[Union[UploadFile, str]] = Body(None),
    conversation_id: Optional[int] = None,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Endpoint chính cho chat với RAG và file processing"""
    return ChatService.chat_with_rag(
        message=message,
        file=file,
        conversation_id=conversation_id,
        user_id=user_id,
        db=db
    )
