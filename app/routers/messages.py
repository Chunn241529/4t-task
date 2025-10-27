import json
import faiss
from fastapi import APIRouter, Depends, HTTPException
import numpy as np
from sqlalchemy.orm import Session
from typing import List

from app.db import get_db
from app.models import ChatMessage as ModelChatMessage, Conversation as ModelConversation, User
from app.schemas import ChatMessage, ChatMessageUpdate
from app.routers.task import get_current_user
from app.services.embedding_service import EmbeddingService
from app.services.rag_service import RAGService

router = APIRouter(prefix="/messages", tags=["messages"])
embedding_service = EmbeddingService()
rag_service = RAGService()

@router.get("/conversations/{conversation_id}/messages", response_model=List[ChatMessage])
def get_messages(conversation_id: int, user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    conversation = db.query(ModelConversation).filter(ModelConversation.id == conversation_id, ModelConversation.user_id == user_id).first()
    if not conversation:
        raise HTTPException(404, "Conversation not found or not authorized")

    messages = db.query(ModelChatMessage).filter(
        ModelChatMessage.conversation_id == conversation_id,
        ModelChatMessage.user_id == user_id
    ).order_by(ModelChatMessage.timestamp.asc()).all()

    result = []
    for msg in messages:
        msg_dict = msg.__dict__
        if msg.embedding and isinstance(msg.embedding, str):
            try:
                parsed_embedding = json.loads(msg.embedding)
                msg_dict['embedding'] = parsed_embedding
            except json.JSONDecodeError:
                msg_dict['embedding'] = None
        result.append(ChatMessage(**msg_dict))

    return result

@router.get("/{id}", response_model=ChatMessage)
def get_message(id: int, user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    message = db.query(ModelChatMessage).filter(ModelChatMessage.id == id, ModelChatMessage.user_id == user_id).first()
    if not message:
        raise HTTPException(404, "Message not found or not authorized")

    msg_dict = message.__dict__
    if msg_dict['embedding'] and isinstance(msg_dict['embedding'], str):
        try:
            parsed_embedding = json.loads(msg_dict['embedding'])
            msg_dict['embedding'] = parsed_embedding
        except json.JSONDecodeError:
            msg_dict['embedding'] = None
    return ChatMessage(**msg_dict)

@router.put("/{id}", response_model=ChatMessage)
def update_message(id: int, msg_update: ChatMessageUpdate, user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    message = db.query(ModelChatMessage).filter(ModelChatMessage.id == id, ModelChatMessage.user_id == user_id).first()
    if not message:
        raise HTTPException(404, "Message not found or not authorized")

    if msg_update.content is not None:
        message.content = msg_update.content
        new_embedding = embedding_service.get_embedding(msg_update.content, max_length=1024)
        message.embedding = json.dumps(new_embedding.tolist())

        index, _ = rag_service.load_faiss(user_id, message.conversation_id)
        all_messages = db.query(ModelChatMessage).filter(ModelChatMessage.conversation_id == message.conversation_id).all()
        embs = np.array([json.loads(m.embedding) for m in all_messages if m.embedding])

        index.reset()
        if len(embs) > 0:
            index.add(embs)
        faiss.write_index(index, rag_service.get_faiss_path(user_id, message.conversation_id))

    db.commit()
    db.refresh(message)
    msg_dict = message.__dict__
    if msg_dict['embedding'] and isinstance(msg_dict['embedding'], str):
        try:
            parsed_embedding = json.loads(msg_dict['embedding'])
            msg_dict['embedding'] = parsed_embedding
        except json.JSONDecodeError:
            msg_dict['embedding'] = None
    return ChatMessage(**msg_dict)

@router.delete("/{id}")
def delete_message(id: int, user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    message = db.query(ModelChatMessage).filter(ModelChatMessage.id == id, ModelChatMessage.user_id == user_id).first()
    if not message:
        raise HTTPException(404, "Message not found or not authorized")

    db.delete(message)
    db.commit()

    index, _ = rag_service.load_faiss(user_id, message.conversation_id)
    embs = np.array([json.loads(m.embedding) for m in db.query(ModelChatMessage).filter(ModelChatMessage.conversation_id == message.conversation_id).all() if m.embedding])

    index.reset()
    if len(embs) > 0:
        index.add(embs)
    faiss.write_index(index, rag_service.get_faiss_path(user_id, message.conversation_id))

    return {"message": "Message deleted"}
