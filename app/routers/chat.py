# ========== TỐI ƯU HOÁN CHỈNH: chat.py ==========
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, File, UploadFile, Body
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import json
import base64
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
import os
import ollama
from ollama import web_search, web_fetch
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
from rank_bm25 import BM25Okapi
import PyPDF2
from docx import Document
import io

from app.db import get_db
from app.models import ChatMessage as ModelChatMessage, Conversation as ModelConversation, User
from app.schemas import ChatMessage, ConversationCreate, ChatMessageIn
from app.routers.task import get_current_user
from app.services.rag_service import RAGService
from app.services.file_service import FileService
from app.services.chat_service import ChatService

router = APIRouter(prefix="/send", tags=["chat"])
rag_service = RAGService()
file_service = FileService()
chat_service = ChatService()

logger = logging.getLogger(__name__)

DIM = 768
executor = ThreadPoolExecutor(max_workers=4)
RAG_FILES_DIR = "rag_files"
os.makedirs(RAG_FILES_DIR, exist_ok=True)


def extract_text_from_file(file_content: Union[bytes, str]) -> str:
    if isinstance(file_content, str):
        try:
            file_content = base64.b64decode(file_content)
        except:
            return file_content

    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = '\n'.join(page.extract_text() or "" for page in reader.pages)
        return text[:20000] if text.strip() else ""
    except:
        pass
    
    try:
        return file_content.decode('utf-8', errors='replace')[:20000]
    except:
        pass
    
    return ""

def process_file_for_rag(file_content: bytes, user_id: int, conversation_id: int, filename: str = "") -> str:
    try:
        file_text = extract_text_from_file(file_content)
        if not file_text.strip():
            return ""
        return f"[File: {filename}] Loaded {len(file_text)} characters"
    except Exception as e:
        logger.error(f"Error processing file for RAG: {e}")
        return ""



def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            for break_char in ['\n===', '\n', '. ', '! ', '? ', '。', '!', '?', '; ']:
                pos = text.rfind(break_char, start, end)
                if pos != -1 and pos > start + chunk_size // 2:
                    end = pos + len(break_char)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break
    return chunks


# ========== API CHAT – TỐI ƯU FAISS ==========

@router.post("", response_class=StreamingResponse)
def chat(
    message: ChatMessageIn = Body(...),
    file: Optional[Union[UploadFile, str]] = Body(None),
    conversation_id: Optional[int] = None,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    gender = user.gender
    xung_ho = "anh" if gender == "male" else "chị" if gender == "female" else "bạn"
    current_time = datetime.now().strftime("%Y-%m-%d %I:%M %p %z")

    system_prompt = chat_service.build_system_prompt(gender, current_time)

    # Tìm/tạo conversation
    conversation = None
    if conversation_id is not None:
        conversation = db.query(ModelConversation).filter(
            ModelConversation.id == conversation_id,
            ModelConversation.user_id == user_id
        ).first()
        if not conversation:
            raise HTTPException(404, "Conversation not found or not authorized")
    else:
        conversation = ModelConversation(user_id=user_id, created_at=datetime.utcnow())
        db.add(conversation)
        db.flush()

    # Xử lý file
    is_image = False
    images = None
    effective_query = message.message
    file_rag_context = ""

    if file:
        file_bytes = None
        filename = ""
        if isinstance(file, UploadFile):
            file_bytes = file.read()
            filename = file.filename or ""
        elif isinstance(file, dict):
            # client may send {"content": base64, "filename": name}
            content = file.get("content")
            filename = file.get("filename", "uploaded_file")
            try:
                file_bytes = base64.b64decode(content)
            except Exception:
                file_bytes = b""
        else:
            # legacy: client sent data-uri or raw base64 string
            try:
                file_bytes = base64.b64decode(file.split(',')[1]) if ',' in file else base64.b64decode(file)
            except Exception:
                try:
                    file_bytes = base64.b64decode(file)
                except Exception:
                    file_bytes = b""
            filename = "uploaded_file"

        is_image = filename and any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp'])
        if is_image:
            images = [base64.b64encode(file_bytes).decode('utf-8')]
        else:
            file_rag_context = process_file_for_rag(file_bytes, user_id, conversation.id, filename)
            file_content = extract_text_from_file(file_bytes)
            effective_query = f"{message.message}\nFile content: {file_content}"

    # Chọn model
    eval_result = chat_service.evaluate_user_input(effective_query)
    if is_image:
        model_name = "qwen3-vl:235b-cloud"
        tools = None
    elif eval_result["needs_logic"]:
        model_name = "4T-Logic"
        tools = [web_search, web_fetch]
    elif eval_result["needs_reasoning"]:
        model_name = "4T-Reasoning"
        tools = [web_search, web_fetch]
    else:
        model_name = "4T"
        tools = [web_search, web_fetch]

    # Delegate context preparation and model selection to ChatService
    # Lấy history (service will use it to search RAG if needed)
    history_messages = db.query(ModelChatMessage).filter(
        ModelChatMessage.user_id == user_id,
        ModelChatMessage.conversation_id == conversation.id
    ).order_by(ModelChatMessage.timestamp.asc()).limit(50).all()

    final_context = chat_service.prepare_chat_context(
        effective_query, file_rag_context, user_id, conversation.id, history_messages
    )

    full_prompt = f"Context: {final_context}\nUser: {effective_query}" if not is_image else effective_query

    # Ask service to choose model and tools
    model_name, tools, think_level = chat_service.select_model(eval_result, is_image)

    def generate_stream():
        # Immediate response so client can react quickly
        start_time = datetime.utcnow()
        yield f"data: {json.dumps({'conversation_id': conversation.id})}\n\n"
        # Send a typing indicator immediately so frontend can show feedback
        yield f"data: {json.dumps({'typing': True})}\n\n"

        q: "queue.Queue" = queue.Queue()

        def bg_worker():
            """Background worker: runs model stream and pushes chunks into the queue."""
            full_response_parts = []
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ]
                if is_image:
                    messages[-1]["images"] = images

                stream = ollama.chat(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    stream=True,
                    options={"temperature": 0.8},
                    think=think_level
                )

                for chunk in stream:
                    if chunk.get("message", {}).get("content"):
                        delta = chunk["message"]["content"]
                        full_response_parts.append(delta)
                        try:
                            q.put({'type': 'chunk', 'content': delta}, block=True, timeout=5)
                        except Exception:
                            logger.exception("Failed to put chunk into queue")

                # signal done
                q.put({'type': 'done'})

                # Offload saving embeddings and FAISS update so they don't block streaming
                def save_and_update(full_resp_str: str):
                    try:
                        user_emb = chat_service.embedding_service.get_embedding(effective_query, max_length=1024)
                        ass_emb = chat_service.embedding_service.get_embedding(full_resp_str, max_length=1024)

                        user_msg = ModelChatMessage(
                            user_id=user_id, conversation_id=conversation.id,
                            content=effective_query, role="user",
                            embedding=json.dumps(user_emb.tolist())
                        )
                        ass_msg = ModelChatMessage(
                            user_id=user_id, conversation_id=conversation.id,
                            content=full_resp_str, role="assistant",
                            embedding=json.dumps(ass_emb.tolist())
                        )
                        try:
                            db.add_all([user_msg, ass_msg])
                            db.commit()
                        except Exception:
                            logger.exception("DB commit failed for chat messages")

                        try:
                            idx, _ = rag_service.load_faiss(user_id, conversation.id)
                            rag_service.safe_add_to_faiss(
                                idx,
                                [user_emb.tolist(), ass_emb.tolist()],
                                user_id,
                                conversation.id
                            )
                        except Exception:
                            logger.exception("FAISS update failed")
                    except Exception:
                        logger.exception("Error in save_and_update")

                full_resp_str = ''.join(full_response_parts)
                # run save in executor to avoid blocking
                try:
                    chat_service.executor.submit(save_and_update, full_resp_str)
                except Exception:
                    logger.exception("Failed to submit save task to executor")

            except Exception as e:
                logger.exception(f"Error in bg_worker: {e}")
                try:
                    q.put({'type': 'error', 'error': str(e)})
                except Exception:
                    pass

        # Start background worker thread using executor so it runs concurrently
        try:
            chat_service.executor.submit(bg_worker)
        except Exception:
            # fallback: start a thread
            t = threading.Thread(target=bg_worker, daemon=True)
            t.start()

        # Consume queue and yield chunks as they arrive. Measure time-to-first-byte.
        first_chunk_sent = False
        while True:
            try:
                item = q.get()
            except Exception:
                break

            if not isinstance(item, dict):
                continue

            if item.get('type') == 'chunk':
                # On first chunk, log TTFB metric and clear typing indicator
                if not first_chunk_sent:
                    first_chunk_time = datetime.utcnow()
                    latency_ms = (first_chunk_time - start_time).total_seconds() * 1000
                    logger.info(f"TTFB for conversation {conversation.id}: {latency_ms:.1f} ms")
                    # send typing=false before the first real content
                    yield f"data: {json.dumps({'typing': False})}\n\n"
                    first_chunk_sent = True

                yield f"data: {json.dumps({'content': item['content']})}\n\n"
            elif item.get('type') == 'error':
                yield f"data: {json.dumps({'error': item.get('error')})}\n\n"
                break
            elif item.get('type') == 'done':
                # If no chunks were sent, still log that we finished
                if not first_chunk_sent:
                    first_chunk_time = datetime.utcnow()
                    latency_ms = (first_chunk_time - start_time).total_seconds() * 1000
                    logger.info(f"No content produced; TTFB for conversation {conversation.id}: {latency_ms:.1f} ms")
                    # end typing indicator
                    yield f"data: {json.dumps({'typing': False})}\n\n"

                yield f"data: {json.dumps({'done': True})}\n\n"
                total_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                logger.info(f"Total stream time for conversation {conversation.id}: {total_time_ms:.1f} ms")
                break

    return StreamingResponse(generate_stream(), media_type="text/event-stream")

# ========== WEBSOCKET VÀ CÁC API KHÁC GIỮ NGUYÊN ==========

@router.websocket("/ws/{user_id}/{conversation_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int, conversation_id: int, db: Session = Depends(get_db)):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            if message_type == "chat":
                await handle_chat_message(websocket, data, user_id, conversation_id, db)
            elif message_type == "file":
                await handle_file_upload(websocket, data, user_id, conversation_id, db)
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"error": str(e)})

async def handle_chat_message(websocket: WebSocket, data: dict, user_id: int, conversation_id: int, db: Session):
    """Xử lý tin nhắn chat qua WebSocket"""
    try:
        content = data.get("content", "")

        conversation = db.query(ModelConversation).filter(
            ModelConversation.id == conversation_id,
            ModelConversation.user_id == user_id
        ).first()

        if not conversation:
            await websocket.send_json({"error": "Conversation not found"})
            return

        # Immediate ack + typing indicator
        await websocket.send_json({"type": "ack", "conversation_id": conversation_id, "timestamp": datetime.utcnow().isoformat()})
        await websocket.send_json({"type": "typing", "status": True})

        # Prepare history and prompt; we'll let background worker call the model
        history_messages = db.query(ModelChatMessage).filter(
            ModelChatMessage.conversation_id == conversation_id
        ).order_by(ModelChatMessage.timestamp.asc()).all()

        # Build a lightweight system prompt
        system_prompt = "Bạn là trợ lý AI hữu ích."
        full_prompt_messages = [
            {"role": "system", "content": system_prompt},
            *[{"role": msg.role, "content": msg.content} for msg in history_messages[-10:]]
        ]

        q = queue.Queue()
        start_time = datetime.utcnow()

        def bg_worker():
            """Run the model stream in a background thread and push chunks to the queue."""
            try:
                # Evaluate and choose model
                eval_result = chat_service.evaluate_user_input(content)
                model_name, tools, think_level = chat_service.select_model(eval_result, False)

                # Compute user embedding and persist user message (non-blocking to caller)
                try:
                    user_emb = chat_service.embedding_service.get_embedding(content, max_length=1024)
                    user_msg = ModelChatMessage(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        content=content,
                        role="user",
                        timestamp=datetime.utcnow(),
                        embedding=json.dumps(user_emb.tolist())
                    )
                    db.add(user_msg)
                    db.commit()
                except Exception:
                    logger.exception("Failed to save user message in bg_worker")

                # Call the model
                try:
                    stream = ollama.chat(
                        model=model_name,
                        messages=full_prompt_messages,
                        tools=tools,
                        stream=True,
                        think=think_level
                    )

                    full_response_parts = []
                    for chunk in stream:
                        if chunk.get("message", {}).get("content"):
                            delta = chunk["message"]["content"]
                            full_response_parts.append(delta)
                            try:
                                q.put({'type': 'chunk', 'content': delta}, block=True, timeout=5)
                            except Exception:
                                logger.exception("Failed to put websocket chunk into queue")

                    # signal done
                    q.put({'type': 'done'})

                    # Offload assistant save + FAISS update
                    def save_assistant_and_faiss(full_resp_str: str):
                        try:
                            ass_emb = chat_service.embedding_service.get_embedding(full_resp_str, max_length=1024)
                            ass_msg = ModelChatMessage(
                                conversation_id=conversation_id,
                                user_id=user_id,
                                content=full_resp_str,
                                role="assistant",
                                timestamp=datetime.utcnow(),
                                embedding=json.dumps(ass_emb.tolist())
                            )
                            try:
                                db.add(ass_msg)
                                db.commit()
                            except Exception:
                                logger.exception("Failed to commit assistant message")

                            try:
                                idx, _ = rag_service.load_faiss(user_id, conversation_id)
                                rag_service.safe_add_to_faiss(
                                    idx,
                                    [ass_emb.tolist()],
                                    user_id,
                                    conversation_id
                                )
                            except Exception:
                                logger.exception("Failed to update FAISS from websocket bg task")
                        except Exception:
                            logger.exception("Error saving assistant message in websocket bg task")

                    full_resp_str = ''.join(full_response_parts)
                    try:
                        chat_service.executor.submit(save_assistant_and_faiss, full_resp_str)
                    except Exception:
                        logger.exception("Failed to submit assistant save task")

                except Exception as e:
                    logger.exception(f"WebSocket model stream error: {e}")
                    try:
                        q.put({'type': 'error', 'error': str(e)})
                    except Exception:
                        pass

            except Exception as e:
                logger.exception(f"Unexpected error in websocket bg_worker: {e}")
                try:
                    q.put({'type': 'error', 'error': str(e)})
                except Exception:
                    pass

        # start background worker
        try:
            chat_service.executor.submit(bg_worker)
        except Exception:
            t = threading.Thread(target=bg_worker, daemon=True)
            t.start()

        # consume queue and forward to websocket
        first_chunk_sent = False
        loop = __import__('asyncio').get_event_loop()
        while True:
            try:
                item = await loop.run_in_executor(None, q.get)
            except Exception:
                break

            if not isinstance(item, dict):
                continue

            if item.get('type') == 'chunk':
                if not first_chunk_sent:
                    first_chunk_time = datetime.utcnow()
                    latency_ms = (first_chunk_time - start_time).total_seconds() * 1000
                    logger.info(f"WebSocket TTFB for conv {conversation_id}: {latency_ms:.1f} ms")
                    # clear typing indicator
                    await websocket.send_json({"type": "typing", "status": False})
                    first_chunk_sent = True

                await websocket.send_json({"type": "chunk", "content": item['content']})

            elif item.get('type') == 'error':
                await websocket.send_json({"type": "error", "error": item.get('error')})
                break

            elif item.get('type') == 'done':
                if not first_chunk_sent:
                    first_chunk_time = datetime.utcnow()
                    latency_ms = (first_chunk_time - start_time).total_seconds() * 1000
                    logger.info(f"WebSocket no content; TTFB for conv {conversation_id}: {latency_ms:.1f} ms")
                    await websocket.send_json({"type": "typing", "status": False})

                await websocket.send_json({"type": "complete"})
                total_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                logger.info(f"WebSocket total stream time for conv {conversation_id}: {total_time_ms:.1f} ms")
                break

    except Exception as e:
        logger.error(f"Error handling chat message: {e}")
        await websocket.send_json({"error": str(e)})

async def handle_file_upload(websocket: WebSocket, data: dict, user_id: int, conversation_id: int, db: Session):
    """Xử lý upload file qua WebSocket"""
    try:
        file_data = data.get("file_data", "")
        filename = data.get("filename", "")
        
        if not file_data:
            await websocket.send_json({"error": "No file data provided"})
            return

        file_content = base64.b64decode(file_data)
        rag_context = process_file_for_rag(file_content, user_id, conversation_id, filename)
        
        await websocket.send_json({
            "type": "file_processed",
            "filename": filename,
            "rag_context": rag_context
        })
        
    except Exception as e:
        await websocket.send_json({"error": f"File processing error: {str(e)}"})
        
        

