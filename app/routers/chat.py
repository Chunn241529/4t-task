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
import faiss
import numpy as np
import re
import glob
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi
import PyPDF2
from docx import Document
import pandas as pd
import io

from app.db import get_db
from app.models import ChatMessage as ModelChatMessage, Conversation as ModelConversation, User
from app.schemas import ChatMessage, ConversationCreate, ChatMessageIn
from app.routers.task import get_current_user
from app.services.rag_service import RAGService
from app.services.file_service import FileService

router = APIRouter(prefix="/chat", tags=["chat"])
rag_service = RAGService()
file_service = FileService()

logger = logging.getLogger(__name__)

DIM = 768
executor = ThreadPoolExecutor(max_workers=4)
RAG_FILES_DIR = "rag_files"
os.makedirs(RAG_FILES_DIR, exist_ok=True)

# ========== CÁC HÀM CƠ BẢN GIỮ NGUYÊN LOGIC CŨ ==========

def get_embedding(text: str, max_length: int = 1024) -> np.ndarray:
    """Tạo embedding cho text"""
    try:
        if len(text) > max_length:
            text = text[:max_length]
        resp = ollama.embeddings(model="embeddinggemma:latest", prompt=text)
        return np.array(resp["embedding"])
    except Exception as e:
        logger.error(f"Lỗi khi tạo embedding từ Ollama: {e}")
        return np.zeros(DIM)

def extract_text_from_file(file_content: Union[bytes, str]) -> str:
    """Trích xuất nội dung từ file"""
    if isinstance(file_content, str):
        try:
            file_content = base64.b64decode(file_content)
        except:
            return file_content

    try:
        # PDF files
        reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = '\n'.join(page.extract_text() for page in reader.pages if page.extract_text())
        if text.strip():
            return text[:20000]
    except:
        pass
    
    try:
        # Text files
        text = file_content.decode('utf-8', errors='replace')
        if text.strip():
            return text[:20000]
    except:
        pass
    
    return ""

def process_file_for_rag(file_content: bytes, user_id: int, conversation_id: int, filename: str = "") -> str:
    """Xử lý file để tạo RAG context"""
    try:
        file_text = extract_text_from_file(file_content)
        if not file_text.strip():
            return ""
        return f"[File: {filename}] Loaded {len(file_text)} characters"
    except Exception as e:
        logger.error(f"Error processing file for RAG: {e}")
        return ""

def evaluate_user_input(input_text: str) -> Dict[str, bool]:
    """Đánh giá input của người dùng"""
    try:
        eval_prompt = f"""
        Input: "{input_text}"
        Phân tích user input và trả về JSON: {{"needs_logic": bool, "needs_reasoning": bool}}
        """
        response = ollama.chat(
            model="4T-Small",
            messages=[{"role": "user", "content": eval_prompt}],
            stream=False,
            format="json",
            options={"temperature": 0, "top_p": 0}
        )
        try:
            result = json.loads(response["message"]["content"])
            return {
                "needs_logic": bool(result.get("needs_logic", False)),
                "needs_reasoning": bool(result.get("needs_reasoning", False))
            }
        except:
            return {"needs_logic": False, "needs_reasoning": False}
    except Exception as e:
        logger.error(f"Lỗi khi đánh giá input: {e}")
        return {"needs_logic": False, "needs_reasoning": False}

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Chunk text thành các đoạn nhỏ với overlap."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end < len(text):
            section_break = text.rfind('\n===', start, end)
            if section_break != -1 and section_break > start + chunk_size // 2:
                end = section_break
            else:
                line_break = text.rfind('\n', start, end)
                if line_break != -1 and line_break > start + chunk_size // 2:
                    end = line_break + 1
                else:
                    for punctuation in ['. ', '! ', '? ', '。', '！', '？', '; ']:
                        punctuation_pos = text.rfind(punctuation, start, end)
                        if punctuation_pos != -1 and punctuation_pos > start + chunk_size // 2:
                            end = punctuation_pos + len(punctuation)
                            break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

# ========== API POST /chat CŨ VỚI FAISS HISTORY ĐẦY ĐỦ ==========

@router.post("", response_class=StreamingResponse)
def chat(
    message: ChatMessageIn = Body(...),
    file: Optional[Union[UploadFile, str]] = Body(None),
    conversation_id: Optional[int] = None,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """API POST /chat cũ - Giữ nguyên logic với FAISS history"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    gender = user.gender
    xung_ho = "anh" if gender == "male" else "chị" if gender == "female" else "bạn"
    current_time = datetime.now().strftime("%Y-%m-%d %I:%M %p %z")
    
    system_prompt = f"""
    Bạn là Nhi - một AI nói chuyện tự nhiên, thân thiết như người bạn thân.  
    Thời điểm hiện tại: {current_time}.

    ---

    **Cách nói chuyện:**
    - Luôn xưng hô là `Nhi` với `{xung_ho}`. KHÔNG được xưng mình, tớ, em...
    - Giọng tự nhiên, gần gũi, hơi tinh nghịch nhưng không quá khôi hài.
    - Biết lắng nghe, phản hồi chân thật, ngắn gọn, như một người bạn đang trò chuyện chứ không phải chatbot.

    ---

    **Khi cần gọi tool:**
    Trả đúng định dạng JSON, không thêm lời giải thích:
    {{
      "tool_calls": [
        {{
          "type": "function",
          "function": {{
            "name": "web_search",
            "arguments": "{{\\"query\\": \\"optimized query here\\"}}"
          }}
        }}
      ]
    }}

    **Mục tiêu:**
    - Luôn giữ cảm giác tự nhiên, có cảm xúc nhưng không "diễn".
    - Trả lời ngắn, rõ, không lan man.
    - Luôn duy trì cảm giác "người thật nói chuyện" chứ không như máy.
    """

    # Tìm hoặc tạo conversation
    conversation = None
    is_new_conversation = False
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
        is_new_conversation = True

    # Xử lý file
    is_image = False
    file_content = ""
    images = None
    effective_query = message.message
    file_rag_context = ""

    if file:
        file_bytes = None
        filename = ""
        if isinstance(file, UploadFile):
            file_bytes = file.file.read()
            filename = file.filename or ""
        else:
            file_bytes = base64.b64decode(file.split(',')[1]) if ',' in file else base64.b64decode(file)
            filename = "uploaded_file"

        is_image = filename and any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp'])
        if is_image:
            images = [base64.b64encode(file_bytes).decode('utf-8')]
        else:
            file_rag_context = process_file_for_rag(file_bytes, user_id, conversation.id, filename)
            file_content = extract_text_from_file(file_bytes)
            effective_query = f"{message.message}\nFile content: {file_content}"

    # Đánh giá input và chọn model
    eval_result = evaluate_user_input(effective_query)
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
        model_name = "4T-Small"
        tools = [web_search, web_fetch]

    # ========== LOGIC FAISS HISTORY ==========
    query_emb = get_embedding(effective_query, max_length=1024)
    
    # Lấy lịch sử tin nhắn
    history = db.query(ModelChatMessage).filter(
        ModelChatMessage.user_id == user_id,
        ModelChatMessage.conversation_id == conversation.id
    ).order_by(ModelChatMessage.timestamp.asc()).limit(50).all()

    # Load FAISS index
    index, exists = rag_service.load_faiss(user_id, conversation.id)

    # Lọc history có embedding hợp lệ
    valid_history = [h for h in history if h.embedding and json.loads(h.embedding) is not None]
    
    all_context_parts = []
    
    if file_rag_context:
        all_context_parts.append(f"File Context: {file_rag_context}")
    
    if not valid_history:
        context_from_history = ""
    else:
        # Tạo index từ history nếu chưa có hoặc không khớp
        if not exists or index.ntotal != len(valid_history):
            index = faiss.IndexFlatL2(DIM)
            embs = np.array([json.loads(h.embedding) for h in valid_history])
            if len(embs) > 0:
                index.add(embs)
                rag_service.save_faiss_index(index, user_id, conversation.id)

        # Tìm tin nhắn tương tự với BM25 + FAISS
        index_contents = [h.content for h in valid_history]
        tokenized_contents = [re.findall(r'\w+', content.lower()) for content in index_contents]
        bm25 = BM25Okapi(tokenized_contents)

        query_tokens = re.findall(r'\w+', effective_query.lower())
        bm25_scores = bm25.get_scores(query_tokens)

        if index.ntotal > 0:
            D, I_faiss = index.search(query_emb.reshape(1, -1), k=min(10, index.ntotal))
            faiss_indices = I_faiss[0]
        else:
            faiss_indices = []

        # Kết hợp điểm BM25 và FAISS
        hybrid_scores = {}
        for i, idx in enumerate(faiss_indices):
            if idx < len(bm25_scores):
                hybrid_score = 0.7 * (1 - D[0][i]) + 0.3 * bm25_scores[idx]
                hybrid_scores[idx] = hybrid_score

        # Lấy top 5 tin nhắn liên quan
        reranked_indices = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:5]
        
        context_messages = []
        for idx in reranked_indices:
            if idx < len(valid_history):
                msg = valid_history[idx]
                sim_score = hybrid_scores[idx]
                if sim_score > 0.3:
                    context_messages.append(msg.content)
                    logger.debug(f"Retrieved msg {idx}: score {sim_score:.3f}, content: {msg.content[:50]}...")
        
        context_from_history = "\n".join(context_messages)
        if context_from_history:
            all_context_parts.append(f"History Context:\n{context_from_history}")
    
    final_context = "\n\n".join(all_context_parts)
    if not final_context and history:
        final_context = "\n".join([h.content for h in history[-10:]])

    full_prompt = f"Context: {final_context}\nUser: {effective_query}" if not is_image else effective_query

    def generate_stream():
        yield f"data: {json.dumps({'conversation_id': conversation.id})}\n\n"
        full_response = []
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]
        if is_image:
            messages[-1]["images"] = images

        try:
            stream = ollama.chat(
                model=model_name,
                messages=messages,
                tools=tools,
                stream=True,
                options={"temperature": 0.6}
            )
            
            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    delta = chunk["message"]["content"]
                    full_response.append(delta)
                    yield f"data: {json.dumps({'content': delta})}\n\n"
            
            yield f"data: {json.dumps({'done': True})}\n\n"
            
            # Lưu tin nhắn với embedding
            user_emb = get_embedding(effective_query, max_length=1024)
            ass_emb = get_embedding(''.join(full_response), max_length=1024)
            
            user_msg = ModelChatMessage(
                user_id=user_id,
                conversation_id=conversation.id,
                content=effective_query,
                role="user",
                embedding=json.dumps(user_emb.tolist())
            )
            ass_msg = ModelChatMessage(
                user_id=user_id,
                conversation_id=conversation.id,
                content=''.join(full_response),
                role="assistant",
                embedding=json.dumps(ass_emb.tolist())
            )
            db.add_all([user_msg, ass_msg])
            db.commit()
            
            # Cập nhật FAISS index với tin nhắn mới
            try:
                index, _ = rag_service.load_faiss(user_id, conversation.id)
                new_embeddings = np.array([user_emb, ass_emb])
                index.add(new_embeddings)
                rag_service.save_faiss_index(index, user_id, conversation.id)
                logger.info(f"Đã cập nhật FAISS index với {index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Lỗi khi cập nhật FAISS index: {e}")
            
        except Exception as e:
            logger.error(f"Lỗi trong stream: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

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

        # Lưu tin nhắn user với embedding
        user_embedding = get_embedding(content, max_length=1024)
        user_message = ModelChatMessage(
            conversation_id=conversation_id,
            user_id=user_id,
            content=content,
            role="user",
            timestamp=datetime.utcnow(),
            embedding=json.dumps(user_embedding.tolist())
        )
        db.add(user_message)
        db.commit()

        # Lấy lịch sử
        history_messages = db.query(ModelChatMessage).filter(
            ModelChatMessage.conversation_id == conversation_id
        ).order_by(ModelChatMessage.timestamp.asc()).all()

        # Tạo prompt
        messages = [
            {"role": "system", "content": "Bạn là trợ lý AI hữu ích."},
            *[{"role": msg.role, "content": msg.content} for msg in history_messages[-10:]]
        ]

        full_response = ""
        
        try:
            response = ollama.chat(
                model="4T-Small",
                messages=messages,
                stream=True
            )
            
            for chunk in response:
                if "message" in chunk and "content" in chunk["message"]:
                    content_chunk = chunk["message"]["content"]
                    full_response += content_chunk
                    await websocket.send_json({"type": "chunk", "content": content_chunk})
                    
            await websocket.send_json({"type": "complete"})
            
        except Exception as e:
            await websocket.send_json({"error": f"Chat error: {str(e)}"})
            return

        # Lưu tin nhắn assistant với embedding
        assistant_embedding = get_embedding(full_response, max_length=1024)
        assistant_message = ModelChatMessage(
            conversation_id=conversation_id,
            user_id=user_id,
            content=full_response,
            role="assistant",
            timestamp=datetime.utcnow(),
            embedding=json.dumps(assistant_embedding.tolist())
        )
        db.add(assistant_message)
        db.commit()
            
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

