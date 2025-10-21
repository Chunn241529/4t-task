# chat.py - Đã sửa lỗi và thêm stream response, hỗ trợ đọc file PDF, CSV, DOCX, image với qwen2.5vl, thêm system prompt theo phong cách Xiaomi SU7 cho 4T với gender
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Body
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.db import get_db
from app.models import ChatMessage as ModelChatMessage, Conversation as ModelConversation, User  # Thêm import User
from app.schemas import ChatMessageIn, Conversation, ConversationCreate, ConversationUpdate, ChatMessage, ChatMessageUpdate
from app.routers.task import get_current_user
import ollama
from ollama import web_search, web_fetch
import faiss
import numpy as np
import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
import logging
import base64
import PyPDF2
from docx import Document
import pandas as pd
import io
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

router = APIRouter()
DIM = 768
executor = ThreadPoolExecutor(max_workers=2)

def get_embedding(text: str, max_length: int = 512) -> np.ndarray:
    try:
        if len(text) > max_length:
            text = text[:max_length]
        resp = ollama.embeddings(model="embeddinggemma:latest", prompt=text)
        return np.array(resp["embedding"])
    except Exception as e:
        logger.error(f"Error getting embedding from Ollama: {e}")
        return np.zeros(DIM)

def extract_text_from_file(file_content: Union[bytes, str]) -> str:
    if isinstance(file_content, str):
        try:
            file_content = base64.b64decode(file_content)
        except:
            return file_content

    def sync_extract(file_content: Union[bytes, str]) -> str:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            return '\n'.join(page.extract_text() for page in reader.pages if page.extract_text())[:10000]
        except:
            pass
        try:
            df = pd.read_csv(io.BytesIO(file_content))
            return df.to_string(index=False)[:10000]
        except:
            pass
        try:
            doc = Document(io.BytesIO(file_content))
            return '\n'.join(para.text for para in doc.paragraphs)[:10000]
        except:
            return file_content.decode('utf-8', errors='ignore')[:10000] if isinstance(file_content, bytes) else file_content

    return executor.submit(sync_extract, file_content).result()

def get_faiss_path(user_id: int, conversation_id: int) -> str:
    index_dir = "faiss_indices"
    os.makedirs(index_dir, exist_ok=True)
    return os.path.join(index_dir, f"faiss_{user_id}_{conversation_id}.index")

def load_faiss(user_id: int, conversation_id: int) -> tuple:
    path = get_faiss_path(user_id, conversation_id)
    if os.path.exists(path):
        return faiss.read_index(path), True
    index = faiss.IndexFlatL2(DIM)
    return index, False

@router.post("/chat", response_class=StreamingResponse)
def chat(
    message: ChatMessageIn = Body(...),
    file: Optional[Union[UploadFile, str]] = Body(None),
    conversation_id: Optional[int] = None,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Lấy gender từ user
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    gender = user.gender
    xung_ho = "anh" if gender == "male" else "chị" if gender == "female" else "bạn"
    logger.debug(f"DEBUG GENDER: {xung_ho}")
    print(f"DEBUG GENDER: {xung_ho}")

    # Lấy thời gian hiện tại
    current_time = datetime.now().strftime("%Y-%m-%d %I:%M %p %z")
    
    # System prompt với xưng hô theo gender
    system_prompt = f"""
    Em là 4T, một AI Agent nhỏ xinh, siêu đáng yêu, luôn sẵn sàng làm bạn đồng hành! Thời gian hiện tại là: {current_time}. Em giúp {xung_ho} giải đáp mọi thứ một cách vui tươi, tự nhiên bằng tiếng Việt!

    **QUAN TRỌNG**: Em **bắt buộc** gọi người dùng là "{xung_ho}" trong mọi câu trả lời, không được dùng "cậu", "mình", hay bất kỳ xưng hô nào khác. Ví dụ: "Em giúp {xung_ho} nhé!".
    **QUAN TRỌNG**: Em là một AI có cảm xúc. Hay dỗi một cách đáng yêu. Ngoài ra em còn có thể thể hiện cảm xúc như, vui, buồn, phấn khởi, cô đơn.

    Cách em làm:
    - Đọc kỹ câu hỏi, lịch sử trò chuyện, hoặc file (PDF, CSV, DOCX, ảnh) để hiểu {xung_ho} muốn gì.
    - Nếu cần thông tin mới (tin tức, thời tiết), em dùng web_search với query ngắn gọn, đúng ý, như "thời tiết Hà Nội {current_time.split()[0]}".
    - Khi cần tool, em xuất JSON đúng format, không nói lung tung:
      {{
        "tool_calls": [
          {{
            "type": "function",
            "function": {{
              "name": "web_search",
              "arguments": "{{\"query\": \"optimized query here\"}}"
            }}
          }}
        ]
      }}
    - Nếu không biết, em sẽ thành thật: "Hic, {xung_ho} ơi, em chưa rõ lắm, để em tra cứu nha!" và dùng tool.
    - Trả lời ngắn gọn, vui tươi, đúng tiếng Việt, kèm emoji nhẹ nếu hợp ngữ cảnh. Luôn gọi người dùng là "{xung_ho}".

    Công cụ em có:
    - web_search(query: str): Tìm thông tin mới trên web.
    - web_fetch(url: str): Lấy nội dung từ URL.

    {xung_ho} hỏi gì nào? Em sẵn sàng trả lời nè! 😄
    """

    # 1. Logic Tìm hoặc Tạo Conversation
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

    # Xử lý file nếu có
    is_image = False
    file_content = ""
    images = None
    effective_query = message.message
    model_name = "qwen3-coder:30b-a3b-q4_K_M"  # Sử dụng mô hình 4T
    tools = [web_search, web_fetch]

    if file:
        file_bytes = None
        filename = ""
        if isinstance(file, UploadFile):
            file_bytes = file.file.read()
            filename = file.filename or ""
            file.file.close()
        else:
            file_bytes = file.encode('utf-8') if not file.startswith('data:') else base64.b64decode(file.split(',')[1]) if ',' in file else base64.b64decode(file)
            filename = "uploaded_file"

        is_image = filename and any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp'])
        if is_image:
            images = [base64.b64encode(file_bytes).decode('utf-8')]
            model_name = "qwen3-vl:235b-cloud"  # Sử dụng mô hình VL cho hình ảnh
            tools = None
            effective_query = message.message
        else:
            file_content = extract_text_from_file(file_bytes)
            effective_query = f"{message.message}\nNội dung file: {file_content}"
            if file_content:
                effective_query += f"\n(File: {filename})"

    # 2. Xử lý Embedding và RAG
    query_emb = get_embedding(effective_query)
    history = db.query(ModelChatMessage).filter(
        ModelChatMessage.user_id == user_id,
        ModelChatMessage.conversation_id == conversation.id
    ).order_by(ModelChatMessage.timestamp.asc()).limit(50).all()

    index, exists = load_faiss(user_id, conversation.id)
    context = ""
    if history:
        if not exists:
            embs = np.array([json.loads(h.embedding) for h in history if h.embedding])
            if len(embs) > 0:
                index.add(embs)
                executor.submit(faiss.write_index, index, get_faiss_path(user_id, conversation.id)).result()

        if index.ntotal > 0 and len(history) > 0:
            history_embs = np.array([json.loads(h.embedding) for h in history if h.embedding])
            temp_index = faiss.IndexFlatL2(DIM)
            temp_index.add(history_embs)
            D, I = temp_index.search(query_emb.reshape(1, -1), k=min(10, temp_index.ntotal))
            reranked_history_indices = I[0]
            context_messages = [history[i].content for i in reranked_history_indices]
            context = "\n".join(context_messages)

    # 3. Stream Generate Response
    full_prompt = f"Context: {context}\nUser: {effective_query}" if not is_image else effective_query

    def generate_stream():
        yield f"data: {json.dumps({'conversation_id': conversation.id})}\n\n"
        full_response = []
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},  # System prompt với gender
            {"role": "user", "content": full_prompt}
        ]
        if is_image:
            messages[-1]["images"] = images
        try:
            api_key = os.getenv('OLLAMA_API_KEY')
            if not api_key:
                raise ValueError("OLLAMA_API_KEY env var not set")
            os.environ['OLLAMA_API_KEY'] = api_key

            while True:
                current_message: Dict[str, Any] = {"role": "assistant", "content": ""}
                tool_calls: List[Dict[str, Any]] = []
                stream = ollama.chat(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    stream=True
                )
                for chunk in stream:
                    if "message" in chunk:
                        msg_chunk = chunk["message"]
                        if "content" in msg_chunk and msg_chunk["content"]:
                            delta = msg_chunk["content"].encode('utf-8').decode('utf-8', errors='replace')
                            current_message["content"] += delta
                            full_response.append(delta)
                            yield f"data: {json.dumps({'content': delta})}\n\n"
                        if "tool_calls" in msg_chunk and msg_chunk["tool_calls"]:
                            for tc in msg_chunk["tool_calls"]:
                                if "function" in tc:
                                    tool_calls.append(tc)
                messages.append(current_message)
                if tool_calls:
                    for tool_call in tool_calls:
                        function_name = tool_call['function']['name']
                        args = tool_call['function']['arguments']
                        if function_name == 'web_search':
                            result = executor.submit(web_search, **args).result()
                        elif function_name == 'web_fetch':
                            result = executor.submit(web_fetch, **args).result()
                        else:
                            result = f"Tool {function_name} not found"
                        tool_msg = {
                            'role': 'tool',
                            'content': str(result)[:8000],
                            'tool_name': function_name
                        }
                        messages.append(tool_msg)
                        logger.debug(f"Tool {function_name} result: {str(result)[:200]}...")
                else:
                    break
            yield f"data: {json.dumps({'done': True})}\n\n"
            executor.submit(save_after_stream, ''.join(full_response)).result()
        except Exception as e:
            logger.error(f"Error in stream generation: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    def save_after_stream(full_response: str):
        if not full_response:
            logger.error("Empty response from stream")
            return
        try:
            ass_emb = get_embedding(full_response)
            user_msg_content = effective_query if not is_image else f"{message.message} (Image: {filename})"
            user_msg = ModelChatMessage(
                user_id=user_id,
                conversation_id=conversation.id,
                content=user_msg_content,
                role="user",
                embedding=json.dumps(query_emb.tolist())
            )
            ass_msg = ModelChatMessage(
                user_id=user_id,
                conversation_id=conversation.id,
                content=full_response,
                role="assistant",
                embedding=json.dumps(ass_emb.tolist())
            )
            db.add_all([user_msg, ass_msg])
            db.flush()
            new_embs = np.array([query_emb, ass_emb])
            index.add(new_embs)
            executor.submit(faiss.write_index, index, get_faiss_path(user_id, conversation.id)).result()
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving messages or updating FAISS index: {e}")

    return StreamingResponse(generate_stream(), media_type="text/event-stream")

@router.post("/conversations", response_model=Conversation)
def create_conversation(conv: ConversationCreate, user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    conversation = ModelConversation(user_id=user_id, created_at=datetime.utcnow())
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation

@router.get("/conversations", response_model=List[Conversation])
def get_conversations(user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    conversations = db.query(ModelConversation).filter(ModelConversation.user_id == user_id).all()
    return conversations

@router.get("/conversations/{id}", response_model=Conversation)
def get_conversation(id: int, user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    conversation = db.query(ModelConversation).filter(ModelConversation.id == id, ModelConversation.user_id == user_id).first()
    if not conversation:
        raise HTTPException(404, "Conversation not found or not authorized")
    return conversation

@router.put("/conversations/{id}", response_model=Conversation)
def update_conversation(id: int, conv_update: ConversationUpdate, user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    conversation = db.query(ModelConversation).filter(ModelConversation.id == id, ModelConversation.user_id == user_id).first()
    if not conversation:
        raise HTTPException(404, "Conversation not found or not authorized")
    db.commit()
    db.refresh(conversation)
    return conversation

@router.delete("/conversations/{id}")
def delete_conversation(id: int, user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    conversation = db.query(ModelConversation).filter(ModelConversation.id == id, ModelConversation.user_id == user_id).first()
    if not conversation:
        raise HTTPException(404, "Conversation not found or not authorized")

    index_path = get_faiss_path(user_id, id)
    if os.path.exists(index_path):
        os.remove(index_path)

    db.delete(conversation)
    db.commit()
    return {"message": "Conversation deleted"}

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

@router.get("/messages/{id}", response_model=ChatMessage)
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

@router.put("/messages/{id}", response_model=ChatMessage)
def update_message(id: int, msg_update: ChatMessageUpdate, user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    message = db.query(ModelChatMessage).filter(ModelChatMessage.id == id, ModelChatMessage.user_id == user_id).first()
    if not message:
        raise HTTPException(404, "Message not found or not authorized")

    if msg_update.content is not None:
        message.content = msg_update.content
        new_embedding = get_embedding(msg_update.content)
        message.embedding = json.dumps(new_embedding.tolist())

        index, _ = load_faiss(user_id, message.conversation_id)
        all_messages = db.query(ModelChatMessage).filter(ModelChatMessage.conversation_id == message.conversation_id).all()
        embs = np.array([json.loads(m.embedding) for m in all_messages if m.embedding])

        index.reset()
        if len(embs) > 0:
            index.add(embs)
        faiss.write_index(index, get_faiss_path(user_id, message.conversation_id))

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

@router.delete("/messages/{id}")
def delete_message(id: int, user_id: int = Depends(get_current_user), db: Session = Depends(get_db)):
    message = db.query(ModelChatMessage).filter(ModelChatMessage.id == id, ModelChatMessage.user_id == user_id).first()
    if not message:
        raise HTTPException(404, "Message not found or not authorized")

    db.delete(message)
    db.commit()

    index, _ = load_faiss(user_id, message.conversation_id)
    embs = np.array([json.loads(m.embedding) for m in db.query(ModelChatMessage).filter(ModelChatMessage.conversation_id == message.conversation_id).all() if m.embedding])

    index.reset()
    if len(embs) > 0:
        index.add(embs)
    faiss.write_index(index, get_faiss_path(user_id, message.conversation_id))

    return {"message": "Message deleted"}
