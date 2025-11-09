import base64
from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse
import ollama
from ollama import web_search, web_fetch
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import logging
from sqlalchemy.orm import Session
from concurrent.futures import ThreadPoolExecutor

from app.models import ChatMessage as ModelChatMessage, Conversation as ModelConversation, User
from app.schemas import ChatMessageIn
from app.services.embedding_service import EmbeddingService
from app.services.file_service import FileService
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)
executor = ThreadPoolExecutor(max_workers=4)

class ChatService:
    
    @staticmethod
    def chat_with_rag(
        message: ChatMessageIn,
        file: Optional[Union[UploadFile, str]],
        conversation_id: Optional[int],
        user_id: int,
        db: Session
    ):
        """Xử lý chat chính với RAG integration"""
        
        # Lấy thông tin user và xưng hô
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        gender = user.gender
        xung_ho = "anh" if gender == "male" else "chị" if gender == "female" else "bạn"
        current_time = datetime.now().strftime("%Y-%m-%d %I:%M %p %z")
        
        # System prompt
        system_prompt = ChatService._build_system_prompt(xung_ho, current_time)
        
        # Xử lý conversation
        conversation, is_new_conversation = ChatService._get_or_create_conversation(
            db, user_id, conversation_id
        )
        
        # Auto-load RAG files cho conversation mới
        if is_new_conversation:
            RAGService.load_rag_files_to_conversation(user_id, conversation.id)
        
        # Xử lý file và context
        file_context = FileService.process_file_for_chat(file, user_id, conversation.id)
        effective_query = ChatService._build_effective_query(message.message, file, file_context)
        
        # Chọn model dựa trên input evaluation
        model_name, tools, level_think = ChatService._select_model(effective_query, file)
        
        # Lấy context từ RAG
        rag_context = RAGService.get_rag_context(
            effective_query, user_id, conversation.id, db
        )
        
        # Tạo full prompt
        full_prompt = ChatService._build_full_prompt(rag_context, effective_query, file)
        
        # Generate stream response
        return ChatService._generate_stream_response(
            system_prompt=system_prompt,
            full_prompt=full_prompt,
            model_name=model_name,
            tools=tools,
            file=file,
            user_id=user_id,
            conversation_id=conversation.id,
            effective_query=effective_query,
            level_think=level_think,
            db=db
        )
    
    @staticmethod
    def _build_system_prompt(xung_ho: str, current_time: str) -> str:
        """Xây dựng system prompt"""
        return f"""
        Bạn là Nhi - một AI nói chuyện tự nhiên như con người, rất thông minh, trẻ con, dí dỏm và thân thiện.
        Bạn tự xưng Nhi và người dùng là {xung_ho}. Ví dụ: "Nhi rất vui được giúp {xung_ho}!"  


        **Khi cần gọi tool:**
        Trả đúng định dạng JSON, không thêm lời giải thích:
        {{
            "tool_calls": [
                {{
                    "type": "function",
                    "function": {{
                        "name": "web_search",
                        "arguments": "{{\\"query\\": \\"A well-optimized English query for web search\\"}}"
                    }}
                }}
        }}

        """
    
    @staticmethod
    def _get_or_create_conversation(db: Session, user_id: int, conversation_id: Optional[int]):
        """Lấy hoặc tạo conversation"""
        if conversation_id is not None:
            conversation = db.query(ModelConversation).filter(
                ModelConversation.id == conversation_id,
                ModelConversation.user_id == user_id
            ).first()
            if not conversation:
                raise HTTPException(404, "Conversation not found or not authorized")
            return conversation, False
        else:
            conversation = ModelConversation(user_id=user_id, created_at=datetime.utcnow())
            db.add(conversation)
            db.flush()
            return conversation, True
    
    @staticmethod
    def _build_effective_query(user_message: str, file, file_context: str) -> str:
        """Xây dựng effective query từ message và file context"""
        if not file:
            return user_message
        
        is_image = FileService.is_image_file(file)
        if is_image:
            return user_message
        else:
            effective_query = f"{user_message}\nNội dung file: {file_context}"
            if hasattr(file, 'filename') and file.filename:
                effective_query += f"\n(File: {file.filename})"
            return effective_query
    
    @staticmethod
    def _select_model(effective_query: str, file) -> tuple:
        """Chọn model phù hợp dựa trên input evaluation và trả về (model_name, tools, level_think)"""
        if file and FileService.is_image_file(file):
            return "qwen3-vl:235b-cloud", None, False
        
        eval_result = ChatService._evaluate_user_input(effective_query)
        logger.debug(f"Đánh giá input: {eval_result}")
        
        # Xác định level_think dựa trên độ phức tạp của câu hỏi
        level_think = ChatService._determine_think_level(effective_query, eval_result)
        
        if eval_result["needs_logic"]:
            return "4T-Logic", [web_search, web_fetch], False
        elif eval_result["needs_reasoning"]:
            return "4T-Reasoning", [web_search, web_fetch], level_think
        else:
            return "4T-S", [web_search, web_fetch], False
    
    @staticmethod
    def _determine_think_level(query: str, eval_result: Dict[str, bool]) -> Union[str, bool]:
        """
        Xác định mức độ think dựa trên độ phức tạp của câu hỏi
        
        Returns:
            Union[str, bool]: 'low', 'medium', 'high', False, True
        """
        # Mặc định là 'medium' cho các câu hỏi thông thường
        if not eval_result["needs_logic"] and not eval_result["needs_reasoning"]:
            return 'medium'
        
        # Phân tích độ dài và độ phức tạp của câu hỏi
        query_length = len(query)
        has_complex_keywords = any(keyword in query.lower() for keyword in [
            'so sánh', 'phân tích', 'đánh giá', 'giải thích chi tiết', 
            'tại sao', 'như thế nào', 'mối quan hệ', 'ưu nhược điểm'
        ])
        
        # Câu hỏi rất phức tạp: dài và có từ khóa phức tạp
        if query_length > 200 and has_complex_keywords:
            return 'high'
        # Câu hỏi phức tạp: có logic hoặc reasoning
        elif eval_result["needs_logic"] or eval_result["needs_reasoning"]:
            return 'high'
        # Câu hỏi trung bình
        elif query_length > 100:
            return 'medium'
        # Câu hỏi đơn giản
        else:
            return 'low'
    
    @staticmethod
    def _evaluate_user_input(input_text: str) -> Dict[str, bool]:
        """Đánh giá input của người dùng"""
        try:
            eval_prompt = f"""
            Bạn là một AI đánh giá input người dùng. Phân tích input sau và trả về JSON với hai trường:
            - "needs_logic": true nếu input liên quan đến toán học, lập trình, hoặc yêu cầu logic phức tạp.
            - "needs_reasoning": true nếu input yêu cầu suy luận sâu, phân tích phức tạp, hoặc trả lời dựa trên lý luận.
            Nếu không thuộc hai trường hợp trên, cả hai trường đều false.
            Input: {input_text}
            Lưu ý: Chỉ trả về định dạng JSON: {{"needs_logic": bool, "needs_reasoning": bool}}, không thêm bất kỳ text nào khác.
            """
            response = ollama.chat(
                model="4T-Evaluate:latest",
                messages=[{"role": "system", "content": eval_prompt}],
                stream=False,
                options={"temperature": 0, "top_p": 0}
            )
            try:
                result = json.loads(response["message"]["content"])
                return {
                    "needs_logic": bool(result.get("needs_logic", False)),
                    "needs_reasoning": bool(result.get("needs_reasoning", False))
                }
            except json.JSONDecodeError:
                logger.error("Lỗi khi parse JSON từ đánh giá input")
                return {"needs_logic": False, "needs_reasoning": False}
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá input: {e}")
            return {"needs_logic": False, "needs_reasoning": False}
    
    @staticmethod
    def _build_full_prompt(rag_context: str, effective_query: str, file) -> str:
        """Xây dựng full prompt cho model"""
        if FileService.is_image_file(file):
            return effective_query
        return f"Context: {rag_context}\nUser: {effective_query}"
    
    @staticmethod
    def _generate_stream_response(
        system_prompt: str,
        full_prompt: str,
        model_name: str,
        tools: list,
        file,
        user_id: int,
        conversation_id: int,
        effective_query: str,
        level_think: Union[str, bool],
        db: Session
    ):
        """Generate streaming response với level_think"""
        def generate_stream():
            yield f"data: {json.dumps({'conversation_id': conversation_id})}\n\n"
            full_response = []
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ]
            
            # Thêm images nếu là file ảnh
            if file and FileService.is_image_file(file):
                file_bytes = FileService.get_file_bytes(file)
                images = [base64.b64encode(file_bytes).decode('utf-8')]
                messages[-1]["images"] = images
            
            try:
                # Setup Ollama API key
                api_key = os.getenv('OLLAMA_API_KEY')
                if not api_key:
                    raise ValueError("OLLAMA_API_KEY env var not set")
                os.environ['OLLAMA_API_KEY'] = api_key

                # Chuẩn bị options cho Ollama
                options = {
                    "temperature": 0.6,
                    "repeat_penalty": 1.2,
                    "num_predict": -1,
                }

                # Tool call handling với vòng lặp
                max_iterations = 5
                current_iteration = 0
                
                while current_iteration < max_iterations:
                    current_iteration += 1
                    current_message: Dict[str, Any] = {"role": "assistant", "content": ""}
                    tool_calls: List[Dict[str, Any]] = []
                    
                    # Gọi Ollama với level_think
                    stream = ollama.chat(
                        model=model_name,
                        messages=messages,
                        tools=tools,
                        stream=True,
                        options=options,
                        think=level_think  # Thêm level_think vào đây
                    )
                    
                    for chunk in stream:
                        if "message" in chunk:
                            msg_chunk = chunk["message"]
                            if "tool_calls" in msg_chunk and msg_chunk["tool_calls"]:
                                serialized_tool_calls = [
                                    {
                                        "function": {
                                            "name": tc["function"]["name"],
                                            "arguments": tc["function"]["arguments"]
                                        }
                                    } for tc in msg_chunk["tool_calls"]
                                ]
                                yield f"data: {json.dumps({'tool_calls': serialized_tool_calls})}\n\n"
                                for tc in msg_chunk["tool_calls"]:
                                    if "function" in tc:
                                        tool_calls.append(tc)
                            if "content" in msg_chunk and msg_chunk["content"]:
                                delta = msg_chunk["content"].encode('utf-8').decode('utf-8', errors='replace')
                                current_message["content"] += delta
                                full_response.append(delta)
                                yield f"data: {json.dumps({'content': delta})}\n\n"
                    
                    messages.append(current_message)
                    
                    # Xử lý tool calls
                    if tool_calls:
                        for tool_call in tool_calls:
                            ChatService._handle_tool_call(tool_call, messages)
                        continue
                    else:
                        break
                    
                yield f"data: {json.dumps({'done': True})}\n\n"
                executor.submit(
                    ChatService._save_conversation_after_stream,
                    ''.join(full_response),
                    effective_query,
                    user_id,
                    conversation_id,
                    db
                ).result()
                
            except Exception as e:
                logger.error(f"Lỗi trong stream generation: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    
    @staticmethod
    def _handle_tool_call(tool_call: Dict[str, Any], messages: List[Dict[str, Any]]):
        """Xử lý tool call và thêm result vào messages"""
        function_name = tool_call['function']['name']
        args_str = tool_call['function']['arguments']
        
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
            
            if function_name == 'web_search':
                result = executor.submit(web_search, **args).result()
            elif function_name == 'web_fetch':
                result = executor.submit(web_fetch, **args).result()
            else:
                result = f"Tool {function_name} not found"
            
            tool_msg = {
                'role': 'tool',
                'content': str(result)[:10000],
                'tool_name': function_name
            }
            messages.append(tool_msg)
            logger.debug(f"Tool {function_name} result: {str(result)[:200]}...")
            
        except Exception as e:
            logger.error(f"Error executing tool {function_name}: {e}")
            tool_msg = {
                'role': 'tool',
                'content': f"Error: {str(e)}",
                'tool_name': function_name
            }
            messages.append(tool_msg)
    
    @staticmethod
    def _save_conversation_after_stream(
        full_response: str,
        effective_query: str,
        user_id: int,
        conversation_id: int,
        db: Session
    ):
        """Lưu conversation sau khi stream kết thúc"""
        if not full_response:
            logger.error("Empty response from stream")
            return
        
        try:
            # Tạo embeddings
            query_emb = EmbeddingService.get_embedding(effective_query)
            ass_emb = EmbeddingService.get_embedding(full_response)
            
            # Lưu messages
            user_msg = ModelChatMessage(
                user_id=user_id,
                conversation_id=conversation_id,
                content=effective_query,
                role="user",
                embedding=json.dumps(query_emb.tolist())
            )
            ass_msg = ModelChatMessage(
                user_id=user_id,
                conversation_id=conversation_id,
                content=full_response,
                role="assistant",
                embedding=json.dumps(ass_emb.tolist())
            )
            
            db.add_all([user_msg, ass_msg])
            db.flush()
            
            # Cập nhật FAISS index
            RAGService.update_faiss_index(user_id, conversation_id, db)
            db.commit()
            
        except Exception as e:
            db.rollback()
            logger.error(f"Lỗi khi lưu tin nhắn hoặc cập nhật FAISS index: {e}")
