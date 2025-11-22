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
from app.db import SessionLocal

from app.models import (
    ChatMessage as ModelChatMessage,
    Conversation as ModelConversation,
    User,
)
from app.schemas import ChatMessageIn
from app.services.embedding_service import EmbeddingService
from app.services.file_service import FileService
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)
executor = ThreadPoolExecutor(max_workers=4)


SEARCH_TRIGGERS = [
    "tìm kiếm",
    "tra cứu",
    "search",
    "google",
    "tin tức",
    "thời tiết",
    "sự kiện",
    "lịch thi đấu",
    "bảng xếp hạng",
    "review",
    "so sánh giá",
]


class ChatService:

    @staticmethod
    def chat_with_rag(
        message: ChatMessageIn,
        file: Optional[Union[UploadFile, str]],
        conversation_id: Optional[int],
        user_id: int,
        db: Session,
    ):
        """Xử lý chat chính với RAG integration - với debug chi tiết"""

        # Lấy thông tin user và xưng hô
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        gender = user.gender
        xung_ho = "anh" if gender == "male" else "chị" if gender == "female" else "bạn"
        current_time = datetime.now().strftime("%Y-%m-%d %I:%M %p %z")

        # Check search triggers
        force_search = any(
            trigger in message.message.lower() for trigger in SEARCH_TRIGGERS
        )
        if force_search:
            logger.info("Search trigger detected, forcing web search")

        # System prompt
        system_prompt = ChatService._build_system_prompt(
            xung_ho, current_time, force_search
        )

        # Xử lý conversation
        conversation, is_new_conversation = ChatService._get_or_create_conversation(
            db, user_id, conversation_id
        )

        logger.info(
            f"Using conversation {conversation.id}, is_new: {is_new_conversation}"
        )

        # ĐẢM BẢO RAG FILES ĐƯỢC LOAD
        logger.info("Ensuring RAG files are loaded...")
        rag_loaded = RAGService._ensure_rag_loaded(user_id, conversation.id)
        logger.info(f"RAG loaded: {rag_loaded}")

        # Xử lý file và context
        file_context = FileService.process_file_for_chat(file, user_id, conversation.id)
        effective_query = ChatService._build_effective_query(
            message.message, file, file_context
        )

        logger.info(f"Effective query: {effective_query[:200]}...")

        # Chọn model dựa trên input evaluation
        model_name, tools, level_think = ChatService._select_model(
            effective_query, file
        )
        logger.info(f"Selected model: {model_name}, level_think: {level_think}")

        # Lấy context từ RAG
        rag_context = RAGService.get_rag_context(
            effective_query, user_id, conversation.id, db
        )

        logger.info(
            f"RAG context retrieved: {len(rag_context) if rag_context else 0} characters"
        )

        # Tạo full prompt với RAG context
        full_prompt = ChatService._build_full_prompt(rag_context, effective_query, file)

        logger.info(f"Full prompt length: {len(full_prompt)} characters")

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
            db=db,
        )

    @staticmethod
    def _build_system_prompt(
        xung_ho: str, current_time: str, force_search: bool = False
    ) -> str:
        """Xây dựng system prompt với hướng dẫn sử dụng RAG"""
        prompt = f"""
        Bạn là Nhi - một AI nói chuyện tự nhiên như con người, rất thông minh, trẻ con, dí dỏm và thân thiện.
        Bạn tự xưng Nhi và người dùng là {xung_ho}. Ví dụ: "Nhi rất vui được giúp {xung_ho}!"  
        
        Thời gian hiện tại: {current_time}
        """

        if force_search:
            prompt += """
            
            QUAN TRỌNG: Người dùng đang yêu cầu tìm kiếm thông tin cụ thể hoặc cập nhật.
            BẠN BẮT BUỘC PHẢI SỬ DỤNG CÔNG CỤ `web_search` để tìm thông tin chính xác và mới nhất trước khi trả lời.
            KHÔNG được bịa đặt thông tin. Nếu không tìm thấy, hãy nói rõ.
            """

        return prompt

    @staticmethod
    def _get_or_create_conversation(
        db: Session, user_id: int, conversation_id: Optional[int]
    ):
        """Lấy hoặc tạo conversation"""
        if conversation_id is not None:
            conversation = (
                db.query(ModelConversation)
                .filter(
                    ModelConversation.id == conversation_id,
                    ModelConversation.user_id == user_id,
                )
                .first()
            )
            if not conversation:
                raise HTTPException(404, "Conversation not found or not authorized")
            return conversation, False
        else:
            conversation = ModelConversation(
                user_id=user_id, created_at=datetime.utcnow()
            )
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
            effective_query = f"{user_message}"
            if file_context:
                effective_query += f"\n\nFile content reference: {file_context}"
            if hasattr(file, "filename") and file.filename:
                effective_query += f"\n(File: {file.filename})"
            return effective_query

    @staticmethod
    def _select_model(effective_query: str, file) -> tuple:
        """Chọn model phù hợp dựa trên input evaluation"""
        if file and FileService.is_image_file(file):
            return "qwen3-vl:235b-cloud", None, False

        eval_result = ChatService._evaluate_user_input(effective_query)
        logger.debug(f"Input evaluation: {eval_result}")

        level_think = ChatService._determine_think_level(effective_query, eval_result)
        tools = [web_search, web_fetch]

        if eval_result["needs_logic"]:
            return "4T-Logic", tools, False
        elif eval_result["needs_reasoning"]:
            return "4T", tools, level_think
        else:
            return "4T-S", tools, False

    @staticmethod
    def _determine_think_level(
        query: str, eval_result: Dict[str, bool]
    ) -> Union[str, bool]:
        """Xác định mức độ think"""
        if not eval_result["needs_logic"] and not eval_result["needs_reasoning"]:
            return "medium"

        query_length = len(query)
        has_complex_keywords = any(
            keyword in query.lower()
            for keyword in [
                "so sánh",
                "phân tích",
                "đánh giá",
                "giải thích chi tiết",
                "tại sao",
                "như thế nào",
                "mối quan hệ",
                "ưu nhược điểm",
                "phân tích",
                "suy luận",
                "suy nghĩ",
                "think",
                "reasoning",
            ]
        )

        if query_length > 2000 and has_complex_keywords:
            return "high"
        elif eval_result["needs_logic"] or eval_result["needs_reasoning"]:
            return "medium"
        elif query_length > 100:
            return "medium"
        else:
            return "low"

    @staticmethod
    def _evaluate_user_input(input_text: str) -> Dict[str, bool]:
        """Đánh giá input của người dùng"""
        try:
            eval_prompt = f"""
            Đánh giá input người dùng và trả về JSON:
            - "needs_logic": true nếu liên quan đến toán học, lập trình, logic phức tạp
            - "needs_reasoning": true nếu yêu cầu suy luận sâu, phân tích phức tạp
            Input: {input_text}
            Chỉ trả về JSON: {{"needs_logic": bool, "needs_reasoning": bool}}
            """
            response = ollama.chat(
                model="4T-Evaluate:latest",
                messages=[{"role": "system", "content": eval_prompt}],
                stream=False,
                options={"temperature": 0, "top_p": 0},
            )
            try:
                result = json.loads(response["message"]["content"])
                return {
                    "needs_logic": bool(result.get("needs_logic", False)),
                    "needs_reasoning": bool(result.get("needs_reasoning", False)),
                }
            except json.JSONDecodeError:
                logger.error("Lỗi khi parse JSON từ đánh giá input")
                return {"needs_logic": False, "needs_reasoning": False}
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá input: {e}")
            return {"needs_logic": False, "needs_reasoning": False}

    @staticmethod
    def _build_full_prompt(rag_context: str, effective_query: str, file) -> str:
        """Xây dựng full prompt cho model - cải thiện để sử dụng RAG context"""
        if FileService.is_image_file(file):
            return effective_query

        if rag_context and rag_context.strip():
            # Tách các context chunks và format lại
            context_chunks = rag_context.split("|||")
            formatted_context = "\n\n".join(
                [f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)]
            )

            prompt = f"""Hãy sử dụng thông tin từ các thông tin dưới đây để trả lời câu hỏi. Nếu thông tin không đủ, hãy sử dụng kiến thức của bạn.

            {formatted_context}

            Câu hỏi: {effective_query}

            Hãy trả lời dựa trên thông tin được cung cấp và luôn trả lời bằng tiếng Việt tự nhiên:"""
        else:
            prompt = effective_query

        return prompt

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
        db: Session,
    ):
        """Generate streaming response với level_think"""

        def generate_stream():
            yield f"data: {json.dumps({'conversation_id': conversation_id})}\n\n"
            full_response = []
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt},
            ]

            if file and FileService.is_image_file(file):
                file_bytes = FileService.get_file_bytes(file)
                images = [base64.b64encode(file_bytes).decode("utf-8")]
                messages[-1]["images"] = images

            try:
                api_key = os.getenv("OLLAMA_API_KEY")
                if not api_key:
                    raise ValueError("OLLAMA_API_KEY env var not set")
                os.environ["OLLAMA_API_KEY"] = api_key

                options = {
                    "temperature": 0.6,
                    "repeat_penalty": 1.2,
                    "num_predict": 8192,
                }

                max_iterations = 5
                current_iteration = 0
                has_tool_calls = False

                while current_iteration < max_iterations:
                    current_iteration += 1
                    current_message: Dict[str, Any] = {
                        "role": "assistant",
                        "content": "",
                    }
                    tool_calls: List[Dict[str, Any]] = []

                    stream = ollama.chat(
                        model=model_name,
                        messages=messages,
                        tools=tools,
                        stream=True,
                        options=options,
                        think=level_think,
                    )

                    iteration_has_tool_calls = False

                    for chunk in stream:
                        if "message" in chunk:
                            msg_chunk = chunk["message"]
                            if "tool_calls" in msg_chunk and msg_chunk["tool_calls"]:
                                iteration_has_tool_calls = True
                                has_tool_calls = True

                                serialized_tool_calls = [
                                    {
                                        "function": {
                                            "name": tc["function"]["name"],
                                            "arguments": tc["function"]["arguments"],
                                        }
                                    }
                                    for tc in msg_chunk["tool_calls"]
                                ]
                                yield f"data: {json.dumps({'tool_calls': serialized_tool_calls})}\n\n"

                                for tc in msg_chunk["tool_calls"]:
                                    if "function" in tc:
                                        tool_calls.append(tc)
                            if "content" in msg_chunk and msg_chunk["content"]:
                                delta = msg_chunk["content"]
                                current_message["content"] += delta

                                if not iteration_has_tool_calls:
                                    full_response.append(delta)
                                    yield f"data: {json.dumps({'content': delta})}\n\n"

                    messages.append(current_message)

                    if tool_calls:
                        for tool_call in tool_calls:
                            function_name = tool_call["function"]["name"]
                            args_str = tool_call["function"]["arguments"]
                            try:
                                if isinstance(args_str, str):
                                    args = json.loads(args_str)
                                else:
                                    args = args_str

                                if function_name == "web_search":
                                    result = executor.submit(
                                        web_search, **args
                                    ).result()
                                elif function_name == "web_fetch":
                                    result = executor.submit(web_fetch, **args).result()
                                else:
                                    result = f"Tool {function_name} not found"

                                tool_msg = {
                                    "role": "tool",
                                    "content": str(result)[:8000],
                                    "tool_name": function_name,
                                }
                                messages.append(tool_msg)

                            except Exception as e:
                                logger.error(
                                    f"Error executing tool {function_name}: {e}"
                                )
                                tool_msg = {
                                    "role": "tool",
                                    "content": f"Error: {str(e)}",
                                    "tool_name": function_name,
                                }
                                messages.append(tool_msg)

                        continue
                    else:
                        break

                # Final response với tool results
                if has_tool_calls and len(messages) > 2:
                    final_stream = ollama.chat(
                        model=model_name,
                        messages=messages,
                        stream=True,
                        options=options,
                    )

                    for chunk in final_stream:
                        if "message" in chunk and "content" in chunk["message"]:
                            delta = chunk["message"]["content"]
                            current_message["content"] += delta
                            full_response.append(delta)
                            yield f"data: {json.dumps({'content': delta})}\n\n"

                yield f"data: {json.dumps({'done': True})}\n\n"

                if full_response:
                    executor.submit(
                        ChatService._save_conversation_after_stream,
                        "".join(full_response),
                        effective_query,
                        user_id,
                        conversation_id,
                    )

            except Exception as e:
                logger.error(f"Lỗi trong stream generation: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    @staticmethod
    def _save_conversation_after_stream(
        full_response: str,
        effective_query: str,
        user_id: int,
        conversation_id: int,
    ):
        """Lưu conversation sau khi stream kết thúc"""
        if not full_response or not full_response.strip():
            logger.warning("Empty response from stream, skipping save")
            return

        db = SessionLocal()
        try:
            query_emb = EmbeddingService.get_embedding(effective_query)
            ass_emb = EmbeddingService.get_embedding(full_response)

            user_msg = ModelChatMessage(
                user_id=user_id,
                conversation_id=conversation_id,
                content=effective_query,
                role="user",
                embedding=json.dumps(query_emb.tolist()),
            )
            ass_msg = ModelChatMessage(
                user_id=user_id,
                conversation_id=conversation_id,
                content=full_response,
                role="assistant",
                embedding=json.dumps(ass_emb.tolist()),
            )

            db.add_all([user_msg, ass_msg])
            db.flush()

            RAGService.update_faiss_index(user_id, conversation_id, db)
            db.commit()

        except Exception as e:
            db.rollback()
            logger.error(f"Lỗi khi lưu tin nhắn: {e}")
        finally:
            db.close()
