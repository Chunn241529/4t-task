import base64
from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse
import ollama
from ollama import web_search, web_fetch
from app.services.tool_service import ToolService
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
tool_service = ToolService()


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

        # Evaluate using keywords instead of LLM
        input_lower = effective_query.lower()

        # Logic keywords (math, coding, technical)
        logic_keywords = [
            "code",
            "python",
            "java",
            "c++",
            "javascript",
            "sql",
            "lập trình",
            "thuật toán",
            "bug",
            "error",
            "fix",
            "debug",
            "toán",
            "tính toán",
            "công thức",
            "phương trình",
            "logic",
            "function",
            "class",
            "api",
        ]
        needs_logic = any(k in input_lower for k in logic_keywords)

        # Reasoning keywords (analysis, comparison, explanation)
        reasoning_keywords = [
            "tại sao",
            "vì sao",
            "như thế nào",
            "giải thích",
            "phân tích",
            "so sánh",
            "đánh giá",
            "ý nghĩa",
            "nguyên nhân",
            "hệ quả",
            "suy luận",
            "quan điểm",
            "nhận xét",
            "ưu điểm",
            "nhược điểm",
            "khác nhau",
            "giống nhau",
        ]
        needs_reasoning = any(k in input_lower for k in reasoning_keywords)

        # Determine think level based on keywords and length
        level_think = "low"
        if needs_reasoning or needs_logic:
            if (
                len(effective_query) > 200
                or "chi tiết" in input_lower
                or "sâu" in input_lower
            ):
                level_think = "high"
            else:
                level_think = "medium"

        tools = tool_service.get_tools()

        if needs_logic:
            return "4T-Logic", tools, False
        elif needs_reasoning:
            return "4T", tools, level_think
        else:
            return "4T-S", tools, False

    @staticmethod
    def _get_conversation_history(
        db: Session, conversation_id: int, limit: int = 20
    ) -> List[Dict[str, str]]:
        """Retrieve conversation history from database"""
        messages = (
            db.query(ModelChatMessage)
            .filter(ModelChatMessage.conversation_id == conversation_id)
            .order_by(ModelChatMessage.timestamp.asc())
            .limit(limit)
            .all()
        )

        return [{"role": msg.role, "content": msg.content} for msg in messages]

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

            # Get RECENT conversation history (last 10 messages for context)
            recent_history = ChatService._get_conversation_history(
                db, conversation_id, limit=10
            )

            # Build messages with recent history
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt}
            ]

            # Add recent history for conversation flow
            messages.extend(recent_history)

            # Add current user message
            messages.append({"role": "user", "content": full_prompt})

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

                            # Handle thinking/reasoning content
                            # Check in message chunk
                            if (
                                "reasoning_content" in msg_chunk
                                and msg_chunk["reasoning_content"]
                            ):
                                delta = msg_chunk["reasoning_content"]
                                yield f"data: {json.dumps({'thinking': delta})}\n\n"
                            elif "think" in msg_chunk and msg_chunk["think"]:
                                delta = msg_chunk["think"]
                                yield f"data: {json.dumps({'thinking': delta})}\n\n"
                            elif "reasoning" in msg_chunk and msg_chunk["reasoning"]:
                                delta = msg_chunk["reasoning"]
                                yield f"data: {json.dumps({'thinking': delta})}\n\n"
                            elif "thought" in msg_chunk and msg_chunk["thought"]:
                                delta = msg_chunk["thought"]
                                yield f"data: {json.dumps({'thinking': delta})}\n\n"

                        # Check top-level chunk for thinking fields (some models might put it here)
                        if "reasoning_content" in chunk and chunk["reasoning_content"]:
                            delta = chunk["reasoning_content"]
                            yield f"data: {json.dumps({'thinking': delta})}\n\n"
                        elif "think" in chunk and chunk["think"]:
                            delta = chunk["think"]
                            yield f"data: {json.dumps({'thinking': delta})}\n\n"

                        # Always stream the raw chunk if it's not a tool call
                        if not iteration_has_tool_calls:
                            # Convert ChatResponse to dict if needed
                            chunk_data = (
                                chunk.model_dump()
                                if hasattr(chunk, "model_dump")
                                else chunk
                            )
                            yield f"data: {json.dumps(chunk_data)}\n\n"

                    messages.append(current_message)

                    if tool_calls:
                        for tool_call in tool_calls:
                            function_name = tool_call["function"]["name"]
                            args_str = tool_call["function"]["arguments"]

                            # Execute tool via service
                            execution_result = tool_service.execute_tool(
                                function_name, args_str
                            )

                            if execution_result["error"]:
                                tool_msg = {
                                    "role": "tool",
                                    "content": f"Error: {execution_result['error']}",
                                    "tool_name": function_name,
                                }
                            else:
                                result = execution_result["result"]

                                # Handle search specific logic (sending status)
                                if function_name == "web_search":
                                    try:
                                        # Parse args for query
                                        if isinstance(args_str, str):
                                            args = json.loads(args_str)
                                        else:
                                            args = args_str

                                        result_data = (
                                            json.loads(result)
                                            if isinstance(result, str)
                                            else result
                                        )
                                        result_count = (
                                            len(result_data.get("results", []))
                                            if isinstance(result_data, dict)
                                            else 0
                                        )

                                        yield f"data: {json.dumps({'search_complete': {'query': args.get('query', ''), 'count': result_count}})}\n\n"
                                    except Exception as e:
                                        logger.debug(
                                            f"Could not parse search results for count: {e}"
                                        )

                                tool_msg = {
                                    "role": "tool",
                                    "content": str(result)[:8000],
                                    "tool_name": function_name,
                                }

                            messages.append(tool_msg)

                        continue
                    else:
                        break

                # Stream raw chunk
                chunk_data = (
                    chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
                )
                yield f"data: {json.dumps(chunk_data)}\n\n"

                # Accumulate content for saving
                if "message" in chunk_data:
                    msg_chunk = chunk_data["message"]
                    if "content" in msg_chunk and msg_chunk["content"]:
                        full_response.append(msg_chunk["content"])

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
