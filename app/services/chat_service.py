import json
import logging
import ollama
from ollama import web_search, web_fetch
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from .rag_service import RAGService
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        self.rag_service = RAGService()
        self.embedding_service = EmbeddingService()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def evaluate_user_input(self, input_text: str) -> Dict[str, bool]:
        """Đánh giá input của người dùng để xác định model phù hợp."""
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
                model="gpt-oss:20b",
                messages=[{"role": "system", "content": eval_prompt}],
                stream=False,
                options={
                    "temperature": 0,
                    "top_p": 0
                },
                think="low",
                format=json
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

    def select_model(self, eval_result: Dict[str, bool], is_image: bool = False) -> tuple:
        """Chọn model phù hợp dựa trên đánh giá input.

        Returns a tuple: (model_name, tools, think_level)
        - think_level: one of 'low', 'medium', 'high' (passed to ollama via `think` arg)
        """
        if is_image:
            return "qwen3-vl:235b-cloud", None, "low"
        elif eval_result.get("needs_logic"):
            return "4T-Logic", [web_search, web_fetch], "low"
        elif eval_result.get("needs_reasoning"):
            # For reasoning requests, prefer the general-purpose gpt-oss with higher `think`
            return "gpt-oss:20b", [web_search, web_fetch], "high"
        else:
            return "gpt-oss:20b", [web_search, web_fetch], "low"

    def build_system_prompt(self, gender: str, current_time: str) -> str:
        """Xây dựng system prompt"""
        xung_ho = "anh" if gender == "male" else "chị" if gender == "female" else "bạn"
        
        return f"""
        Thời điểm hiện tại: {current_time}.
        
        Giao tiếp với người dùng bằng cách xưng hô là "{xung_ho}".
        
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
        - Không lan man.
        - Luôn duy trì cảm giác "người thật nói chuyện" chứ không như máy.
        """

    def prepare_chat_context(self, effective_query: str, file_rag_context: str, 
                           user_id: int, conversation_id: int, history_messages: List) -> str:
        """Chuẩn bị context cho chat"""
        query_embedding = self.embedding_service.get_embedding(effective_query, max_length=1024)
        
        all_context_parts = []
        
        if file_rag_context:
            all_context_parts.append(f"File Context: {file_rag_context}")
        
        context_from_history = self.rag_service.search_similar_context(
            query_embedding, user_id, conversation_id, history_messages
        )
        
        if context_from_history:
            all_context_parts.append(f"History Context:\n{context_from_history}")
        
        final_context = "\n\n".join(all_context_parts)
        if not final_context and history_messages:
            final_context = "\n".join([h.content for h in history_messages[-10:]])
        
        return final_context
