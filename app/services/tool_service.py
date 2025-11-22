import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Union
import json
from ollama import web_search, web_fetch

logger = logging.getLogger(__name__)


class ToolService:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tools_map = {"web_search": web_search, "web_fetch": web_fetch}

    def get_tools(self) -> List[Any]:
        """Return list of available tools for the model"""
        return [web_search, web_fetch]

    def execute_tool(
        self, tool_name: str, args: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute a tool by name with arguments"""
        try:
            if isinstance(args, str):
                try:
                    tool_args = json.loads(args)
                except json.JSONDecodeError:
                    # If args is a string but not JSON, it might be a direct string argument (unlikely for these tools but good for safety)
                    # However, web_search and web_fetch expect kwargs.
                    # Let's assume it's a malformed JSON or just pass as is if the tool supports it?
                    # For now, let's stick to the logic in chat_service which tries to load JSON.
                    tool_args = args
            else:
                tool_args = args

            if tool_name not in self.tools_map:
                return {"error": f"Tool {tool_name} not found", "result": None}

            tool_func = self.tools_map[tool_name]

            # Execute in thread pool
            future = self.executor.submit(tool_func, **tool_args)
            result = future.result()

            return {
                "error": None,
                "result": result,
                "tool_name": tool_name,
                "args": tool_args,
            }

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"error": str(e), "result": None, "tool_name": tool_name}
