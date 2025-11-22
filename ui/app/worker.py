# worker.py
# -*- coding: utf-8 -*-
from asyncio.log import logger
import json
import base64
import gc
from PySide6.QtCore import QThread, Signal
import aiohttp
import asyncio
import re


class OllamaWorker(QThread):
    chunk_received = Signal(str)
    thinking_received = Signal(str)
    search_started = Signal(str)
    search_sources = Signal(str)
    content_started = Signal()
    image_processing = Signal()
    image_description = Signal(str)
    error_received = Signal(str)
    finished = Signal()
    conversation_id_received = Signal(int)

    def __init__(
        self,
        prompt: str,
        token: str,
        image_base64: str = None,
        is_thinking: bool = False,
        conversation_id: int = None,
    ):
        super().__init__()
        self.prompt = prompt
        self.token = token
        self.image_base64 = image_base64
        self.is_thinking = is_thinking
        self.conversation_id = conversation_id
        self.base_url = "http://localhost:8000"
        self.max_image_size = 20 * 1024 * 1024  # 20MB giới hạn
        self.partial_buffer = ""  # Biến để lưu phần còn lại nếu thẻ bị chia cắt
        self.thinking_buffer = ""  # Biến để tích lũy nội dung thinking
        self.in_thinking = False  # Trạng thái đang trong thinking
        self._task = None
        self._loop = None

    def run(self):
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._task = self._loop.create_task(self._stream_response())
            self._loop.run_until_complete(self._task)
        except asyncio.CancelledError:
            logger.info("Worker task cancelled")
        except Exception as e:
            self.error_received.emit(f"Lỗi trong OllamaWorker: {str(e)}")
            print(f"OllamaWorker error: {str(e)}")
        finally:
            if self._loop and self._loop.is_running():
                self._loop.close()
            self.finished.emit()

    def stop(self):
        if self._loop and self._task:
            self._loop.call_soon_threadsafe(self._task.cancel)

    async def _stream_response(self):
        try:
            cleaned_image_base64 = None
            if self.image_base64:
                try:
                    cleaned_image_base64 = self.image_base64
                    if cleaned_image_base64.startswith("data:image"):
                        # Keep the prefix for the backend to detect it as image string
                        pass
                    image_size = len(
                        base64.b64decode(
                            cleaned_image_base64.split(",")[1]
                            if "," in cleaned_image_base64
                            else cleaned_image_base64
                        )
                    )
                    if image_size > self.max_image_size:
                        self.error_received.emit(
                            "Hình ảnh quá lớn, vượt quá giới hạn 20MB"
                        )
                        print("Image size exceeds 20MB limit")
                        return
                except Exception as e:
                    self.error_received.emit(f"Lỗi xử lý ảnh: {str(e)}")
                    print(f"Image processing error: {str(e)}")
                    return

            async with aiohttp.ClientSession() as session:
                # Check if conversation_id is None, if so create new conversation
                if self.conversation_id is None:
                    try:
                        create_headers = {"Authorization": f"Bearer {self.token}"}
                        async with session.post(
                            f"{self.base_url}/conversations/", headers=create_headers
                        ) as create_response:
                            if create_response.status == 200:
                                conv_data = await create_response.json()
                                self.conversation_id = conv_data["id"]
                                self.conversation_id_received.emit(self.conversation_id)
                                print(
                                    f"Created new conversation: {self.conversation_id}"
                                )
                            else:
                                print(
                                    f"Failed to create conversation: {create_response.status}"
                                )
                    except Exception as e:
                        print(f"Error creating conversation: {e}")

                # Construct payload for /send endpoint
                payload = {
                    "message": {"message": self.prompt},
                    "file": cleaned_image_base64 if cleaned_image_base64 else None,
                }

                # Add params for conversation_id
                params = {}
                if self.conversation_id:
                    params["conversation_id"] = str(self.conversation_id)

                logger.debug(
                    f"Gửi payload: {json.dumps(payload, ensure_ascii=False)[:100]}..."
                )

                headers = {"Authorization": f"Bearer {self.token}"}
                async with session.post(
                    f"{self.base_url}/send",
                    json=payload,
                    headers=headers,
                    params=params,
                ) as response:
                    if response.status != 200:
                        self.error_received.emit(f"Lỗi HTTP: {response.status}")
                        print(f"HTTP error: {response.status}")
                        return

                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if not line or not line.startswith("data: "):
                            continue

                        json_str = line[6:]  # Remove "data: "
                        if json_str == "[DONE]":
                            break

                        try:
                            data = json.loads(json_str)

                            if "error" in data:
                                self.error_received.emit(data["error"])
                                continue

                            if "done" in data and data["done"]:
                                break

                            if "conversation_id" in data:
                                self.conversation_id_received.emit(
                                    data["conversation_id"]
                                )

                            if "tool_calls" in data:
                                for tool in data["tool_calls"]:
                                    if tool["function"]["name"] == "web_search":
                                        args = (
                                            json.loads(tool["function"]["arguments"])
                                            if isinstance(
                                                tool["function"]["arguments"], str
                                            )
                                            else tool["function"]["arguments"]
                                        )
                                        query = args.get("query", "")
                                        self.search_started.emit(query)

                            if "content" in data:
                                content = data["content"]
                                if content and isinstance(content, str):
                                    # Existing logic for parsing <think> tags
                                    content = content.replace("\\u003c", "<").replace(
                                        "\\u003e", ">"
                                    )
                                    content = self.partial_buffer + content
                                    self.partial_buffer = ""

                                    if not self.in_thinking:
                                        think_start = content.find("<think>")
                                        if think_start == -1:
                                            if content:
                                                self.chunk_received.emit(content)
                                        else:
                                            before = content[:think_start]
                                            if before:
                                                self.chunk_received.emit(before)
                                            content = content[
                                                think_start + len("<think>") :
                                            ]
                                            self.in_thinking = True
                                            self.thinking_buffer += content
                                            if self.thinking_buffer:
                                                self.thinking_received.emit(
                                                    self.thinking_buffer
                                                )
                                                self.thinking_buffer = ""
                                    else:
                                        think_end = content.find("</think>")
                                        if think_end == -1:
                                            self.thinking_buffer += content
                                            if self.thinking_buffer:
                                                self.thinking_received.emit(
                                                    self.thinking_buffer
                                                )
                                                self.thinking_buffer = ""
                                        else:
                                            thinking_part = content[:think_end]
                                            if thinking_part:
                                                self.thinking_buffer += thinking_part
                                                self.thinking_received.emit(
                                                    self.thinking_buffer
                                                )
                                                self.thinking_buffer = ""
                                            after = content[
                                                think_end + len("</think>") :
                                            ]
                                            if after:
                                                self.chunk_received.emit(after)
                                            self.in_thinking = False

                                    if content and (
                                        content.endswith("<think")
                                        or content.endswith("</think")
                                        or content.endswith("<")
                                    ):
                                        self.partial_buffer = content

                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Lỗi giải mã JSON: {e}, Raw line: {line[:100]}"
                            )
                            continue
                        finally:
                            gc.collect()
        except Exception as e:
            self.error_received.emit(f"Lỗi kết nối server: {str(e)}")
            print(f"Server connection error: {str(e)}")
        finally:
            if self.thinking_buffer:
                self.thinking_received.emit(self.thinking_buffer)
                self.thinking_buffer = ""
            if self.partial_buffer:
                self.chunk_received.emit(self.partial_buffer)
                self.partial_buffer = ""
            gc.collect()
            self.finished.emit()
