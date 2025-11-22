# command_worker.py
# -*- coding: utf-8 -*-
from PySide6.QtCore import QThread, Signal
import aiohttp
import asyncio
import json


class CommandWorker(QThread):
    result_ready = Signal(object)  # Emits result data (list, dict, or bool)
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(self, command: str, base_url: str, token: str, **kwargs):
        super().__init__()
        self.command = command
        self.base_url = base_url
        self.token = token
        self.kwargs = kwargs
        self._loop = None

    def run(self):
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._execute_command())
        except Exception as e:
            self.error_occurred.emit(f"Lỗi CommandWorker: {str(e)}")
        finally:
            if self._loop and self._loop.is_running():
                self._loop.close()
            self.finished.emit()

    async def _execute_command(self):
        headers = {"Authorization": f"Bearer {self.token}"}
        async with aiohttp.ClientSession(headers=headers) as session:
            try:
                if self.command == "history":
                    async with session.get(
                        f"{self.base_url}/conversations/"
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            self.result_ready.emit({"type": "history", "data": data})
                        else:
                            self.error_occurred.emit(
                                f"Lỗi tải lịch sử: {response.status}"
                            )

                elif self.command == "load":
                    conv_id = self.kwargs.get("conversation_id")
                    async with session.get(
                        f"{self.base_url}/messages/conversations/{conv_id}/messages"
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            self.result_ready.emit(
                                {
                                    "type": "load",
                                    "data": data,
                                    "conversation_id": conv_id,
                                }
                            )
                        elif response.status == 404:
                            self.error_occurred.emit(
                                f"Conversation #{conv_id} không tồn tại."
                            )
                        else:
                            self.error_occurred.emit(
                                f"Lỗi tải cuộc hội thoại: {response.status}"
                            )

                elif self.command == "delete":
                    conv_id = self.kwargs.get("conversation_id")
                    if conv_id:
                        async with session.delete(
                            f"{self.base_url}/conversations/{conv_id}"
                        ) as response:
                            if response.status == 200:
                                self.result_ready.emit(
                                    {"type": "delete", "success": True}
                                )
                            else:
                                self.error_occurred.emit(
                                    f"Lỗi xóa cuộc hội thoại: {response.status}"
                                )
                    else:
                        self.error_occurred.emit("Không có cuộc hội thoại nào để xóa.")

                elif self.command == "delete_all":
                    async with session.delete(
                        f"{self.base_url}/conversations/"
                    ) as response:
                        if response.status == 200:
                            self.result_ready.emit(
                                {"type": "delete_all", "success": True}
                            )
                        else:
                            self.error_occurred.emit(
                                f"Lỗi xóa tất cả: {response.status}"
                            )

            except aiohttp.ClientError as e:
                self.error_occurred.emit(f"Lỗi kết nối: {str(e)}")
            except Exception as e:
                self.error_occurred.emit(f"Lỗi không xác định: {str(e)}")
