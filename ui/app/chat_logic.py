# chat_logic.py
# -*- coding: utf-8 -*-
import markdown
import json
from urllib.parse import urlparse
from typing import Optional
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QTextEdit
from worker import OllamaWorker
from screenshot_capture import ScreenshotOverlay


class ChatLogic:
    def __init__(self, parent):
        self.parent = parent
        self.ollama_thread: Optional[OllamaWorker] = None
        self.chunk_buffer = ""
        self.thinking_buffer = ""
        self.full_thinking_md = ""
        self.buffer_timer = QTimer()
        self.buffer_timer.setInterval(
            30
        )  # Increased from 5ms to 30ms to reduce CPU usage
        self.buffer_timer.timeout.connect(self._flush_buffer)
        self.parent.user_scrolling = False
        self.last_resize_time = 0
        self.resize_throttle_ms = 100  # Throttle resizing to every 100ms

    def setup_connections(self) -> None:
        self.parent.ui.send_stop_button.send_clicked.connect(self.send_prompt)
        self.parent.ui.send_stop_button.stop_clicked.connect(self.stop_worker)

    def handle_key_press(self, event) -> None:
        if event.key() == Qt.Key_Return and not event.modifiers() & Qt.ShiftModifier:
            event.accept()
            if self.ollama_thread and self.ollama_thread.isRunning():
                print("Worker is running, ignoring new prompt")
                return
            self.send_prompt()
        else:
            QTextEdit.keyPressEvent(self.parent.ui.input_box, event)

    def send_prompt(self) -> None:
        prompt_text = self.parent.ui.input_box.toPlainText().strip()
        if not prompt_text:
            print("Prompt is empty, ignoring")
            return

        # Handle commands
        if prompt_text.startswith("/"):
            self.handle_command(prompt_text)
            self.parent.ui.input_box.clear()
            return

        print(f"Sending prompt: {prompt_text}")
        self.parent.ui.scroll_area.setVisible(True)
        self.parent.ui.input_box.setDisabled(True)
        self.parent.full_response_md = ""
        self.full_thinking_md = ""
        self.chunk_buffer = ""
        self.thinking_buffer = ""
        self.parent.ui.response_display.clear()
        self.parent.ui.thinking_display.clear()
        self.parent.ui.thinking_widget.hide()
        self.parent.ui.input_box.setPlaceholderText("AI đang suy nghĩ...")
        self.parent.user_scrolling = False
        self.parent.sources_data = []
        self.parent.waiting_for_response = True
        self.parent.spinner_logic.start_thinking()
        self.parent.ui.send_stop_button.set_running(True)

        image_base64 = self.parent.current_screenshot_base64

        # Clear the screenshot preview after capturing the data
        if image_base64:
            self.parent.ui.delete_screenshot()

        if self.ollama_thread:
            if self.ollama_thread.isRunning():
                print("Waiting for previous thread to finish")
                self.ollama_thread.quit()
                self.ollama_thread.wait()
            self.ollama_thread.deleteLater()
            self.ollama_thread = None

        token = self.parent.token
        # Pass conversation_id if available
        conversation_id = getattr(self.parent, "conversation_id", None)

        # Note: We need to update OllamaWorker to accept conversation_id if we want to continue chat
        # For now, let's assume the backend handles it or we need to pass it.
        # Since OllamaWorker doesn't take conversation_id in __init__ yet, we might need to update it too
        # or just rely on the fact that we are sending a new request.
        # Wait, the backend /send endpoint takes conversation_id.
        # We should update OllamaWorker to accept conversation_id.

        self.ollama_thread = OllamaWorker(
            prompt_text,
            token=token,
            image_base64=image_base64,
            is_thinking=True,
            conversation_id=conversation_id,
        )
        # Inject conversation_id into worker if needed, or update worker init.
        # For this step, let's just stick to command logic.

        self.ollama_thread.chunk_received.connect(self._buffer_chunk)
        self.ollama_thread.thinking_received.connect(self._buffer_thinking)
        self.ollama_thread.search_started.connect(self.on_search_started)
        self.ollama_thread.search_complete.connect(self.on_search_complete)
        self.ollama_thread.search_sources.connect(self.on_search_sources)
        self.ollama_thread.content_started.connect(self.on_content_started)
        self.ollama_thread.image_processing.connect(self.on_image_processing)
        self.ollama_thread.image_description.connect(self.on_image_description)
        self.ollama_thread.error_received.connect(self.handle_error)
        self.ollama_thread.finished.connect(self.on_generation_finished)
        self.ollama_thread.conversation_id_received.connect(
            self.on_conversation_id_received
        )
        self.ollama_thread.start()
        print("OllamaWorker started")

    def on_conversation_id_received(self, conversation_id: int):
        print(f"Received conversation_id: {conversation_id}")
        self.parent.conversation_id = conversation_id

    def handle_command(self, command_text: str):
        parts = command_text.split(" ", 1)
        cmd = parts[0]
        args = parts[1].strip() if len(parts) > 1 else ""

        chat_display = self.parent.ui.response_display

        if cmd == "/help":
            chat_display.clear()
            help_text = """
            <h3>📚 HƯỚNG DẪN SỬ DỤNG FourT AI</h3>
            <ul>
                <li><b>/new</b>: Bắt đầu cuộc hội thoại mới</li>
                <li><b>/history</b>: Xem lịch sử hội thoại</li>
                <li><b>/load &lt;id&gt;</b>: Tải lại cuộc hội thoại</li>
                <li><b>/file &lt;path&gt;</b>: Đính kèm file</li>
                <li><b>/clearfile</b>: Gỡ file đính kèm</li>
                <li><b>/clear</b>: Xóa màn hình chat</li>
                <li><b>/delete</b>: Xóa cuộc hội thoại hiện tại</li>
                <li><b>/delete_all</b>: Xóa tất cả hội thoại</li>
                <li><b>/logout</b>: Đăng xuất</li>
            </ul>
            """
            chat_display.append(help_text)

        elif cmd == "/clear":
            chat_display.clear()

        elif cmd == "/new":
            self.parent.conversation_id = None
            chat_display.clear()
            chat_display.append("<i>Đã bắt đầu cuộc hội thoại mới.</i>")

        elif cmd == "/logout":
            import os
            import sys  # Added import for sys

            if os.path.exists("token.txt"):
                os.remove("token.txt")
            sys.exit(0)

        elif cmd == "/file":
            import os  # Added import for os

            if os.path.exists(args):
                # Logic to attach file (needs implementation in parent or here)
                # For now just show message
                chat_display.append(
                    f"<i>Đã đính kèm file: {args} (Chưa hỗ trợ gửi file qua command này trong UI)</i>"
                )
            else:
                chat_display.append(
                    f"<span style='color:red'>File không tồn tại: {args}</span>"
                )

        elif cmd == "/history":
            self.start_command_worker("history")

        elif cmd == "/load":
            if args.isdigit():
                self.start_command_worker("load", conversation_id=int(args))
            else:
                chat_display.append("<span style='color:red'>ID không hợp lệ.</span>")

        elif cmd == "/delete":
            if getattr(self.parent, "conversation_id", None):
                self.start_command_worker(
                    "delete", conversation_id=self.parent.conversation_id
                )
            else:
                chat_display.append(
                    "<span style='color:yellow'>Chưa có cuộc hội thoại nào để xóa.</span>"
                )

        elif cmd == "/delete_all":
            self.start_command_worker("delete_all")

        else:
            chat_display.append(
                f"<span style='color:yellow'>Lệnh không xác định: {cmd}</span>"
            )

        # Maximize height for commands
        self.parent.ui.adjust_window_height(staged=False)

    def start_command_worker(self, command, **kwargs):
        from command_worker import CommandWorker

        if (
            hasattr(self, "cmd_worker")
            and self.cmd_worker
            and self.cmd_worker.isRunning()
        ):
            self.cmd_worker.quit()
            self.cmd_worker.wait()

        token = self.parent.token
        base_url = "http://localhost:8000"  # Should be configurable

        self.cmd_worker = CommandWorker(command, base_url, token, **kwargs)
        self.cmd_worker.result_ready.connect(self.handle_command_result)
        self.cmd_worker.error_occurred.connect(self.handle_command_error)
        self.cmd_worker.start()

    def handle_command_result(self, result):
        chat_display = self.parent.ui.response_display

        if result["type"] == "history":
            data = result["data"]
            if not data:
                chat_display.append("<i>Chưa có cuộc hội thoại nào.</i>")
            else:
                html = "<b>Danh sách cuộc hội thoại:</b><br>"
                for conv in data:
                    html += f"- ID: {conv['id']} (Tạo lúc: {conv['created_at']})<br>"
                chat_display.append(html)

        elif result["type"] == "load":
            data = result["data"]
            conv_id = result["conversation_id"]
            self.parent.conversation_id = conv_id
            chat_display.clear()
            chat_display.append(f"<i>Đang xem cuộc hội thoại #{conv_id}</i><br>")
            for msg in data:
                if msg["role"] == "user":
                    chat_display.append(f"<br><b>&gt;&gt;&gt; {msg['content']}</b><br>")
                else:
                    content = markdown.markdown(msg["content"])
                    chat_display.append(f"{content}<br>")

        elif result["type"] == "delete":
            self.parent.conversation_id = None
            chat_display.clear()
            chat_display.append("<i>Đã xóa cuộc hội thoại hiện tại.</i>")

        elif result["type"] == "delete_all":
            self.parent.conversation_id = None
            chat_display.clear()
            chat_display.append("<i>Đã xóa tất cả cuộc hội thoại.</i>")

        # Maximize height after command result
        self.parent.ui.adjust_window_height(staged=False)

    def handle_command_error(self, error_msg):
        self.parent.ui.response_display.append(
            f"<span style='color:red'>{error_msg}</span>"
        )

    def stop_worker(self):
        if self.ollama_thread and self.ollama_thread.isRunning():
            print("Stopping OllamaWorker")
            self.ollama_thread.stop()
            self.ollama_thread.wait()
            self.ollama_thread.deleteLater()
            self.ollama_thread = None
        self.parent.waiting_for_response = False
        self.parent.ui.send_stop_button.set_running(False)
        self.parent.spinner_logic.reset_to_idle()
        self.parent.ui.input_box.setEnabled(True)
        self.parent.ui.input_box.setPlaceholderText("Nhập tin nhắn hoặc /help...")
        self.parent.ui.thinking_widget.hide()
        self.parent.ui.response_display.append("\n[Đã dừng phản hồi]")

    def extract_image_from_input(self):
        return None

    def _buffer_chunk(self, chunk: str) -> None:
        self.chunk_buffer += chunk
        if not self.buffer_timer.isActive():
            self.buffer_timer.start()

    def _buffer_thinking(self, thinking: str) -> None:
        if thinking and thinking.strip():
            # Loại bỏ dòng trống và thêm xuống dòng nếu cần
            thinking = thinking.strip()
            if self.thinking_buffer and not self.thinking_buffer.endswith("\n"):
                self.thinking_buffer += "\n"
            self.thinking_buffer += thinking + "\n"
            if not self.buffer_timer.isActive():
                self.buffer_timer.start()

    def _flush_buffer(self) -> None:
        if not self.chunk_buffer and not self.thinking_buffer:
            return

        self.parent.spinner_logic.start_responding()

        if self.thinking_buffer:
            self.full_thinking_md += self.thinking_buffer
            if self.full_thinking_md.strip():
                self.parent.ui.thinking_widget.show()
                html_content = markdown.markdown(
                    self.full_thinking_md,
                    extensions=["fenced_code", "tables", "codehilite"],
                )
                self.parent.ui.thinking_display.setHtml(
                    f'<div style="padding: 10px;">{html_content}</div>'
                )
                # Cuộn tự động đến cuối
                cursor = self.parent.ui.thinking_display.textCursor()
                cursor.movePosition(QTextCursor.End)
                self.parent.ui.thinking_display.setTextCursor(cursor)
                self.parent.ui.thinking_display.ensureCursorVisible()
                # Chỉ toggle nếu thinking_display chưa mở
                if self.parent.ui.thinking_display.height() == 0:
                    self.parent.ui.toggle_thinking(show_full_content=False)
            else:
                self.parent.ui.thinking_widget.hide()

        if self.chunk_buffer:
            self.parent.full_response_md += self.chunk_buffer
            html_content = markdown.markdown(
                self.parent.full_response_md,
                extensions=["fenced_code", "tables", "codehilite"],
            )
            wrapped_html = f'<div style="padding: 15px 10px;">{html_content}</div>'
            self.parent.ui.response_display.setHtml(wrapped_html)

            if not self.parent.user_scrolling and self.parent.full_response_md:
                cursor = self.parent.ui.response_display.textCursor()
                cursor.movePosition(QTextCursor.End)
                self.parent.ui.response_display.setTextCursor(cursor)
                self.parent.ui.response_display.ensureCursorVisible()

        self.chunk_buffer = ""
        self.thinking_buffer = ""

        # Throttle window resizing during streaming
        import time

        current_time = time.time() * 1000
        if current_time - self.last_resize_time > self.resize_throttle_ms:
            self.parent.ui.adjust_window_height(staged=True)
            self.last_resize_time = current_time

    def on_search_started(self, query: str):
        """Display search started message"""
        # Hide spinner so message is visible
        self.parent.spinner_logic.reset_to_idle()

        if query and query.strip():
            search_msg = (
                f'<div style="'
                f"background-color: rgba(255, 255, 255, 0.05); "
                f"color: white; "
                f"font-weight: bold; "
                f"padding: 15px 20px; "
                f"border: 1px solid rgba(255, 255, 255, 0.2); "
                f"border-bottom: 4px solid rgba(0, 0, 0, 0.5); "
                f"border-radius: 12px; "
                f"margin: 15px 0; "
                f"font-size: 14px;"
                f'">'
                f"🔍 Đang tìm kiếm: {query.strip()}..."
                f"</div>"
            )
            self.parent.ui.response_display.append(search_msg)

    def on_search_complete(self, data: dict):
        """Display search completion message"""
        # User requested to remove "Found X results" message
        pass

    def on_search_sources(self, sources_json: str):
        try:
            self.parent.sources_data = json.loads(sources_json)

            # Styled container for search results
            sources_html = (
                '<div style="margin: 15px 0 10px 0; padding: 12px; '
                'background-color: #252526; border-radius: 8px; border: 1px solid #3e3e42;">'
                '<div style="font-weight: bold; color: #4ec9b0; margin-bottom: 8px; '
                'font-size: 13px; font-family: Segoe UI, sans-serif;">'
                "🔍 KẾT QUẢ TÌM KIẾM"
                "</div>"
                '<table style="width: 100%; border-collapse: collapse;">'
            )

            for source in self.parent.sources_data:
                try:
                    domain = urlparse(source["url"]).netloc
                except Exception:
                    domain = "External Link"

                sources_html += (
                    f"<tr><td style='padding: 6px 0; border-bottom: 1px solid #333333;'>"
                    f"<a href='{source['url']}' style='color: #ce9178; text-decoration: none; "
                    f"font-weight: 600; font-size: 13px;'>{source['title']}</a>"
                    f"<div style='font-size: 11px; color: #858585; margin-top: 2px;'>🔗 {domain}</div>"
                    f"</td></tr>"
                )
            sources_html += "</table></div>"

            current_html = self.parent.ui.response_display.toHtml()
            body_start = current_html.find("<body>")
            body_end = current_html.find("</body>")
            if body_start != -1 and body_end != -1:
                body_content = current_html[body_start + 6 : body_end]
                new_html = (
                    current_html[: body_start + 6]
                    + sources_html
                    + body_content
                    + current_html[body_end:]
                )
                self.parent.ui.response_display.setHtml(new_html)
                print("Sources HTML appended")
                if not self.parent.user_scrolling:
                    cursor = self.parent.ui.response_display.textCursor()
                    cursor.movePosition(QTextCursor.End)
                    self.parent.ui.response_display.setTextCursor(cursor)
                    self.parent.ui.response_display.ensureCursorVisible()
            else:
                print("Could not find <body> tags in current HTML")
        except json.JSONDecodeError as e:
            print(f"Error parsing sources JSON: {e}")
        except Exception as e:
            print(f"Error processing sources: {e}")

    def on_content_started(self):
        self.parent.spinner_logic.start_thinking()

    def on_image_processing(self):
        self.parent.spinner_logic.start_thinking()

    def on_image_description(self, description: str):
        self.parent.spinner_logic.start_thinking()

    def on_scroll_changed(self, value: int) -> None:
        scroll_bar = self.parent.ui.scroll_area.verticalScrollBar()
        max_value = scroll_bar.maximum()
        scroll_threshold = max(20, self.parent.ui.scroll_area.height() // 4)
        self.parent.user_scrolling = max_value - value > scroll_threshold
        print(
            f"Scroll changed, user_scrolling: {self.parent.user_scrolling}, value: {value}, max: {max_value}"
        )
        self.parent.last_scroll_value = value

    def on_screenshot_clicked(self):
        print("Bắt đầu chụp hình")
        self.parent.hide()
        self.screenshot_overlay = ScreenshotOverlay()
        self.screenshot_overlay.screenshot_captured.connect(self.on_screenshot_captured)
        self.screenshot_overlay.cancelled.connect(self.on_screenshot_cancelled)
        self.screenshot_overlay.show()

    def on_screenshot_captured(self, pixmap):
        print("Screenshot đã được chụp")
        self.parent.show()
        self.parent.raise_()
        self.parent.activateWindow()
        self.parent.show_screenshot_preview(pixmap)

    def on_screenshot_cancelled(self):
        print("Chụp hình bị hủy")
        self.parent.show()
        self.parent.raise_()
        self.parent.activateWindow()

    def handle_error(self, error_message):
        self.parent.ui.input_box.setEnabled(True)
        self.parent.ui.input_box.setPlaceholderText("Nhập tin nhắn hoặc /help...")
        self.parent.waiting_for_response = False
        self.parent.spinner_logic.reset_to_idle()
        self.parent.ui.response_display.setHtml(
            f'<div style="padding: 15px 10px; color: #f44336;">{error_message}</div>'
        )
        self.parent.ui.thinking_widget.hide()
        if self.ollama_thread:
            if self.ollama_thread.isRunning():
                self.ollama_thread.quit()
                self.ollama_thread.wait()
            self.ollama_thread.deleteLater()
            self.ollama_thread = None
        self.parent.ui.send_stop_button.set_running(False)

    def on_generation_finished(self):
        print("Generation finished")
        self.parent.ui.input_box.setEnabled(True)
        self.parent.ui.input_box.setPlaceholderText("Nhập tin nhắn hoặc /help...")
        self.parent.ui.input_box.clear()
        self.parent.waiting_for_response = False
        self.parent.spinner_logic.reset_to_idle()
        # Mở thinking widget hoàn toàn sau khi generation finished
        if self.parent.ui.thinking_widget.isVisible() and self.full_thinking_md.strip():
            self.parent.ui.toggle_thinking(show_full_content=True)
        else:
            self.parent.ui.thinking_widget.hide()
        # Tăng height tối đa sau khi response hoàn tất
        self.parent.ui.adjust_window_height(staged=False)
        if self.ollama_thread:
            if self.ollama_thread.isRunning():
                self.ollama_thread.quit()
                self.ollama_thread.wait()
            self.ollama_thread.deleteLater()
            self.ollama_thread = None
        self.parent.ui.send_stop_button.set_running(False)
