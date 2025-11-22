# chat_window.py
# -*- coding: utf-8 -*-
from PySide6.QtWidgets import QWidget, QVBoxLayout, QApplication
from PySide6.QtCore import Qt, QPoint, Signal
from PySide6.QtGui import QTextCursor, QPixmap
from ui_components import UIComponents
from chat_logic import ChatLogic
from spinner_logic import SpinnerLogic
from screenshot_capture import ScreenshotOverlay
from PySide6.QtCore import QByteArray, QBuffer, QIODevice


class ChatWindow(QWidget):
    MAX_HEIGHT = 500
    toSearch = Signal()
    toThinking = Signal()
    toResponding = Signal()
    toIdle = Signal()

    def __init__(self):
        super().__init__()
        self._is_stable = False
        self.full_response_md = ""
        self.dragging = False
        self.drag_position = QPoint()
        self.waiting_for_response = False
        self.user_scrolling = False
        self.last_scroll_value = 0
        self.sources_data = []
        self.current_screenshot_base64 = None
        self.conversation_id = None
        self.tray_manager = None  # Thêm attribute để access tray

        self.ui = UIComponents(self)
        self.chat_logic = ChatLogic(self)
        self.spinner_logic = SpinnerLogic(self)

        self.init_ui()

    def center_and_show(self):
        """Hiển thị window ở giữa màn hình"""
        screen = QApplication.primaryScreen().geometry()
        window_width = self.width()
        window_height = self.height()
        center_x = (screen.width() - window_width) // 2
        center_y = (screen.height() - window_height) // 2
        self.move(center_x, center_y)
        self.show()

    def init_ui(self):
        self.ui.setup_ui()
        self.chat_logic.setup_connections()

    def minimize_to_tray(self):
        """Thu nhỏ window về tray với notification"""
        self.hide()
        if self.tray_manager:
            self.tray_manager.show_message(
                "4T Assistant", "Đã thu nhỏ. Click icon để mở lại.", duration=2000
            )

    def focusInEvent(self, event):
        self._is_stable = True
        super().focusInEvent(event)

    def showEvent(self, event):
        self._is_stable = False
        super().showEvent(event)
        self.ui.input_box.setFocus()

    def focusOutEvent(self, event):
        if self._is_stable and not self.waiting_for_response:
            print("focusOutEvent triggered, hiding window")
            self.hide()
        else:
            print("focusOutEvent ignored due to waiting_for_response or unstable state")
        super().focusOutEvent(event)

    def mousePressEvent(self, event):
        self.ui.mouse_press_event(event)

    def mouseMoveEvent(self, event):
        self.ui.mouse_move_event(event)

    def mouseReleaseEvent(self, event):
        self.ui.mouse_release_event(event)

    def handle_key_press(self, event):
        self.chat_logic.handle_key_press(event)

    def send_prompt(self):
        self.chat_logic.send_prompt()

    def update_response(self, chunk):
        self.chat_logic.update_response(chunk)

    def handle_error(self, error_message):
        self.chat_logic.handle_error(error_message)

    def on_generation_finished(self):
        self.chat_logic.on_generation_finished()

    def on_search_started(self):
        self.chat_logic.on_search_started()

    def on_search_sources(self, sources_json):
        self.chat_logic.on_search_sources(sources_json)

    def on_scroll_changed(self, value):
        self.chat_logic.on_scroll_changed(value)

    def on_screenshot_clicked(self):
        self.chat_logic.on_screenshot_clicked()

    def adjust_window_height(self):
        self.ui.adjust_window_height()

    def apply_stylesheet(self):
        self.ui.apply_stylesheet()

    def pixmap_to_base64(self, pixmap):
        scaled_pixmap = pixmap.scaled(
            40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.ui.icon_label.setPixmap(scaled_pixmap)
        self.ui.name_label.setText("Screenshot.png")
        self.ui.size_label.setText(f"{pixmap.width()}x{pixmap.height()}")
        self.ui.preview_widget.show()
        self.adjust_window_height()

        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QIODevice.WriteOnly)
        pixmap.save(buffer, "PNG")
        base64_str = byte_array.toBase64().data().decode()
        return f"data:image/png;base64,{base64_str}"

    def show_screenshot_preview(self, pixmap):
        self.current_screenshot_base64 = self.pixmap_to_base64(pixmap)
