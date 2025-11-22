# spinner_logic.py
# -*- coding: utf-8 -*-
from asyncio.log import logger
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout
from PySide6.QtStateMachine import QStateMachine, QState
from PySide6.QtGui import QFont


class SpinnerLogic:
    def __init__(self, parent):
        self.parent = parent
        self.state_machine = QStateMachine(parent)
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_index = 0
        self.spinner_timer = None
        self.spinner_label = None
        self.text_label = None
        self.overlay = None
        self.setup_states()

    def setup_states(self):
        """Thiết lập state machine cho các trạng thái spinner"""
        idle_state = QState()
        search_state = QState()
        thinking_state = QState()
        responding_state = QState()

        # Kiểm tra tín hiệu trước khi thêm transition
        if (
            not hasattr(self.parent, "toSearch")
            or not hasattr(self.parent, "toThinking")
            or not hasattr(self.parent, "toResponding")
            or not hasattr(self.parent, "toIdle")
        ):
            logger.error(
                "One or more signals (toSearch, toThinking, toResponding, toIdle) not defined in parent"
            )
            return

        # Transitions sử dụng Signal từ ChatWindow
        idle_state.addTransition(self.parent.toSearch, search_state)
        idle_state.addTransition(self.parent.toThinking, thinking_state)
        search_state.addTransition(self.parent.toThinking, thinking_state)
        search_state.addTransition(self.parent.toResponding, responding_state)
        search_state.addTransition(self.parent.toIdle, idle_state)
        thinking_state.addTransition(self.parent.toSearch, search_state)
        thinking_state.addTransition(self.parent.toResponding, responding_state)
        thinking_state.addTransition(self.parent.toIdle, idle_state)
        responding_state.addTransition(self.parent.toIdle, idle_state)
        responding_state.addTransition(self.parent.toSearch, search_state)
        responding_state.addTransition(self.parent.toThinking, thinking_state)

        # Idle: Không hiển thị spinner
        def enter_idle():
            print("Entered idle state")
            self._hide_spinner()
            self.parent.ui.adjust_window_height()

        idle_state.entered.connect(enter_idle)

        # Search: Hiển thị spinner + "Đang tìm kiếm..."
        def enter_search():
            print("Entered search state")
            self._show_spinner("Đang tìm kiếm...")
            print("Search spinner started")

        search_state.entered.connect(enter_search)

        # Thinking: Hiển thị spinner + "Đang suy nghĩ..."
        def enter_thinking():
            print("Entered thinking state")
            self._show_spinner("Chờ trong giây lát...")
            if self.spinner_timer and not self.spinner_timer.isActive():
                self.spinner_timer.start()
                print("Spinner timer started in thinking state")

        thinking_state.entered.connect(enter_thinking)

        # Responding: Ẩn spinner
        def enter_responding():
            print("Entered responding state")
            self._hide_spinner()
            self.parent.ui.adjust_window_height()

        responding_state.entered.connect(enter_responding)

        self.state_machine.addState(idle_state)
        self.state_machine.addState(search_state)
        self.state_machine.addState(thinking_state)
        self.state_machine.addState(responding_state)
        self.state_machine.setInitialState(idle_state)
        self.state_machine.start()
        print("State machine started")

    def _show_spinner(self, text, query=None):
        """Hiển thị overlay spinner với ký tự ASCII"""
        if self.overlay:
            self._hide_spinner()
            print("Existing spinner hidden before showing new one")

        display_text = text
        if query:
            # Xử lý query để hiển thị trên một dòng và giới hạn độ dài
            one_line_query = query.replace("\n", " ").replace("\r", "").strip()
            max_length = 50  # Số ký tự tối đa
            if len(one_line_query) > max_length:
                one_line_query = one_line_query[: max_length - 3] + "..."
            display_text = f"Đang tìm kiếm {one_line_query}"

        self.parent.ui.scroll_area.setVisible(True)
        scroll_width = max(self.parent.ui.scroll_area.width(), 100)
        scroll_height = max(self.parent.ui.scroll_area.height(), 50)
        print(f"scroll_area set visible, size: {scroll_width}x{scroll_height}")

        self.overlay = QWidget(self.parent.ui.scroll_area)
        self.overlay.setStyleSheet("background: transparent;")
        self.overlay.setGeometry(0, 0, scroll_width, 50)
        self.overlay.raise_()
        print(
            f"Spinner overlay created, geometry: {self.overlay.geometry().width()}x{self.overlay.geometry().height()}, visible: {self.overlay.isVisible()}"
        )

        # Container widget cho spinner và text
        container = QWidget(self.overlay)
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(10, 5, 10, 5)
        container_layout.setSpacing(5)

        # Spinner label
        self.spinner_label = QLabel(self.spinner_chars[self.spinner_index])
        self.spinner_label.setStyleSheet(
            "color: #61afef; font-size: 16px; font-family: 'Courier New', monospace;"
        )
        container_layout.addWidget(self.spinner_label)

        # Text label
        self.text_label = QLabel(display_text)
        self.text_label.setStyleSheet("color: #e0e0e0; font-size: 14px;")
        self.text_label.setWordWrap(False)  # Không wrap text để tránh xuống hàng
        self.text_label.setMinimumWidth(10)  # Đảm bảo có kích thước tối thiểu
        container_layout.addWidget(self.text_label, stretch=1)
        container_layout.addStretch()

        container.setGeometry(0, 0, self.overlay.width(), 50)
        container.show()

        # Điều chỉnh kích thước overlay dựa trên text
        text_width = (
            self.text_label.fontMetrics().boundingRect(self.text_label.text()).width()
            + 20
        )
        spinner_width = self.spinner_label.width()
        total_width = max(text_width + spinner_width + 20, scroll_width)
        self.overlay.setGeometry(0, 0, total_width, 50)
        print(
            f"Text label shown with text: {display_text}, visible: {self.text_label.isVisible()}"
        )

        # Khởi tạo timer nếu chưa có
        if not self.spinner_timer:
            self.spinner_timer = QTimer(self.overlay)
            self.spinner_timer.setInterval(200)  # Tăng interval để giảm tải
            self.spinner_timer.timeout.connect(self._update_spinner)
            print(f"Spinner timer initialized, active: {self.spinner_timer.isActive()}")

        self.spinner_timer.start()
        self.overlay.show()
        self.overlay.raise_()
        print(f"Spinner overlay shown, visible: {self.overlay.isVisible()}")
        self.parent.ui.adjust_window_height()

    def _hide_spinner(self):
        """Ẩn spinner"""
        if self.spinner_timer:
            self.spinner_timer.stop()
            self.spinner_timer.deleteLater()
            self.spinner_timer = None
        if self.spinner_label:
            self.spinner_label.deleteLater()
            self.spinner_label = None
        if self.text_label:
            self.text_label.deleteLater()
            self.text_label = None
        if self.overlay:
            self.overlay.deleteLater()
            self.overlay = None
        print("Spinner hidden and cleaned up")

    def _update_spinner(self):
        """Cập nhật ký tự spinner"""
        if self.spinner_label and self.overlay and self.overlay.isVisible():
            self.spinner_index = (self.spinner_index + 1) % len(self.spinner_chars)
            self.spinner_label.setText(self.spinner_chars[self.spinner_index])

    def start_search(self, query=None):
        """Trigger chuyển sang trạng thái search với query"""
        print(f"Starting search with query: {query}")
        self.parent.toSearch.emit()
        display_text = f"Đang tìm kiếm {query}" if query else "Đang tìm kiếm..."
        self._show_spinner(display_text)

    def update_search_text(self, query: str):
        """Cập nhật text cho spinner khi có query mới"""
        if self.text_label and query:
            one_line_query = query.replace("\n", " ").replace("\r", "").strip()
            max_length = 50
            if len(one_line_query) > max_length:
                one_line_query = one_line_query[: max_length - 3] + "..."
            display_text = f"Đang tìm kiếm {one_line_query}"
            self.text_label.setText(display_text)
            text_width = (
                self.text_label.fontMetrics()
                .boundingRect(self.text_label.text())
                .width()
                + 20
            )
            spinner_width = self.spinner_label.width() if self.spinner_label else 0
            total_width = max(
                text_width + spinner_width + 20, self.parent.ui.scroll_area.width()
            )
            if self.overlay:
                self.overlay.setGeometry(0, 0, total_width, 50)
                print(f"Search text updated: {display_text}")

    def start_thinking(self):
        """Trigger chuyển sang trạng thái thinking"""
        self.parent.toThinking.emit()

    def start_responding(self):
        """Trigger chuyển sang trạng thái responding"""
        self.parent.toResponding.emit()

    def reset_to_idle(self):
        """Trigger chuyển về trạng thái idle"""
        self.parent.toIdle.emit()
