# ui_components.py
# -*- coding: utf-8 -*-
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTextEdit,
    QFrame,
    QScrollArea,
    QTextBrowser,
    QPushButton,
    QHBoxLayout,
    QLabel,
    QGraphicsDropShadowEffect,
)
from PySide6.QtCore import (
    Qt,
    QPropertyAnimation,
    QRect,
    QEasingCurve,
    QParallelAnimationGroup,
)
from PySide6.QtGui import QColor
from send_stop_button import SendStopButton
from minimize_button import MinimizeButton
import time


class UIComponents:
    def __init__(self, parent):
        self.parent = parent
        self.main_container = None
        self.container_layout = None
        self.main_frame = None
        self.input_box = None
        self.scroll_area = None
        self.response_display = None
        self.button_widget = None
        self.screenshot_button = None
        self.send_stop_button = None
        self.minimize_button = None
        self.preview_widget = None
        self.icon_label = None
        self.name_label = None
        self.size_label = None
        self.delete_button = None
        self.thinking_widget = None
        self.thinking_display = None
        self.toggle_button = None

        # Khởi tạo các đối tượng animation
        self.height_animation = None
        self.input_box_animation_group = None
        self.thinking_animation = None

    def setup_ui(self):
        self.parent.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.parent.setAttribute(Qt.WA_TranslucentBackground)
        self.parent.setWindowTitle("FourT Assistant")
        self.parent.setFixedWidth(600)
        self.parent.hide()

        self.parent.layout = QVBoxLayout(self.parent)
        self.parent.layout.setContentsMargins(0, 0, 0, 0)

        self.main_container = QWidget()
        self.main_container.setObjectName("mainContainer")
        self.container_layout = QVBoxLayout(self.main_container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.container_layout.setSpacing(10)

        self.main_frame = QFrame(self.main_container)
        self.main_frame.setObjectName("mainFrame")

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(QColor(0, 0, 0, 160))
        self.main_frame.setGraphicsEffect(shadow)

        frame_layout = QVBoxLayout(self.main_frame)
        frame_layout.setSpacing(10)

        self.preview_widget = QWidget(self.main_frame)
        self.preview_widget.setObjectName("previewWidget")
        preview_layout = QHBoxLayout(self.preview_widget)
        preview_layout.setContentsMargins(10, 5, 10, 5)
        preview_layout.setSpacing(10)

        self.icon_label = QLabel()
        self.icon_label.setFixedSize(40, 40)
        self.icon_label.setScaledContents(True)
        preview_layout.addWidget(self.icon_label)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        self.name_label = QLabel("Screenshot.png")
        self.name_label.setStyleSheet("color: #e0e0e0; font-size: 12px;")
        self.size_label = QLabel("0x0")
        self.size_label.setStyleSheet("color: #a0a0a0; font-size: 10px;")
        info_layout.addWidget(self.name_label)
        info_layout.addWidget(self.size_label)
        preview_layout.addLayout(info_layout)

        preview_layout.addStretch()

        self.delete_button = QPushButton("✕")
        self.delete_button.setObjectName("deleteButton")
        self.delete_button.setFixedSize(20, 20)
        self.delete_button.setCursor(Qt.PointingHandCursor)
        self.delete_button.clicked.connect(self.delete_screenshot)
        preview_layout.addWidget(self.delete_button)

        frame_layout.addWidget(self.preview_widget)
        self.preview_widget.hide()

        self.input_box = QTextEdit(self.main_frame)
        self.input_box.setObjectName("inputBox")
        self.input_box.setPlaceholderText("Nhập tin nhắn hoặc /help")
        self.input_box.setAcceptRichText(False)
        self.input_box.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.input_box.textChanged.connect(self.adjust_input_box_height)
        self.input_box.keyPressEvent = self.parent.handle_key_press
        self.adjust_input_box_height()
        frame_layout.addWidget(self.input_box)

        self.thinking_widget = QWidget(self.main_frame)
        self.thinking_widget.setObjectName("thinkingWidget")
        thinking_layout = QVBoxLayout(self.thinking_widget)
        thinking_layout.setContentsMargins(10, 5, 10, 5)
        thinking_layout.setSpacing(5)

        self.toggle_button = QPushButton("Suy luận ▼")
        self.toggle_button.setObjectName("toggleButton")
        self.toggle_button.setFixedHeight(30)
        self.toggle_button.setCursor(Qt.PointingHandCursor)
        self.toggle_button.clicked.connect(self.toggle_thinking)
        thinking_layout.addWidget(self.toggle_button)

        self.thinking_display = QTextBrowser()
        self.thinking_display.setObjectName("thinkingDisplay")
        self.thinking_display.setOpenExternalLinks(True)
        self.thinking_display.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.thinking_display.setFixedHeight(0)
        thinking_layout.addWidget(self.thinking_display)

        frame_layout.addWidget(self.thinking_widget)
        self.thinking_widget.hide()

        self.scroll_area = QScrollArea(self.main_frame)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setObjectName("scrollArea")
        self.scroll_area.verticalScrollBar().valueChanged.connect(
            self.parent.on_scroll_changed
        )
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.response_display = QTextBrowser(self.scroll_area)
        self.response_display.setObjectName("responseDisplay")
        self.response_display.setOpenExternalLinks(True)
        self.scroll_area.setWidget(self.response_display)
        self.response_display.setText("Bạn cần mình giúp gì không?")

        frame_layout.addWidget(self.scroll_area)

        self.container_layout.addWidget(self.main_frame)

        self.button_widget = QWidget()
        button_layout = QHBoxLayout(self.button_widget)
        button_layout.setContentsMargins(5, 0, 0, 0)
        button_layout.setSpacing(10)
        button_layout.setAlignment(Qt.AlignLeft)

        self.screenshot_button = QPushButton("📷", self.parent)
        self.screenshot_button.setObjectName("screenshotButton")
        self.screenshot_button.setFixedSize(60, 30)
        self.screenshot_button.clicked.connect(self.parent.on_screenshot_clicked)
        self.screenshot_button.setCursor(Qt.PointingHandCursor)
        button_layout.addWidget(self.screenshot_button)

        self.send_stop_button = SendStopButton(self.button_widget)
        self.send_stop_button.set_running(False)
        button_layout.addWidget(self.send_stop_button)

        button_layout.addStretch()

        self.minimize_button = MinimizeButton(self.button_widget)
        self.minimize_button.minimize_clicked.connect(self.parent.minimize_to_tray)
        button_layout.addWidget(self.minimize_button)

        frame_layout.addWidget(self.button_widget)

        self.parent.layout.addWidget(self.main_container)
        self.parent.setLayout(self.parent.layout)

        self.apply_stylesheet()

    def apply_stylesheet(self):
        stylesheet = """
        #mainContainer {
            background-color: transparent;
        }
        #mainFrame {
            background-color: rgba(28, 29, 35, 0.85);
            border: 1px solid #505050;
            border-radius: 20px;
        }
        #previewWidget {
            background-color: rgba(28, 29, 35, 0.85);
            border: 1px solid #505050;
            border-radius: 9px;
        }
        #previewWidget:hover {
            background-color: #3a3b45;
            border: 1px solid #61afef;
        }
        #thinkingWidget {
              background-color: transparent;
              border-radius: 9px;
              max-height: 40px;
              transition: max-height 0.2s ease-in-out;
        }
        #thinkingWidget.expanded {
            max-height: 250px;
        }
        #thinkingWidget:hover {
            background-color: transparent;
            border: 1px solid #61afef;
        }
        #toggleButton {
            background-color: transparent;
            border: none;
            color: #e0e0e0;
            font-size: 14px;
            text-align: left;
            padding: 5px;
        }
        #toggleButton:hover {
            background-color: transparent;
        }
        #toggleButton:pressed {
            background-color: transparent;
        }
        #thinkingDisplay {
            background-color: #2c2d35;
            border: none;
            color: #e0e0e0;
            font-size: 14px;
        }
        #thinkingDisplay a {
            color: #61afef;
            text-decoration: none;
        }
        #thinkingDisplay a:hover {
            text-decoration: underline;
        }
        #inputBox {
            background-color: #2c2d35;
            border: 1px solid #505050;
            border-radius: 10px;
            color: #e0e0e0;
            font-size: 14px;
            padding: 10px;
        }
        #inputBox:focus {
            background-color: #2c2d35;
            border: 2px solid #61afef;
        }
        #inputBox::placeholder {
            color: #a0a0a0;
        }
        #scrollArea, #scrollArea > QWidget > QWidget {
            border: none;
            background: transparent;
        }
        #responseDisplay {
            background-color: transparent;
            color: #e0e0e0;
            font-size: 14px;
            border: none;
        }
        #screenshotButton {
            background-color: rgba(28, 29, 35, 0.85);
            border: 1px solid #505050;
            border-radius: 5px;
            color: #e0e0e0;
            font-size: 14px;
            padding-bottom: 2px;
        }
        #screenshotButton:hover {
            background-color: #3a3b45;
            border: 1px solid #61afef;
        }
        #screenshotButton:pressed {
            background-color: #1a1b25;
        }
        #deleteButton {
            background-color: #f44336;
            border: none;
            border-radius: 10px;
            color: #e0e0e0;
            font-size: 12px;
            min-width: 20px;
            max-width: 20px;
            min-height: 20px;
            max-height: 20px;
            padding-bottom: 2px;
        }
        #deleteButton:hover {
            background-color: #d32f2f;
        }
        QScrollBar:vertical {
            width: 0px;
        }
        QScrollBar:horizontal {
            height: 0px;
        }
        """
        self.parent.setStyleSheet(stylesheet)

        # Apply HTML-specific stylesheet for QTextBrowser content
        html_stylesheet = """
        a {
            color: #61afef;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        table {
            border-collapse: collapse;
            margin: 1em 0;
            width: auto;
            min-width: 100%;
            border: 1px solid #606060;
        }
        th, td {
            border: 1px solid #606060;
            padding: 8px;
            text-align: left;
            white-space: nowrap;
        }
        th {
            background-color: #3a3b45;
            color: #e0e0e0;
            font-weight: bold;
        }
        td {
            background-color: #2c2d35;
        }
        .codehilite {
            background: #2c2d35;
            border-radius: 5px;
            padding: 10px;
            font-size: 13px;
            margin: 1em 0;
        }
        .codehilite pre {
            margin: 0;
            white-space: pre-wrap;
        }
        .codehilite .k { color: #c678dd; }
        .codehilite .s2 { color: #98c379; }
        .codehilite .nf { color: #61afef; }
        .codehilite .mi { color: #d19a66; }
        .codehilite .n { color: #abb2bf; }
        .codehilite .p { color: #abb2bf; }
        .codehilite .o { color: #56b6c2; }
        .codehilite .nb { color: #d19a66; }
        .codehilite .c1 { color: #7f848e; font-style: italic; }
        """
        self.response_display.document().setDefaultStyleSheet(html_stylesheet)

        # Disable scrollbars on response_display (content) to hide them
        self.response_display.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.response_display.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def mouse_press_event(self, event):
        if event.button() == Qt.LeftButton:
            self.parent.dragging = True
            self.parent.drag_position = (
                event.globalPosition().toPoint() - self.parent.pos()
            )
            event.accept()

    def mouse_move_event(self, event):
        if self.parent.dragging:
            self.parent.move(
                event.globalPosition().toPoint() - self.parent.drag_position
            )
            event.accept()

    def mouse_release_event(self, event):
        if event.button() == Qt.LeftButton:
            self.parent.dragging = False
            event.accept()

    def delete_screenshot(self):
        self.preview_widget.hide()
        self.parent.current_screenshot_base64 = None
        self.icon_label.clear()
        self.parent.adjust_window_height()

    def toggle_thinking(self, show_full_content=False):
        if (
            self.thinking_animation
            and self.thinking_animation.state() == QParallelAnimationGroup.Running
        ):
            self.thinking_animation.stop()

        self.thinking_animation = QParallelAnimationGroup(self.parent)
        current_height = self.parent.height()
        is_expanding = self.thinking_display.height() == 0

        # Toggle expanded class
        if is_expanding:
            self.thinking_widget.setProperty("class", "expanded")
        else:
            self.thinking_widget.setProperty("class", "")
        self.thinking_widget.style().unpolish(self.thinking_widget)
        self.thinking_widget.style().polish(self.thinking_widget)

        if not is_expanding:
            # Ẩn thinking_display và giữ nguyên window height
            max_anim = QPropertyAnimation(self.thinking_display, b"maximumHeight")
            max_anim.setDuration(200)
            max_anim.setStartValue(self.thinking_display.height())
            max_anim.setEndValue(0)
            max_anim.setEasingCurve(QEasingCurve.InOutQuad)

            min_anim = QPropertyAnimation(self.thinking_display, b"minimumHeight")
            min_anim.setDuration(200)
            min_anim.setStartValue(self.thinking_display.height())
            min_anim.setEndValue(0)
            min_anim.setEasingCurve(QEasingCurve.InOutQuad)

            self.thinking_animation.addAnimation(max_anim)
            self.thinking_animation.addAnimation(min_anim)
            self.toggle_button.setText("Suy luận ▼")
        else:
            # Hiển thị thinking_display
            doc_height = self.thinking_display.document().size().toSize().height()
            # Nếu show_full_content=True, mở toàn bộ nội dung, nếu không thì mở tối thiểu
            target_height = (
                min(doc_height + 20, 200)
                if show_full_content and doc_height > 0
                else 100
            )

            max_anim = QPropertyAnimation(self.thinking_display, b"maximumHeight")
            max_anim.setDuration(200)
            max_anim.setStartValue(0)
            max_anim.setEndValue(target_height)
            max_anim.setEasingCurve(QEasingCurve.InOutQuad)

            min_anim = QPropertyAnimation(self.thinking_display, b"minimumHeight")
            min_anim.setDuration(200)
            min_anim.setStartValue(0)
            min_anim.setEndValue(0)  # Giữ minimumHeight = 0
            min_anim.setEasingCurve(QEasingCurve.InOutQuad)

            self.thinking_animation.addAnimation(max_anim)
            self.thinking_animation.addAnimation(min_anim)
            self.toggle_button.setText("Suy luận để cho kết quả tốt hơn")

        # Chỉ gọi adjust_window_height sau khi animation hoàn tất
        self.thinking_animation.finished.connect(
            lambda: self.adjust_window_height(staged=not is_expanding)
        )
        self.thinking_animation.start()

    def adjust_input_box_height(self):
        min_height = 80
        max_height = 150

        doc_height = self.input_box.document().size().toSize().height()
        vertical_padding = 20
        target_height = doc_height + vertical_padding
        final_height = int(max(min_height, min(target_height, max_height)))

        current_height = self.input_box.height()

        if current_height != final_height:
            if (
                self.input_box_animation_group
                and self.input_box_animation_group.state()
                == QParallelAnimationGroup.Running
            ):
                self.input_box_animation_group.stop()

            self.input_box_animation_group = QParallelAnimationGroup(self.parent)

            min_anim = QPropertyAnimation(self.input_box, b"minimumHeight")
            min_anim.setDuration(150)
            min_anim.setStartValue(current_height)
            min_anim.setEndValue(final_height)
            min_anim.setEasingCurve(QEasingCurve.InOutQuad)

            max_anim = QPropertyAnimation(self.input_box, b"maximumHeight")
            max_anim.setDuration(150)
            max_anim.setStartValue(current_height)
            max_anim.setEndValue(final_height)
            max_anim.setEasingCurve(QEasingCurve.InOutQuad)

            self.input_box_animation_group.addAnimation(min_anim)
            self.input_box_animation_group.addAnimation(max_anim)
            self.input_box_animation_group.start()

    def adjust_window_height(self, staged=False):
        doc_height = self.response_display.document().size().toSize().height()
        input_height = self.input_box.height()
        preview_height = (
            self.preview_widget.sizeHint().height()
            if self.preview_widget.isVisible()
            else 0
        )
        button_height = self.button_widget.sizeHint().height()
        container_margins = self.container_layout.contentsMargins()
        container_margin = container_margins.top() + container_margins.bottom()
        frame_margins = self.main_frame.layout().contentsMargins()
        frame_margin = frame_margins.top() + frame_margins.bottom()
        spacing = self.main_frame.layout().spacing()
        response_padding = 20

        thinking_height = 0
        if self.thinking_widget.isVisible():
            if self.thinking_display.height() > 0:
                thinking_height = 40 + self.thinking_display.height()
            else:
                thinking_height = 40

        if staged:
            target_height = self.parent.height()
        else:
            target_height = int(
                input_height
                + doc_height
                + preview_height
                + thinking_height
                + button_height
                + container_margin
                + frame_margin
                + spacing * 3
                + response_padding
            )

        final_height = min(target_height, self.parent.MAX_HEIGHT)

        current_height = self.parent.height()

        if current_height != final_height:
            # Avoid starting new animation if one is already running
            if (
                self.height_animation
                and self.height_animation.state() == QPropertyAnimation.Running
            ):
                # Only restart if the target is significantly different
                if abs(self.height_animation.endValue().height() - final_height) < 10:
                    return
                self.height_animation.stop()

            self.height_animation = QPropertyAnimation(self.parent, b"geometry")
            self.height_animation.setDuration(150)
            self.height_animation.setEasingCurve(QEasingCurve.InOutQuad)

            current_geometry = self.parent.geometry()
            target_geometry = QRect(
                current_geometry.x(),
                current_geometry.y(),
                current_geometry.width(),
                final_height,
            )

            self.height_animation.setStartValue(current_geometry)
            self.height_animation.setEndValue(target_geometry)
            self.height_animation.start()
