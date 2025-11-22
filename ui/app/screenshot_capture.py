# screenshot_capture.py
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor
from PySide6.QtCore import Qt, QRect, QPoint, Signal, QByteArray, QBuffer, QIODevice


class ScreenshotOverlay(QWidget):
    screenshot_captured = Signal(QPixmap)  # Signal phát ra khi có screenshot
    cancelled = Signal()  # Signal phát ra khi hủy bỏ

    def __init__(self, parent=None):
        super().__init__(parent)
        # overlay fullscreen
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 0.3);")

        # Lấy screenshot toàn màn hình
        screen = QApplication.primaryScreen()
        if screen:
            self.full_screenshot = screen.grabWindow(0)
        else:
            self.full_screenshot = QPixmap()

        self.start_pos = None
        self.end_pos = None
        self.drawing = False
        self.selection_rect = QRect()
        self._pending_pixmap = None  # lưu ảnh tạm khi đã chọn vùng, chờ xác nhận

        # Tạo preview widget (cửa sổ riêng biệt -> tránh overlay chặn event)
        self.preview_widget = QWidget(None)  # None = top-level window
        self.preview_widget.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
        )
        self.preview_widget.setAttribute(Qt.WA_TranslucentBackground)
        self.preview_widget.setStyleSheet("background-color: rgba(0, 0, 0, 0.85); border-radius: 10px;")
        self.preview_widget.setFixedSize(200, 150)

        preview_layout = QVBoxLayout(self.preview_widget)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("color: white; font-size: 12px;")
        preview_layout.addWidget(self.preview_label)

        button_layout = QHBoxLayout()
        self.confirm_button = QPushButton("Xác nhận")
        self.cancel_button = QPushButton("Hủy")
        self.confirm_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)

        self.confirm_button.clicked.connect(self.confirm_selection)
        self.cancel_button.clicked.connect(self.cancel_selection)

        button_layout.addWidget(self.confirm_button)
        button_layout.addWidget(self.cancel_button)
        preview_layout.addLayout(button_layout)

        self.preview_widget.hide()

    def showEvent(self, event):
        super().showEvent(event)
        # Set kích thước full màn hình
        screen = QApplication.primaryScreen()
        if screen:
            self.setGeometry(screen.geometry())
        # ẩn preview khi overlay vừa hiện
        self.preview_widget.hide()
        self._pending_pixmap = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Vẽ overlay màu xám
        overlay_color = QColor(0, 0, 0, 100)
        painter.fillRect(self.rect(), overlay_color)

        # Nếu đang chọn vùng, vẽ vùng chọn
        if not self.selection_rect.isNull():
            painter.setPen(QPen(Qt.white, 2))
            painter.setBrush(QColor(255, 255, 255, 30))
            painter.drawRect(self.selection_rect)

            # Hiển thị kích thước vùng chọn
            size_text = f"{self.selection_rect.width()} x {self.selection_rect.height()}"
            painter.setPen(Qt.white)
            painter.drawText(self.selection_rect.topLeft() + QPoint(5, -5), size_text)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
            self.drawing = True
            self.selection_rect = QRect()
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing and self.start_pos:
            self.end_pos = event.pos()
            self.selection_rect = QRect(self.start_pos, self.end_pos).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            if not self.selection_rect.isNull() and self.selection_rect.width() > 10 and self.selection_rect.height() > 10:
                self.show_preview()
            else:
                self.selection_rect = QRect()
                self.update()

    def show_preview(self):
        """
        Cắt ảnh từ vùng chọn, lưu tạm vào self._pending_pixmap
        Hiển thị preview_widget để người dùng xác nhận/hủy
        """
        if not self.selection_rect.isNull():
            # Cắt ảnh từ vùng chọn
            cropped_pixmap = self.full_screenshot.copy(self.selection_rect)
            # Lưu tạm để confirm dùng sau
            self._pending_pixmap = cropped_pixmap

            # Scale ảnh để hiển thị trong preview (giữ tỷ lệ)
            scaled_pixmap = cropped_pixmap.scaled(180, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # Hiển thị ảnh trong preview
            self.preview_label.setPixmap(scaled_pixmap)
            self.preview_label.setText("")

            # Hiển thị preview widget ở góc dưới bên phải
            screen_geo = QApplication.primaryScreen().geometry()
            preview_x = screen_geo.right() - 220
            preview_y = screen_geo.bottom() - 170
            self.preview_widget.move(preview_x, preview_y)
            self.preview_widget.show()
            # đảm bảo preview thực sự ở trên cùng và có thể nhận event
            self.preview_widget.raise_()
            self.preview_widget.activateWindow()

            # Ẩn vùng chọn (vẽ vùng chọn xong rồi) - nhưng ảnh đã được lưu trong _pending_pixmap
            self.selection_rect = QRect()
            self.update()

    def confirm_selection(self):
        """
        Khi người dùng bấm Xác nhận: emit ảnh đã lưu trong _pending_pixmap
        """
        if self._pending_pixmap is not None and not self._pending_pixmap.isNull():
            pixmap = self._pending_pixmap
            # reset state
            self._pending_pixmap = None
            self.preview_widget.hide()
            self.hide()
            self.screenshot_captured.emit(pixmap)
        else:
            # không có ảnh tạm -> như hủy
            self.cancel_selection()

    def cancel_selection(self):
        # Hủy bỏ ảnh tạm nếu có
        self._pending_pixmap = None
        self.preview_widget.hide()
        self.hide()
        self.cancelled.emit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.cancel_selection()
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.confirm_selection()
        else:
            super().keyPressEvent(event)

    @staticmethod
    def pixmap_to_base64(pixmap: QPixmap) -> str:
        """Chuyển QPixmap sang base64 string"""
        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QIODevice.WriteOnly)
        pixmap.save(buffer, "PNG")
        return byte_array.toBase64().data().decode()
