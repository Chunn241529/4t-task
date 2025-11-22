# minimize_button.py
from PySide6.QtWidgets import QPushButton
from PySide6.QtCore import Signal, Qt


class MinimizeButton(QPushButton):
    minimize_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("minimizeButton")
        self.setFixedSize(30, 30)
        self.setCursor(Qt.PointingHandCursor)
        self.setText("↑")

        # Định nghĩa các stylesheet cho từng trạng thái
        self.normal_style = """
            #minimizeButton {
                background: transparent;
                border: none;
                color: #e0e0e0;
                font-size: 14px;
                text-align: center;
                padding: 0;
                padding-bottom: 4px;
            }
        """
        self.hover_style = """
            #minimizeButton {
                background: transparent;
                border: none;
                color: #ff9800;
                font-size: 14px;
                text-align: center;
                padding: 0;
                padding-bottom: 4px;
            }
        """
        self.pressed_style = """
            #minimizeButton {
                background: transparent;
                border: none;
                color: #f57c00;
                font-size: 14px;
                text-align: center;
                padding: 0;
                padding-bottom: 4px;
            }
        """

        self.setStyleSheet(self.normal_style)

    def enterEvent(self, event):
        super().enterEvent(event)
        self.setStyleSheet(self.hover_style)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.setStyleSheet(self.normal_style)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setStyleSheet(self.pressed_style)
            self.minimize_clicked.emit()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.setStyleSheet(self.normal_style)  # Quay về normal sau khi release
