# send_stop_button.py
from PySide6.QtWidgets import QPushButton
from PySide6.QtCore import Signal, Qt


class SendStopButton(QPushButton):
    send_clicked = Signal()
    stop_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_running = False
        self.setObjectName("sendStopButton")
        self.setFixedSize(30, 30)  # Kích thước vuông
        self.setCursor(Qt.PointingHandCursor)
        self.update_state()

    def update_state(self):
        """Cập nhật trạng thái nút dựa trên is_running"""
        if self.is_running:
            self.setText("■")  # Stop
            self.setStyleSheet(
                """
                #sendStopButton {
                    background-color: #f44336;
                    border: 1px solid #505050;
                    border-radius: 9px;
                    color: #e0e0e0;
                    font-size: 14px;
                    text-align: center;
                    padding: 0;
                    padding-bottom: 4px;
                }
                #sendStopButton:hover {
                    background-color: #d32f2f;
                    border: 1px solid #f44336;
                }
                #sendStopButton:pressed {
                    background-color: #b71c1c;
                }
            """
            )
        else:
            self.setText("▶")  # Send
            self.setStyleSheet(
                """
                #sendStopButton {
                    background-color: rgba(28, 29, 35, 0.85);
                    border: 1px solid #505050;
                    border-radius: 9px;
                    color: #e0e0e0;
                    font-size: 14px;
                    text-align: center;
                    padding: 0;
                    padding-bottom: 4px;
                }
                #sendStopButton:hover {
                    background-color: #3a8cd7;
                    border: 1px solid #61afef;
                }
                #sendStopButton:pressed {
                    background-color: #1c6bb0;
                }
            """
            )

    def set_running(self, running: bool):
        """Cập nhật trạng thái worker và giao diện nút"""
        self.is_running = running
        self.update_state()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.is_running:
                self.stop_clicked.emit()
            else:
                self.send_clicked.emit()
        super().mousePressEvent(event)
