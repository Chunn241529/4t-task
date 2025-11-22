# token_dialog.py
# -*- coding: utf-8 -*-
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QMessageBox,
    QFrame,
)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QCursor
import os


class TokenDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Đăng nhập")
        self.setModal(True)
        self.setMinimumWidth(450)

        self.token_file = "token.txt"

        # Styling
        self.setStyleSheet(
            """
            QDialog {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 14px;
            }
            QLineEdit {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 1px solid #3498db;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton#cancelButton {
                background-color: #e74c3c;
            }
            QPushButton#cancelButton:hover {
                background-color: #c0392b;
            }
            QPushButton#linkButton {
                background-color: transparent;
                color: #3498db;
                text-align: left;
                padding: 0;
                font-weight: normal;
                text-decoration: underline;
            }
            QPushButton#linkButton:hover {
                color: #5dade2;
            }
        """
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(25, 25, 25, 25)

        # Instruction label
        label = QLabel(
            "Vui lòng nhập token của bạn để tiếp tục sử dụng FourT Assistant."
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        # Token input
        self.token_input = QLineEdit()
        self.token_input.setPlaceholderText("Nhập token của bạn vào đây...")
        self.token_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.token_input)

        # Get Token Link
        link_layout = QHBoxLayout()
        link_label = QLabel("Bạn chưa có token?")
        link_label.setStyleSheet("color: #888; font-size: 12px;")

        self.link_button = QPushButton("Lấy Token")
        self.link_button.setObjectName("linkButton")
        self.link_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.link_button.clicked.connect(self.open_token_url)

        link_layout.addWidget(link_label)
        link_layout.addWidget(self.link_button)
        link_layout.addStretch()
        layout.addLayout(link_layout)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #3d3d3d;")
        layout.addWidget(line)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_button = QPushButton("Hủy")
        self.cancel_button.setObjectName("cancelButton")
        self.cancel_button.clicked.connect(self.reject)

        self.save_button = QPushButton("Đăng nhập")
        self.save_button.clicked.connect(self.save_token)
        self.save_button.setDefault(True)

        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)
        layout.addLayout(button_layout)

        # Load existing token if available
        self.load_token()

    def load_token(self):
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, "r", encoding="utf-8") as f:
                    token = f.read().strip()
                    if token:
                        self.token_input.setText(token)
            except Exception as e:
                print(f"Error loading token: {e}")

    def save_token(self):
        token = self.token_input.text().strip()
        if not token:
            QMessageBox.warning(self, "Error", "Token cannot be empty.")
            return

        try:
            with open(self.token_file, "w", encoding="utf-8") as f:
                f.write(token)
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save token: {e}")

    def get_token(self):
        return self.token_input.text().strip()

    def open_token_url(self):
        url = QUrl("https://living-tortoise-polite.ngrok-free.app/")
        QDesktopServices.openUrl(url)
