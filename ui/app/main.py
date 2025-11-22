# main.py
# -*- coding: utf-8 -*-
import sys
import os
import shutil
import gc
from PySide6.QtWidgets import QApplication, QSplashScreen, QLabel, QMessageBox, QDialog
from PySide6.QtGui import QPixmap, QPainter, QFont, QColor, Qt
from PySide6.QtCore import Qt as QtCore
from chat_window import ChatWindow
from tray_icon import TrayIconManager
import torch  # Để clean GPU nếu có


def clear_python_cache():
    """Xóa các thư mục __pycache__ trong sys.path"""
    for path in sys.path:
        cache_dir = os.path.join(path, "__pycache__")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"Đã xóa cache tại: {cache_dir}")


def clean_resources():
    """Clean RAM và GPU"""
    gc.collect()
    print("Đã clean RAM (garbage collection)")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Đã clean GPU cache")


def create_FourT_pixmap(width, height):
    """Tạo pixmap với chữ FourT"""
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    font = QFont("Arial", 100, QFont.Bold)
    painter.setFont(font)
    painter.setPen(QColor("white"))
    text_rect = painter.fontMetrics().boundingRect("FourT")
    text_x = (pixmap.width() - text_rect.width()) // 2
    text_y = (pixmap.height() - text_rect.height()) // 2
    painter.drawText(text_x, text_y + painter.fontMetrics().ascent(), "FourT")
    painter.end()
    return pixmap


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    # Tạo pixmap cho splash với chữ FourT lớn ở giữa
    splash_pixmap = create_FourT_pixmap(400, 300)
    splash = QSplashScreen(splash_pixmap, QtCore.WindowStaysOnTopHint)
    splash.setWindowFlags(QtCore.FramelessWindowHint | QtCore.WindowStaysOnTopHint)

    # Tạo label cho message và căn giữa
    message_label = QLabel("Khởi động FourT Assistant...", splash)
    message_label.setAlignment(QtCore.AlignCenter)
    message_label.setStyleSheet(
        "color: white; font-size: 14px; background: transparent;"
    )
    splash.show()
    # Căn giữa label
    splash_size = splash.size()
    label_size = message_label.sizeHint()
    message_label.move(
        (splash_size.width() - label_size.width()) // 2, splash_size.height() - 50
    )
    app.processEvents()

    # Clean resources
    # message_label.setText("Kiểm tra tài nguyên...")
    # Cập nhật vị trí label sau khi text thay đổi
    label_size = message_label.sizeHint()
    message_label.move(
        (splash_size.width() - label_size.width()) // 2, splash_size.height() - 50
    )
    app.processEvents()
    clear_python_cache()
    clean_resources()

    # Check for token
    from token_dialog import TokenDialog

    token_dialog = TokenDialog()
    token = token_dialog.get_token()

    if not token:
        # Hide splash to show dialog clearly
        splash.hide()
        if token_dialog.exec() == QDialog.Accepted:
            token = token_dialog.get_token()
            # Show splash again briefly or just proceed
            splash.show()
        else:
            sys.exit(0)

    chat_window = ChatWindow()
    chat_window.token = token  # Pass token to chat window
    TrayIconManager(app, chat_window)

    # Ẩn splash
    splash.finish(chat_window)

    # Show chat window immediately
    chat_window.center_and_show()

    # # Hiển thị thông báo với  FourT
    # msg_box = QMessageBox()
    # msg_box.setWindowTitle("FourT Assistant")
    # msg_box.setText("FourT Assistant đang chạy ở chế độ nền trong khay hệ thống.")
    # # msg_box.setIconPixmap(create_FourT_pixmap(64, 64))  # Icon nhỏ cho QMessageBox
    # msg_box.setStandardButtons(QMessageBox.Ok)
    # msg_box.exec()

    print("Ứng dụng đã khởi động. Click vào icon trên khay hệ thống.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
