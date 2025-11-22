# tray_icon.py
# -*- coding: utf-8 -*-
from PySide6.QtWidgets import QSystemTrayIcon, QMenu
from PySide6.QtGui import QIcon, QPainter, QColor, QPixmap, QAction
from PySide6.QtCore import Qt


class TrayIconManager:
    def __init__(self, app, chat_window):
        self.app = app
        self.chat_window = chat_window
        self.tray_icon = None
        self._create_tray_icon()

    def _create_icon(self):
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setBrush(QColor("white"))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "FourT")
        painter.end()
        return QIcon(pixmap)

    def _create_tray_icon(self):
        icon = self._create_icon()
        self.tray_icon = QSystemTrayIcon(icon, self.app)
        self.tray_icon.setToolTip("Trợ lý AI FourT")

        menu = QMenu()
        show_action = QAction("Hỏi FourT", self.app)
        quit_action = QAction("Thoát", self.app)

        show_action.triggered.connect(self.chat_window.center_and_show)
        quit_action.triggered.connect(self.app.quit)

        menu.addAction(show_action)
        menu.addAction(quit_action)
        self.tray_icon.setContextMenu(menu)
        self.tray_icon.show()

    def show_message(
        self, title, message, icon=QSystemTrayIcon.Information, duration=3000
    ):
        if self.tray_icon:
            self.tray_icon.showMessage(title, message, icon, duration)
