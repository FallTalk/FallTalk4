from PySide6.QtCore import Qt, QRect, QPropertyAnimation, QEasingCurve, QEvent
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPalette
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QApplication, QLabel, QFrame
)
from qfluentwidgets import ToolButton, FluentIcon as FIF, isDarkTheme


class RightDrawer(QFrame):
    """Right-aligned drawer with custom margins"""

    def __init__(self, parent=None, width=800, title="Settings", icon=FIF.SETTING):
        super().__init__(parent)
        self.parent = parent
        self.drawer_width = width
        self.corner_radius = 12

        # Window setup
        self.setWindowFlags(Qt.WindowType.SubWindow | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        # Initial geometry (offscreen)
        self.setGeometry(parent.width(), 0, self.drawer_width, parent.height())

        # Main layout with outer margins
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)  # Outer margins set to 0
        self.main_layout.setSpacing(0)

        # Header container with custom margins
        self.header_container = QFrame(self)
        self.header_container.setObjectName("headerContainer")
        self.header_container.setStyleSheet("""
            #headerContainer {
                background: transparent;
                border: none;
            }
        """)
        self.header_layout = QVBoxLayout(self.header_container)
        self.header_layout.setContentsMargins(20, 12, 20, 20)  # Header margins
        self.header_layout.setSpacing(0)

        # Header content
        self.header = QFrame()
        self.header.setObjectName("header")
        self.header.setFixedHeight(28)
        self.header_layout_inner = QHBoxLayout(self.header)
        self.header_layout_inner.setContentsMargins(0, 0, 0, 0)
        self.header_layout_inner.setSpacing(8)

        self.icon = QLabel()
        self.icon.setPixmap(icon.icon().pixmap(20, 20))
        self.title = QLabel(title)
        self.title.setStyleSheet("font-size: 15px; font-weight: 500;")
        self.title.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

        self.close_btn = ToolButton()
        self.close_btn.setFixedSize(28, 28)
        self.close_btn.setIcon(FIF.CLOSE.icon())
        self.close_btn.clicked.connect(self.close_drawer)

        self.header_layout_inner.addWidget(self.icon)
        self.header_layout_inner.addWidget(self.title)
        self.header_layout_inner.addStretch()
        self.header_layout_inner.addWidget(self.close_btn)

        self.header_layout.addWidget(self.header)
        self.main_layout.addWidget(self.header_container)

        # Content area with zero margins
        self.content_widget = QWidget()
        self.content_widget.setObjectName("contentWidget")
        self.content_widget.setStyleSheet("""
            #contentWidget {
                background: transparent;
                border: none;
                margin: 0;
                padding: 0;
            }
        """)
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)  # Zero margins for content
        self.content_layout.setSpacing(12)
        self.main_layout.addWidget(self.content_widget, 1)  # Stretch factor 1

        # Animations
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setEasingCurve(QEasingCurve.Type.OutQuad)
        self.animation.setDuration(250)

        # Click outside handling
        self._click_outside_to_close = True
        self._event_filter_installed = False

        # Ensure background is properly set
        self.updateBackground()
        self.current_widget = None

    def backgroundColor(self):
        return QColor(40, 40, 40) if isDarkTheme() else QColor(248, 248, 248)

    def borderColor(self):
        return QColor(0, 0, 0, 45) if isDarkTheme() else QColor(0, 0, 0, 17)

    def updateBackground(self):
        """Update background color based on theme"""
        palette = self.palette()
        if isDarkTheme():
            palette.setColor(QPalette.ColorRole.Window, self.backgroundColor())
        else:
            palette.setColor(QPalette.ColorRole.Window, self.backgroundColor())
        self.setPalette(palette)

    def paintEvent(self, event):
        """Draw styled background with rounded corners"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath()
        rect = self.rect()
        path.addRoundedRect(rect, self.corner_radius, self.corner_radius)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.backgroundColor())
        painter.drawPath(path)

        # Draw border if needed
        painter.setPen(self.borderColor())
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5),
                                self.corner_radius, self.corner_radius)

    def addWidget(self, widget):
        """Add widget to content area"""
        if self.current_widget:
            self.content_layout.removeWidget(self.current_widget)

        self.content_layout.addWidget(widget)
        self.current_widget = widget

    def setClickOutsideToClose(self, enable: bool):
        """Set whether clicking outside closes the drawer"""
        self._click_outside_to_close = enable
        if enable and not self._event_filter_installed:
            QApplication.instance().installEventFilter(self)
            self._event_filter_installed = True
        elif not enable and self._event_filter_installed:
            QApplication.instance().removeEventFilter(self)
            self._event_filter_installed = False

    def eventFilter(self, obj, event):
        """Handle click outside events"""
        if (self._click_outside_to_close and
                self.isVisible() and
                event.type() == QEvent.Type.MouseButtonPress):

            click_pos = event.globalPosition().toPoint()
            if not self.rect().contains(self.mapFromGlobal(click_pos)):
                self.close_drawer()
                return True

        return super().eventFilter(obj, event)

    def open_drawer(self):
        """Animate drawer open"""
        self.raise_()
        self.show()

        if self._click_outside_to_close and not self._event_filter_installed:
            QApplication.instance().installEventFilter(self)
            self._event_filter_installed = True

        try:
            self.animation.finished.disconnect()
        except:
            pass

        self.animation.stop()
        self.animation.setStartValue(QRect(self.parent.width(), 0, self.drawer_width, self.parent.height()))
        self.animation.setEndValue(
            QRect(self.parent.width() - self.drawer_width, 0, self.drawer_width, self.parent.height()))
        self.animation.start()

    def close_drawer(self):
        """Animate drawer closed"""
        try:
            self.animation.finished.disconnect()
        except:
            pass

        self.animation.stop()
        self.animation.setStartValue(self.geometry())
        self.animation.setEndValue(QRect(self.parent.width(), 0, self.drawer_width, self.parent.height()))
        self.animation.finished.connect(self._finalize_close)
        self.animation.start()

    def _finalize_close(self):
        """Clean up after closing animation"""
        self.hide()
        if self._event_filter_installed:
            QApplication.instance().removeEventFilter(self)
            self._event_filter_installed = False

    def showEvent(self, event):
        """Update position when shown"""
        self.setGeometry(self.parent.width(), 0, self.drawer_width, self.parent.height())
        super().showEvent(event)

    def closeEvent(self, event):
        """Clean up when closed"""
        if self._event_filter_installed:
            QApplication.instance().removeEventFilter(self)
            self._event_filter_installed = False
        super().closeEvent(event)