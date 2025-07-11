from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QWidget


class FallTalkWidget(QFrame):

    def __init__(self, text: str, parent=None, vertical=False):
        super().__init__(parent=parent)
        if vertical:
            self.boxLayout = QVBoxLayout(self)
        else:
            self.boxLayout = QHBoxLayout(self)
        self.title = text
        # self.settingLabel = SubtitleLabel(self.tr(text), self)
        # self.settingLabel.move(6, 5)
        # self.settingLabel.setFixedWidth(400)

        self.setObjectName(text.replace(' ', '-'))
        # !IMPORTANT: leave some space for title bar
        self.boxLayout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(self.boxLayout)

    def addToFrame(self, widget: QWidget) -> None:
        self.boxLayout.addWidget(widget)