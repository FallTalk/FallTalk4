from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout

class FallTalkWidget(QFrame):
    """
    Base widget class for FallTalk application.
    """
    def __init__(self, text: str, parent=None, vertical=False):
        super().__init__(parent=parent)
        self.setObjectName(text.replace(" ", ""))
        self.text = text
        
        if vertical:
            self.main_layout = QVBoxLayout(self)
        else:
            self.main_layout = QHBoxLayout(self)
            
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.main_layout)

    def addToFrame(self, widget):
        self.main_layout.addWidget(widget)