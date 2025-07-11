from PySide6.QtWidgets import QWidget

from src.widgets.falltalk_widget import FallTalkWidget
from src.widgets.faq import FAQPage


class FaqWidget(FallTalkWidget):
    def __init__(self, parent: QWidget = None):
        super().__init__(parent=parent, text="FAQ", vertical=True)
        self.faq = FAQPage(parent)
        self.addToFrame(self.faq)
