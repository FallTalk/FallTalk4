from src.widgets.falltalk_widget import FallTalkWidget
from src.widgets.faq import FAQPage

class FaqWidget(FallTalkWidget):
    """
    Widget for displaying FAQ information.
    """
    def __init__(self, parent=None):
        super().__init__(text="FAQ", parent=parent)
        self.addToFrame(FAQPage(self))
