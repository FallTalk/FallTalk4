from src.widgets.generation_widget import GenerationWidget

class XttsWidget(GenerationWidget):
    """
    Widget for XTTS (X Text-to-Speech) generation.
    """
    def __init__(self, parent=None):
        super().__init__(text="XTTS v2", parent=parent)
        self.text_input.setPlaceholderText("Please Enter Text")