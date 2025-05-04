from src.config.config import cfg
from src.widgets.generation_widget import GenerationWidget

class XttsWidget(GenerationWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent, text="XTTS")
        self.addTempAndRep()
        self.addGenSettings()
        self.text_input.setPlaceholderText("Please enter text")
        self.addGenerationButton()
        self.setVisible(cfg.engine.value == "XTTSv2")
        self.media_player.setVisible(cfg.engine.value == "XTTSv2")
        self.setEnabled(False)