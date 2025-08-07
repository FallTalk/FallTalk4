from enum import Enum

from PySide6.QtGui import QIcon, QColor
from qfluentwidgets import FluentIconBase, Theme, getIconColor
from qfluentwidgets.common.icon import SvgIconEngine, writeSvg


class FallTalkIcons(FluentIconBase, Enum):
    """ Custom icons """
    # Regular icons
    GPU = "gpu"
    VAULT_BOY = "vault_boy"
    KO_FI = "ko-fi"
    NEXUS = "nexus"
    DISCORD = "discord"
    LOOP = "loop"
    RAM = "ram"
    STEPS = "steps"
    SCALE = "scale"
    RECORD = "record"
    CPU = "cpu"
    IMPORTANT = "important"
    HUGGING_FACE = "huggingface"
    NEW = "NEW"
    VOICE_OVER = "voice-over"
    VOICE_SQUARE = "voice-square"
    STYLE = "style"
    TTS = "tts"
    VOICE = "voice"
    G = "g"
    FROG = "frog"
    BULK = "bulk"
    REPLACE = "replace"
    FX = "fx"
    ENHANCE = "enhance"
    SINE = "sine"
    API = "api"
    LLAMA = "llama"
    PILLS = "pills"
    TRIANGLE = "triangle"
    DIA = "dia"
    F5 = "f5"
    FISH = "fish"
    MAGIC = "magic"
    UP = "up"
    SPARK = "spark"
    CSM = "sesame"
    HIGGS = "higgs"
    CHATTERBOX = "chatterbox"
    DMO2 = "dmo2"

    # Stroke-only icons
    ALPHA = "alpha"
    BETA = "beta"
    BUG = "bug"
    DELETE = "delete"
    MUSIC = "music"

    def icon(self, theme=Theme.AUTO, color: QColor = None, stroke: bool = False) -> QIcon:
        """ create a fluent icon

        Parameters
        ----------
        theme: Theme
            the theme of icon
            * `Theme.Light`: black icon
            * `Theme.DARK`: white icon
            * `Theme.AUTO`: icon color depends on `qconfig.theme`

        color: QColor | Qt.GlobalColor | str
            icon color, only applicable to svg icon
            
        stroke: bool
            whether to render the icon as a stroke instead of a fill
        """
        path = self.path(theme)

        if not color:
            color = getIconColor(theme)

        if stroke:
            style = f"fill: none; stroke: rgb(0, 0, 0); stroke-linecap: round; stroke-linejoin: round; stroke-width: 2; stroke: {color}"
        else:
            style = f"fill: {color}"

        return QIcon(SvgIconEngine(writeSvg(path, style=style)))

    def path(self, theme=Theme.AUTO):
        return f'resource/icons/{self.value}.svg'