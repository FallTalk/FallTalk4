from enum import Enum

from PySide6.QtGui import QIcon, QColor
from qfluentwidgets import FluentIconBase, Theme, getIconColor
from qfluentwidgets.common.icon import SvgIconEngine, writeSvg


class FallTalkStrokeIcons(FluentIconBase, Enum):
    """ Custom icons """

    ALPHA = "alpha"
    BETA = "beta"
    BUG = "bug"
    RECORD = "record"
    DELETE = "delete"
    HUGGING_FACE = "huggingface"
    NEW = "NEW"
    VOICE_OVER = "voice-over"
    VOICE_SQUARE = "voice-square"
    FROG = "frog"
    MUSIC = "music"
    REPLACE = "replace"
    ENHANCE = "enhance"
    SINE = "sine"


    def icon(self, theme=Theme.AUTO, color: QColor = None) -> QIcon:
        path = self.path(theme)

        if not color:
            return QIcon(SvgIconEngine(writeSvg(path, style=f"fill: none; stroke: rgb(0, 0, 0); stroke-linecap: round; stroke-linejoin: round; stroke-width: 2; stroke: {getIconColor(theme)}")))

        color = QColor(color).name()
        return QIcon(SvgIconEngine(writeSvg(path, style=f"fill: none; stroke: rgb(0, 0, 0); stroke-linecap: round; stroke-linejoin: round; stroke-width: 2; stroke: {color}")))

    def path(self, theme=Theme.AUTO):
        return f'resource/icons/{self.value}.svg'


class FallTalkIcons(FluentIconBase, Enum):
    """ Custom icons """

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
    def icon(self, theme=Theme.AUTO, color: QColor = None) -> QIcon:
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
        """
        path = self.path(theme)

        if path.endswith('.svg') and not color:
            return QIcon(SvgIconEngine(writeSvg(path, fill=getIconColor(theme))))

        color = QColor(color).name()
        return QIcon(SvgIconEngine(writeSvg(path, fill=color)))

    def path(self, theme=Theme.AUTO):
        return f'resource/icons/{self.value}.svg'
