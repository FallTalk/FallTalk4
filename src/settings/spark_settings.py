from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    FluentIcon as FIF, SubtitleLabel, ComboBoxSettingCard
)
from qfluentwidgets import ScrollArea, ExpandLayout

from src.config.config import cfg


class SparkSettings(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("SparkSettings")
        self.scrollWidget = QWidget()
        self.expandLayout = ExpandLayout(self.scrollWidget)

        # Settings title
        self.settingLabel = SubtitleLabel("Spark-TTS Settings", self)
        self.expandLayout.addWidget(self.settingLabel)

        # Language settings
        self.languageCard = ComboBoxSettingCard(
            cfg.language,
            FIF.SETTING,
            "Language",
            "Select the language for speech synthesis",
            texts=["English", "Chinese"],
            parent=self.scrollWidget
        )
        self.expandLayout.addWidget(self.languageCard)

        # Speaker settings
        self.speakerCard = ComboBoxSettingCard(
            cfg.speaker,
            FIF.SETTING,
            "Speaker",
            "Select the speaker voice",
            texts=["Default"],
            parent=self.scrollWidget
        )
        self.expandLayout.addWidget(self.speakerCard)

        # Quality settings
        self.qualityCard = ComboBoxSettingCard(
            cfg.quality,
            FIF.SETTING,
            "Quality",
            "Select the quality of speech synthesis",
            texts=["Low", "Medium", "High"],
            parent=self.scrollWidget
        )
        self.expandLayout.addWidget(self.qualityCard)

        # Speed settings
        self.speedCard = ComboBoxSettingCard(
            cfg.speed,
            FIF.SETTING,
            "Speed",
            "Select the speech speed",
            texts=["Slow", "Normal", "Fast"],
            parent=self.scrollWidget
        )
        self.expandLayout.addWidget(self.speedCard)

        # Pitch settings
        self.pitchCard = ComboBoxSettingCard(
            cfg.pitch,
            FIF.SETTING,
            "Pitch",
            "Select the pitch of the voice",
            texts=["Low", "Normal", "High"],
            parent=self.scrollWidget
        )
        self.expandLayout.addWidget(self.pitchCard)

        # Add stretch to push all settings to the top
        self.expandLayout.addStretch(1)

        # Set the scroll widget
        self.setWidget(self.scrollWidget)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.viewport().setStyleSheet('background-color: transparent;') 