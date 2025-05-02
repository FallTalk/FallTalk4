from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QFileDialog
from qfluentwidgets import FluentIcon as FIF, PrimaryPushButton, SwitchSettingCard

from src.widgets.falltalk_widget import FallTalkWidget
from src.config.config import  cfg
from ui.cards import TextSettingCard, ComboBoxSettingsCard


class UpscaleWidget(FallTalkWidget):
    """
    Widget for audio upscaling functionality.
    """
    def __init__(self, parent=None):
        super().__init__(text="Bulk Enhancement", parent=parent, vertical=True)
        
        # Create upscale directory card
        self.upscale_dir = TextSettingCard(
            cfg.upscale_dir,
            FIF.FOLDER,
            self.tr('Upscale Directory'),
            self.tr('Directory containing audio files to upscale'),
            parent=self
        )
        self.upscale_dir.clicked.connect(self.__onFolderCardClicked)
        
        # Create audio mode card
        self.audio_mode = ComboBoxSettingsCard(
            cfg.audio_mode,
            FIF.MUSIC,
            self.tr('Audio Mode'),
            self.tr('Mode to use for audio upscaling'),
            texts=['Music', 'Voice', 'Ambient'],
            parent=self
        )
        
        # Create sample rate card
        self.sample_rate = ComboBoxSettingsCard(
            cfg.sample_rate,
            FIF.MUSIC,
            self.tr('Sample Rate'),
            self.tr('Sample rate to use for upscaled audio'),
            texts=['44100', '48000'],
            parent=self
        )
        
        # Create replace existing switch
        self.replace_existing = SwitchSettingCard(
            cfg.replace_existing,
            FIF.REPLACE,
            self.tr('Replace Existing'),
            self.tr('Replace existing files instead of creating new ones'),
            parent=self
        )
        
        # Create include subdirectories switch
        self.include_subdir = SwitchSettingCard(
            cfg.include_subdir,
            FIF.FOLDER,
            self.tr('Include Subdirectories'),
            self.tr('Process files in subdirectories'),
            parent=self
        )
        
        # Create switches group
        self.switches_group = QGroupBox()
        self.switches_group.setStyleSheet("border: none")
        self.switches_layout = QHBoxLayout()
        self.switches_layout.setContentsMargins(0, 0, 0, 0)
        self.switches_layout.addWidget(self.replace_existing)
        self.switches_layout.addWidget(self.include_subdir)
        self.switches_group.setLayout(self.switches_layout)
        
        # Create generate button
        self.generate_button = PrimaryPushButton('Enhance', self, FIF.PLAY)
        self.generate_button.clicked.connect(lambda: self.parent().upscale_folder())
        
        self.button_layout = QHBoxLayout()
        self.button_layout.addStretch(1)
        self.button_layout.addWidget(self.generate_button)
        self.button_layout.addStretch(1)
        
        # Add widgets to layout
        self.main_layout.addWidget(self.upscale_dir)
        self.main_layout.addWidget(self.audio_mode)
        self.main_layout.addWidget(self.sample_rate)
        self.main_layout.addWidget(self.switches_group)
        self.main_layout.addLayout(self.button_layout)

    def __onFolderCardClicked(self):
        """Handle folder card click to select upscale directory"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Upscale Directory"),
            ""
        )
        if folder_path:
            self.upscale_dir.configItem.setText(folder_path)