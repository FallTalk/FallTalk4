import os

import torch.cuda
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QFileDialog
from qfluentwidgets import FluentIcon as FIF, SettingCardGroup, isDarkTheme, OptionsSettingCard, SwitchSettingCard, \
    PushSettingCard, CustomColorSettingCard, HyperlinkCard, qconfig, InfoBar
from qfluentwidgets import ScrollArea, ExpandLayout

from falltalk.config import cfg, HELP_URL, YEAR, AUTHOR, VERSION, NEXUS_URL, KOFI_URL, DISCORD_URL, HUGGING_FACE
from icons import FallTalkIcons


class FallTalkSettings(ScrollArea):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr('General'), self.scroll_widget)

        self.tts_engine_card = OptionsSettingCard(
            cfg.engine,
            FIF.DEVELOPER_TOOLS,
            self.tr('Engine'),
            self.tr('Backend to use for generation'),
            texts=["XTTSv2",
                   "VoiceCraft",
                   "GPT SoVITSv2",
                   "StyleTTS2",
                   "RVC"],
            parent=self.settings_group
        )

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"Number of CUDA GPUs: {num_gpus}")
            if num_gpus > 1:
                texts = ["CPU",
                         "GPU 1",
                         "GPU 2"]
            else:
                texts = ["CPU",
                         "GPU"]
        else:
            texts = ["CPU"]

        self.tts_device_card = OptionsSettingCard(
            cfg.device,
            FallTalkIcons.GPU.icon(),
            self.tr('Device'),
            self.tr('Use GPU or CPU for generation. GPU recommended since its much faster'),
            texts=texts,
            parent=self.settings_group
        )

        self.fallout_4_directory = PushSettingCard(
            self.tr('Fallout 4 Directory'),
            FallTalkIcons.VAULT_BOY.icon(),
            self.tr("Install Directory for your packed Fallout 4"),
            cfg.get(cfg.fallout_4_directory),
            self.settings_group
        )

        self.output_dir = PushSettingCard(
            self.tr('Output Directory'),
            FIF.SAVE,
            self.tr("Where to save all the generated audio"),
            cfg.get(cfg.output_dir),
            self.settings_group
        )

        self.hugging_face_cache = PushSettingCard(
            self.tr('Huggingface Cache Directory'),
            FallTalkIcons.HUGGING_FACE.icon(),
            self.tr("Where the huggingface model cache is stored. (music gen, transcription, system files)"),
            cfg.get(cfg.huggingface_cache_dir),
            self.settings_group
        )

        self.load_engine_art_start = SwitchSettingCard(
            FIF.STOP_WATCH,
            self.tr('Load Engine At Start'),
            self.tr('Load the last used engine when starting the app'),
            configItem=cfg.load_engine_art_start,
            parent=self.settings_group
        )

        self.personalGroup = SettingCardGroup(self.tr('Personalization'), self.scroll_widget)
        self.themeCard = OptionsSettingCard(
            cfg.themeMode,
            FIF.BRUSH,
            self.tr('Application theme'),
            self.tr("Change the appearance of your application"),
            texts=[
                self.tr('Light'), self.tr('Dark'),
                self.tr('Use system setting')
            ],
            parent=self.personalGroup
        )
        self.themeColorCard = CustomColorSettingCard(
            cfg.themeColor,
            FIF.PALETTE,
            self.tr('Theme color'),
            self.tr('Change the theme color of you application'),
            self.personalGroup
        )

        self.zoomCard = OptionsSettingCard(
            cfg.dpiScale,
            FIF.ZOOM,
            self.tr("Interface zoom"),
            self.tr("Change the size of widgets and fonts"),
            texts=[
                "75%", "95%", "100%", "105%", "125%", "150%", "175%", "200%",
                self.tr("Use system setting")
            ],
            parent=self.personalGroup
        )

        # self.minimizeToTrayCard = SwitchSettingCard(
        #     FIF.MINIMIZE,
        #     self.tr('Minimize to tray after closing'),
        #     self.tr('PyQt-Fluent-Widgets will continue to run in the background'),
        #     configItem=cfg.minimizeToTray,
        #     parent=self.personalGroup
        # )

        # update software
        self.updateSoftwareGroup = SettingCardGroup(self.tr("Software update"), self.scroll_widget)
        self.download_configs = SwitchSettingCard(
            FIF.UPDATE,
            self.tr('Download Config Files'),
            self.tr('Get the latest JSON files automatically (recommended)'),
            configItem=cfg.download_configs,
            parent=self.updateSoftwareGroup
        )

        self.check_for_updates = SwitchSettingCard(
            FIF.UPDATE,
            self.tr('Check For Updates'),
            self.tr('Check for the latest updates and notify on release (recommended)'),
            configItem=cfg.check_for_updates,
            parent=self.updateSoftwareGroup
        )

        # application
        self.aboutGroup = SettingCardGroup(self.tr('About'), self.scroll_widget)
        self.helpCard = HyperlinkCard(
            HELP_URL,
            self.tr('Github'),
            FIF.GITHUB,
            self.tr('GitHub'),
            self.tr('Source Page for reporting issues and Downloading the latest release'),
            self.aboutGroup
        )

        self.aboutCard = HyperlinkCard(
            NEXUS_URL,
            self.tr('Nexus Mods'),
            FallTalkIcons.NEXUS.icon(),
            self.tr('Nexus'),
            '© ' + self.tr('Copyright') + f" {YEAR}, {AUTHOR}. " +
            self.tr('Version') + f" {VERSION}",
            self.aboutGroup
        )

        self.supportCard = HyperlinkCard(
            KOFI_URL,
            self.tr('ko-fi'),
            FallTalkIcons.KO_FI.icon(),
            self.tr('Support Us'),
            'Anything donated will support the app development and training of models ❤️',
            self.aboutGroup
        )

        self.discord = HyperlinkCard(
            DISCORD_URL,
            self.tr('Discord'),
            FallTalkIcons.DISCORD.icon(),
            self.tr('Join our discord today!'),
            'Our discord will be the place to preview new updates, test new models, share mods, and more.',
            self.aboutGroup
        )

        self.hugging_face = HyperlinkCard(
            HUGGING_FACE,
            self.tr('Huggingface'),
            FallTalkIcons.HUGGING_FACE.icon(),
            self.tr('Huggingface!'),
            'Huggingface model repository',
            self.aboutGroup
        )

        self.resetGroup = SettingCardGroup(self.tr('Reset'), self.scroll_widget)

        self.reset_to_default = PushSettingCard(
            self.tr('Reset Settings To Default'),
            FallTalkIcons.VAULT_BOY.icon(),
            self.tr("Reset All Settings To Default"),
            parent=self.resetGroup
        )

        #
        # self.reference_voice_directory = PushSettingCard(
        #     cfg.reference_voice_directory,
        #     FIF.DOWNLOAD,
        #     self.tr('Custom Reference Voice Directory'),
        #     self.tr("Directory you are going to place any custom reference voices"),
        #     self.settings_group
        # )

        self.__initWidget()

    def __initWidget(self):
        self.resize(1000, 800)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setViewportMargins(0, 0, 0, 20)
        self.setWidget(self.scroll_widget)
        self.setWidgetResizable(True)

        # initialize style sheet
        self.__setQss()

        # initialize layout
        self.__initLayout()
        self.__connectSignalToSlot()

    def __initLayout(self):
        # add cards to group
        self.settings_group.addSettingCard(self.tts_engine_card)
        self.settings_group.addSettingCard(self.tts_device_card)
        self.settings_group.addSettingCard(self.fallout_4_directory)
        self.settings_group.addSettingCard(self.load_engine_art_start)
        self.settings_group.addSettingCard(self.output_dir)
        self.settings_group.addSettingCard(self.hugging_face_cache)


        self.personalGroup.addSettingCard(self.themeCard)
        self.personalGroup.addSettingCard(self.themeColorCard)
        self.personalGroup.addSettingCard(self.zoomCard)

        self.updateSoftwareGroup.addSettingCard(self.download_configs)
        self.updateSoftwareGroup.addSettingCard(self.check_for_updates)

        self.aboutGroup.addSettingCard(self.helpCard)
        self.aboutGroup.addSettingCard(self.supportCard)
        self.aboutGroup.addSettingCard(self.aboutCard)
        self.aboutGroup.addSettingCard(self.discord)
        self.aboutGroup.addSettingCard(self.hugging_face)

        

        self.resetGroup.addSettingCard(self.reset_to_default)
        # add setting card group to layout
        self.expand_layout.setSpacing(28)
        self.expand_layout.setContentsMargins(15, 0, 15, 0)

        self.expand_layout.addWidget(self.settings_group)
        self.expand_layout.addWidget(self.personalGroup)
        self.expand_layout.addWidget(self.updateSoftwareGroup)
        self.expand_layout.addWidget(self.aboutGroup)
        self.expand_layout.addWidget(self.resetGroup)

        cfg.appRestartSig.connect(self.__showRestartTooltip)




    def __setQss(self):
        """ set style sheet """
        self.scroll_widget.setObjectName('scrollWidget')

        theme = 'dark' if isDarkTheme() else 'light'
        with open(f'resource/qss/{theme}/setting_interface.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def __connectSignalToSlot(self):
        self.tts_engine_card.optionChanged.connect(self.parent().onEngineChange)
        self.tts_device_card.optionChanged.connect(self.parent().onDeviceChange)
        self.hugging_face_cache.clicked.connect(self.__onHuggingaceHubClicked)


        self.fallout_4_directory.clicked.connect(self.__onFallout4FolderCardClicked)
        self.output_dir.clicked.connect(self.__onOutputFolderCardClicked)
        self.reset_to_default.clicked.connect(self.__OnResetClick)


    def __onFallout4FolderCardClicked(self):
        """ download folder card clicked slot """
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("Choose Fallout 4 Install Directory"), "./")
        if not folder or cfg.get(cfg.fallout_4_directory) == folder:
            return

        cfg.set(cfg.fallout_4_directory, folder)
        self.fallout_4_directory.setContent(folder)

    def __onHuggingaceHubClicked(self):
        """ download folder card clicked slot """
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("Choose Huggingface Cache Directory"), "./")
        if not folder or cfg.get(cfg.huggingface_cache_dir) == folder:
            return

        cfg.set(cfg.huggingface_cache_dir, folder)
        self.hugging_face_cache.setContent(folder)
        os.environ["HF_HUB_CACHE"] = folder


    def __onOutputFolderCardClicked(self):
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("Choose Output Directory"), "./")
        if not folder or cfg.get(cfg.output_dir) == folder:
            return

        cfg.set(cfg.output_dir, folder)
        self.output_dir.setContent(folder)

    def __OnResetClick(self):
        cfg.resetToDefault()

    def __showRestartTooltip(self):
        """ show restart tooltip """
        InfoBar.warning(
            '',
            self.tr('Configuration takes effect after restart'),
            parent=self.window(),
            duration=3000
        )


