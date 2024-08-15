from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QLineEdit
from qfluentwidgets import FluentIcon as FIF, SettingCardGroup, isDarkTheme, OptionsSettingCard, SwitchSettingCard
from qfluentwidgets import ScrollArea, RangeSettingCard, ExpandLayout

from falltalk.config import cfg, PitchExtractionAlgorithm, RangeSettingCardScaled, TextSettingCard


class RVCSettings(ScrollArea):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr(''), self.scroll_widget)
        self.eleven_labs_group = SettingCardGroup(self.tr('Eleven Labs'), self.scroll_widget)

        self.rvc_pitch_extraction_card = OptionsSettingCard(
            cfg.rvc_pitch_extraction,
            FIF.DEVELOPER_TOOLS,
            self.tr("Pitch Extraction"),
            self.tr("Choose the algorithm used for extracting the pitch (F0) during audio conversion."),
            texts=[member.value for member in PitchExtractionAlgorithm],
            parent=self.settings_group
        )

        self.rvc_training_data_size_card = RangeSettingCard(
            cfg.rvc_training_data_size,
            FIF.TRAIN,
            self.tr("Training Data Size"),
            self.tr("Determines the number of training data points used to train the FAISS index. Increasing the size may improve the quality of the output but can also increase computation time."),
            parent=self.settings_group
        )

        self.rvc_index_influence_card = RangeSettingCardScaled(
            cfg.rvc_index_influence,
            FIF.DICTIONARY,
            self.tr("Index Influence Ratio"),
            self.tr("A higher value increases the impact of the index, potentially enhancing detail but also increasing the risk of artifacts. Fine-tuning this setting helps achieve the desired balance between audio detail and artifact prevention."),
            parent=self.settings_group
        )

        self.rvc_hop_length_card = RangeSettingCardScaled(
            cfg.rvc_hop_length,
            FIF.ROTATE,
            self.tr("Hop Length"),
            self.tr("Smaller hop lengths require more time for inference but tend to yield higher pitch accuracy."),
            parent=self.settings_group
        )

        self.rvc_pitch_card = RangeSettingCard(
            cfg.rvc_pitch,
            FIF.MARKET,
            self.tr("Pitch Adjustment"),
            self.tr("Set the pitch of the audio, the higher the value, the higher the pitch."),
            parent=self.settings_group
        )

        self.rvc_volume_envelope_slider_card = RangeSettingCardScaled(
            cfg.rvc_volume_envelope,
            FIF.VOLUME,
            self.tr("Volume Envelope"),
            self.tr("Substitute or blend with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is employed."),
            parent=self.settings_group
        )

        self.rvc_protect_card = RangeSettingCardScaled(
            cfg.rvc_protect,
            FIF.BROOM,
            self.tr("Protect Voiceless Consonants / Breath Sounds Envelope"),
            self.tr("Prevents sound artifacts. Higher values (up to 0.5) provide stronger protection but may affect indexing."),
            parent=self.settings_group
        )

        self.rvc_filter_radius_card = RangeSettingCard(
            cfg.rvc_filter_radius,
            FIF.FILTER,
            self.tr("Filter Radius"),
            self.tr("If the number is greater than or equal to three, employing median filtering on the collected tone results has the potential to decrease respiration."),
            parent=self.settings_group
        )

        self.rvc_autotune_card = SwitchSettingCard(
            FIF.MUSIC,
            self.tr("Autotune"),
            self.tr("Apply a soft autotune to your inferences, recommended for signing."),
            configItem=cfg.rvc_autotune,
            parent=self.settings_group
        )

        self.rvc_split_audio_card = SwitchSettingCard(
            FIF.CUT,
            self.tr("Split Audio"),
            self.tr("Split the audio into chunks for inference to obtain better results in some cases."),
            configItem=cfg.rvc_split_audio,
            parent=self.settings_group
        )

        self.eleven_labs_key = TextSettingCard(
            cfg.rvc_eleven_labs_key,
            FIF.SAVE_AS,
            self.tr('API Access Key'),
            self.tr('Optional, used to access your custom eleven labs voices'),
            placeholder="Required to use service"
        )
        self.eleven_labs_key.lineEdit.setEchoMode(QLineEdit.EchoMode.Password)

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
        self.settings_group.addSettingCard(self.rvc_pitch_extraction_card)
        self.settings_group.addSettingCard(self.rvc_training_data_size_card)
        self.settings_group.addSettingCard(self.rvc_index_influence_card)
        self.settings_group.addSettingCard(self.rvc_hop_length_card)
        self.settings_group.addSettingCard(self.rvc_pitch_card)
        self.settings_group.addSettingCard(self.rvc_volume_envelope_slider_card)
        self.settings_group.addSettingCard(self.rvc_protect_card)
        self.settings_group.addSettingCard(self.rvc_filter_radius_card)
        self.settings_group.addSettingCard(self.rvc_autotune_card)
        self.settings_group.addSettingCard(self.rvc_split_audio_card)

        self.eleven_labs_group.addSettingCard(self.eleven_labs_key)

        # add setting card group to layout
        self.expand_layout.setSpacing(28)
        self.expand_layout.setContentsMargins(15, 0, 15, 0)

        self.expand_layout.addWidget(self.settings_group)
        self.expand_layout.addWidget(self.eleven_labs_group)

    def __setQss(self):
        """ set style sheet """
        self.scroll_widget.setObjectName('scrollWidget')

        theme = 'dark' if isDarkTheme() else 'light'
        with open(f'resource/qss/{theme}/setting_interface.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def __connectSignalToSlot(self):
        pass
