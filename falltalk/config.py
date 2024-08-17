# coding:utf-8
import os
from enum import Enum
from typing import Union

import torch
from PySide6.QtCore import Qt, QLocale, Signal, QObject
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QButtonGroup, QGroupBox
from qfluentwidgets import (qconfig, QConfig, ConfigItem, OptionsConfigItem, BoolValidator,
                            ColorConfigItem, OptionsValidator, RangeConfigItem, RangeValidator,
                            EnumSerializer, FolderValidator, ConfigSerializer, SettingCard, FluentIconBase, SpinBox, DoubleSpinBox, BodyLabel, ComboBox, PrimaryPushButton,
                            LineEdit, Slider, ConfigValidator, RadioButton)


class Language(Enum):
    """ Language enumeration """

    CHINESE_SIMPLIFIED = QLocale(QLocale.Chinese, QLocale.China)
    CHINESE_TRADITIONAL = QLocale(QLocale.Chinese, QLocale.HongKong)
    ENGLISH = QLocale(QLocale.English)
    AUTO = QLocale()


class LanguageSerializer(ConfigSerializer):
    """ Language serializer """

    def serialize(self, language):
        return language.value.name() if language != Language.AUTO else "Auto"

    def deserialize(self, value: str):
        return Language(QLocale(value)) if value != "Auto" else Language.AUTO


class StyledSettingCard(SettingCard):

    def __init__(self, icon: Union[str, QIcon, FluentIconBase], title, content, parent=None):
        super().__init__(icon, title, content, parent)
        self.setContentsMargins(0, 0, 10, 0)


class SpinSettingCard(StyledSettingCard):

    def __init__(self, configItem: RangeConfigItem, icon: Union[str, QIcon, FluentIconBase], title, content, parent=None):
        super().__init__(icon, title, content, parent)
        self.configItem = configItem
        sb = SpinBox()
        sb.setMinimumWidth(150)
        sb.setSingleStep(10)
        sb.setRange(*configItem.range)
        sb.setValue(configItem.value)
        self.hBoxLayout.addWidget(sb)

        sb.valueChanged.connect(self.setValue)

    def setValue(self, value):
        self.configItem.value = value


class RvcComboBoxSettingsCard(SettingCard):
    """ Setting card with a slider """

    valueChanged = Signal(int)

    def __init__(self, icon: Union[str, QIcon, FluentIconBase], title, content=None, parent=None):
        super().__init__(icon, title, content, parent)
        self.configItem = ComboBox(self)
        self.configItem.setMaxVisibleItems(10)
        self.configItem.setMinimumWidth(200)
        self.hBoxLayout.addStretch(1)
        self.hBoxLayout.addWidget(self.configItem, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addSpacing(16)
        self.transcript = None
        self.configItem.currentIndexChanged.connect(self.__onValueChanged)

    def __onValueChanged(self, value: int):
        selected_word = self.configItem.itemText(value)
        self.setValue(selected_word)
        self.valueChanged.emit(selected_word)

    def setValue(self, value):
        pass


class ComboBoxSettingsCard(SettingCard):
    """ Setting card with a slider """

    valueChanged = Signal(int)

    def __init__(self, icon: Union[str, QIcon, FluentIconBase], title, content=None, parent=None):
        super().__init__(icon, title, content, parent)
        self.configItem = ComboBox(self)
        self.configItem.setMaxVisibleItems(10)
        self.configItem.setMinimumWidth(200)
        self.valueLabel = QLabel(self)
        self.hBoxLayout.addStretch(1)
        self.hBoxLayout.addWidget(self.valueLabel, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addSpacing(6)
        self.hBoxLayout.addWidget(self.configItem, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addSpacing(16)
        self.transcript = None
        self.valueLabel.setObjectName('valueLabel')
        self.configItem.currentIndexChanged.connect(self.__onValueChanged)

    def __onValueChanged(self, value: int):
        selected_word = self.configItem.itemText(value)
        self.setValue(selected_word)
        self.valueChanged.emit(selected_word)

    def setTranscript(self, transcript):
        self.transcript = transcript

    def getWordInfo(self):
        if self.transcript:
            return self.transcript[self.configItem.currentIndex()]
        else:
            return None

    def setValue(self, value):
        word = self.getWordInfo()
        self.valueLabel.setText(f" {word['start']} {word['end']}")
        self.valueLabel.adjustSize()


class RangeSettingCardScaled(SettingCard):
    """ Setting card with a slider """

    valueChanged = Signal(int)

    def __init__(self, configItem, icon: Union[str, QIcon, FluentIconBase], title, content=None, scale=100.0, parent=None):
        super().__init__(icon, title, content, parent)
        self.configItem = configItem
        self.slider = Slider(Qt.Orientation.Horizontal, self)
        self.valueLabel = QLabel(self)
        self.slider.setMinimumWidth(268)
        self.scale = scale
        self.slider.setSingleStep(1)
        self.slider.setRange(*configItem.range)
        self.slider.setValue(configItem.value)
        self.valueLabel.setNum(configItem.value / self.scale)

        self.hBoxLayout.addStretch(1)
        self.hBoxLayout.addWidget(self.valueLabel, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addSpacing(6)
        self.hBoxLayout.addWidget(self.slider, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addSpacing(16)

        self.valueLabel.setObjectName('valueLabel')
        configItem.valueChanged.connect(self.setValue)
        self.slider.valueChanged.connect(self.__onValueChanged)

    def __onValueChanged(self, value: int):
        """ slider value changed slot """
        self.setValue(value)
        self.valueChanged.emit(value / self.scale)

    def setValue(self, value):
        qconfig.set(self.configItem, value)
        self.valueLabel.setNum(value / self.scale)
        self.valueLabel.adjustSize()
        self.slider.setValue(value)


class RadioSettingCard(StyledSettingCard):
    """ setting card with a group of options """

    optionChanged = Signal(OptionsConfigItem)

    def __init__(self, configItem, icon: Union[str, QIcon, FluentIconBase], title, content=None, texts=None, parent=None):
        """
        Parameters
        ----------
        configItem: OptionsConfigItem
            options config item

        icon: str | QIcon | FluentIconBase
            the icon to be drawn

        title: str
            the title of setting card

        content: str
            the content of setting card

        texts: List[str]
            the texts of radio buttons

        parent: QWidget
            parent window
        """
        super().__init__(icon, title, content, parent)
        self.texts = texts or []
        self.configItem = configItem
        self.configName = configItem.name
        self.buttonGroup = QButtonGroup(self)

        # create buttons
        self.options_group_box = QGroupBox(parent=self)
        self.options_group_box.setStyleSheet("border: none")
        self.options_box = QHBoxLayout()
        for text, option in zip(texts, configItem.options):
            button = RadioButton(text, self)
            self.buttonGroup.addButton(button)
            self.options_box.addWidget(button)
            button.setProperty(self.configName, option)
        self.options_group_box.setLayout(self.options_box)
        self.hBoxLayout.addWidget(self.options_group_box)
        self.setValue(qconfig.get(self.configItem))
        configItem.valueChanged.connect(self.setValue)
        self.buttonGroup.buttonClicked.connect(self.__onButtonClicked)

    def __onButtonClicked(self, button: RadioButton):
        value = button.property(self.configName)
        qconfig.set(self.configItem, value)
        self.optionChanged.emit(self.configItem)

    def setValue(self, value):
        """ select button according to the value """
        qconfig.set(self.configItem, value)

        for button in self.buttonGroup.buttons():
            isChecked = button.property(self.configName) == value
            button.setChecked(isChecked)


class DoubleSpinSettingCard(SettingCard):
    configItem: RangeConfigItem

    def __init__(self, configItem: RangeConfigItem, icon: Union[str, QIcon, FluentIconBase], title):
        super().__init__(icon, title)
        self.configItem = configItem
        dsb = DoubleSpinBox()
        dsb.setSingleStep(0.01)
        dsb.setRange(*configItem.range)
        dsb.setValue(configItem.value)
        self.hBoxLayout.addWidget(dsb)
        dsb.setMinimumWidth(150)

        dsb.valueChanged.connect(self.setValue)

    def setValue(self, value):
        self.configItem.value = value


class GroupItemDoubleSpin(QWidget):
    configItem: RangeConfigItem

    def __init__(self, configItem: RangeConfigItem, title):
        super().__init__()
        self.configItem = configItem

        hBoxLayout = QHBoxLayout()
        # hBoxLayout.setContentsMargins(48, 12, 48, 12)

        dsb = DoubleSpinBox()
        dsb.setMinimumWidth(150)
        dsb.setSingleStep(0.01)
        dsb.setRange(*configItem.range)
        dsb.setValue(configItem.value)

        hBoxLayout.addWidget(BodyLabel(title))
        hBoxLayout.addStretch(1)
        hBoxLayout.addWidget(dsb)

        self.setLayout(hBoxLayout)

        dsb.valueChanged.connect(self.setValue)

    def setValue(self, value):
        self.configItem.value = value


class TextSettingCard(StyledSettingCard):
    configItem: ConfigItem

    def __init__(self, configItem, icon: Union[str, QIcon, FluentIconBase], title, content=None, placeholder=None, parent=None):
        super().__init__(icon, title, content, parent)
        self.configItem = configItem
        self.lineEdit = LineEdit()
        self.lineEdit.setPlaceholderText(placeholder)
        self.lineEdit.setMinimumWidth(300)
        self.hBoxLayout.addWidget(self.lineEdit)
        self.lineEdit.setText(configItem.value)
        self.lineEdit.cursorPositionChanged.connect(self.setValue)

    def setValue(self, value):
        self.configItem.value = self.lineEdit.text()


def find_fallout4_exe():
    drives = [d + ':\\' for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
    common_paths = [
        '\\Program Files (x86)\\Steam\\steamapps\\common\\Fallout 4\\',
        '\\Program Files\\Steam\\steamapps\\common\\Fallout 4\\',
        '\\Program Files (x86)\\Fallout 4\\',
        '\\Program Files\\Fallout 4\\',
        '\\Steam\\steamapps\\common\\Fallout 4\\',
        '\\Steam Games\\steamapps\\common\\Fallout 4\\'

    ]

    found_paths = []

    for drive in drives:
        if not os.path.exists(drive):
            continue

        for common_path in common_paths:
            full_path = os.path.join(drive, common_path.lstrip('\\'))
            exe_path = os.path.join(full_path, 'Fallout4.exe')
            if os.path.isfile(exe_path):
                found_paths.append(full_path)

    if found_paths:
        return found_paths[0]
    else:
        return "fallout4.exe not found"


class FileValidator(ConfigValidator):

    def __init__(self, allowed_file_types=None):
        if allowed_file_types is None:
            allowed_file_types = ['csv', 'txt']
        self.allowed_file_types = allowed_file_types


    """ File validator """

    def validate(self, value):
        if os.path.exists(value) and os.path.isfile(value):
            _, file_extension = os.path.splitext(value)
            return file_extension in self.allowed_file_types

        return False


class PitchExtractionAlgorithm(Enum):
    """ Online song quality enumeration class """
    crepe = "crepe"
    crepe_tiny = "crepe-tiny"
    dio = "dio"
    fcpe = "fcpe"
    harvest = "harvest"
    hybrid = "hybird[rmcpe+fcpe]"
    rmvpe = "rmvpe"


class OnEngineChange(QObject):
    engine_signal = Signal(str)


class Fallout4FolderValidator(ConfigValidator):

    def validate(self, value):
        return os.path.exists(value) and os.path.isfile(os.path.join(value, "Fallout4.exe"))


class DeviceValidator(OptionsValidator):
    """ Options validator """

    def __init__(self):
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                super().__init__(["cpu", "cuda", "cuda:1"])
            else:
                super().__init__(["cpu", "cuda"])
        else:
            super().__init__(["cpu"])


class Config(QConfig):
    # RVC
    rvc_pitch = RangeConfigItem("RVC", "rvc_pitch", 0, RangeValidator(-24, 24))
    rvc_hop_length = RangeConfigItem("RVC", "rvc_hop_length", 1, RangeValidator(1, 512))
    rvc_training_data_size = RangeConfigItem("RVC", "rvc_training_data_size", 10000, RangeValidator(0, 50000))
    rvc_index_influence = RangeConfigItem("RVC", "rvc_index_influence", 75, RangeValidator(0, 100))
    rvc_volume_envelope = RangeConfigItem("RVC", "rvc_volume_envelope_Rslider", 100, RangeValidator(1, 100))
    rvc_protect = RangeConfigItem("RVC", "rvc_protect", 50, RangeValidator(0, 50))
    rvc_filter_radius = RangeConfigItem("RVC", "rvc_filter_radius", 3, RangeValidator(0, 7))
    rvc_autotune = ConfigItem("RVC", "rvc_autotune_checkbox", False, BoolValidator())
    rvc_split_audio = ConfigItem("RVC", "rvc_split_audio", False, BoolValidator())
    rvc_pitch_extraction = OptionsConfigItem("RVC", "rvc_pitch_extraction", default=PitchExtractionAlgorithm.rmvpe,
                                             validator=OptionsValidator(PitchExtractionAlgorithm),
                                             serializer=EnumSerializer(PitchExtractionAlgorithm))
    rvc_mode = OptionsConfigItem("RVC", "rvc_mode", "Microphone", OptionsValidator(["Microphone", "EdgeTTS", "Eleven-Labs"]))
    rvc_eleven_labs_key = ConfigItem("RVC", "rvc_eleven_labs_key", None, ConfigValidator())

    rvc_embedder_model = OptionsConfigItem("RVC", "rvc_embedding_model", "contentvec", OptionsValidator(["contentvec", "hubert"]))

    # General Model
    engine = OptionsConfigItem("TTS", "engine", "GPT_SoVITS", OptionsValidator(["XTTSv2", "VoiceCraft", "GPT_SoVITS", "StyleTTS2", "RVC"]))
    load_engine_art_start = ConfigItem("TTS", "load_at_start", False, BoolValidator())
    auto_update_models = ConfigItem("TTS", "auto_update_models", False, BoolValidator())
    device = OptionsConfigItem("TTS", "device", "cuda" if torch.cuda.is_available() else "cpu", DeviceValidator())

    fallout_4_directory = ConfigItem(
        "App", "fallout_4_directory", find_fallout4_exe(), Fallout4FolderValidator())
    fallout_4_directory_check = ConfigItem("App", "fallout_4_directory_check", True, BoolValidator())
    custom_references = ConfigItem(
        "App", "custom_references", "references/", FolderValidator())
    output_dir = ConfigItem("App", "output_dir", "output/", FolderValidator())
    rvc_enabled = ConfigItem("App", "rvc_enabled", True, BoolValidator())
    xwm_enabled = ConfigItem("App", "xwm_enabled", False, BoolValidator())
    download_configs = ConfigItem(
        "App", "download_configs", True, BoolValidator())
    check_for_updates = ConfigItem(
        "App", "check_for_updates", True, BoolValidator())
    auto_play = ConfigItem("App", "auto_play", True, BoolValidator())
    first_start = ConfigItem('App', 'first_start', True, BoolValidator())
    api_only_mode = ConfigItem('App', 'api_only_mode', False, BoolValidator(), restart=True)
    accepted_disclaimer = ConfigItem('App', 'accepted_up1_disclaimer', False, BoolValidator())

    # XTTS
    speed = RangeConfigItem("XTTS", "speed", 100, RangeValidator(1, 200))
    model_temperature = RangeConfigItem("XTTS", "model_temperature", 75, RangeValidator(1, 100))
    model_repetition = RangeConfigItem("XTTS", "model_repetition", 10, RangeValidator(1, 15))
    low_vram = ConfigItem("XTTS", "low_vram", False, BoolValidator())
    deepspeed_enabled = ConfigItem("XTTS", "deepspeed_enabled", False, BoolValidator())

    # VoiceCraft
    mode = OptionsConfigItem("VoiceCraft", "mode", "edit", OptionsValidator(["edit", "tts", "long_tts"]))
    stop_repetition = RangeConfigItem("VoiceCraft", "stop_repetition", 3, RangeValidator(-1, 4))
    sample_batch_size = RangeConfigItem("VoiceCraft", "sample_batch_size", 2, RangeValidator(1, 10))
    seed = RangeConfigItem("VoiceCraft", "seed", -1, RangeValidator(-1, 2 ** 30 - 1))
    kvcache = RangeConfigItem("VoiceCraft", "kvcache", 1, RangeValidator(0, 1))
    left_margin = RangeConfigItem("VoiceCraft", "left_margin", 80, RangeValidator(0, 100))
    right_margin = RangeConfigItem("VoiceCraft", "right_margin", 80, RangeValidator(0, 100))
    top_p = RangeConfigItem("VoiceCraft", "top_p", 90, RangeValidator(0.0, 100))
    top_k = RangeConfigItem("VoiceCraft", "top_k", 0, RangeValidator(0, 100))
    codec_audio_sr = RangeConfigItem("VoiceCraft", "codec_audio_sr", 16000, RangeValidator(1, 48000))
    codec_sr = RangeConfigItem("VoiceCraft", "codec_sr", 50, RangeValidator(1, 100))
    silence_tokens = OptionsConfigItem("VoiceCraft", "silence_tokens", '[1388, 1898, 131]', OptionsValidator(['[1388, 1898, 131]']))
    split_text = OptionsConfigItem("VoiceCraft", "split_text", "Newline", OptionsValidator(["Newline", "Sentence"]))
    smart_transcript = ConfigItem("VoiceCraft", "smart_transcript", True, BoolValidator())
    voicecraft_temperature = RangeConfigItem("VoiceCraft", "model_temperature", 100, RangeValidator(1, 100))
    edit_mode = OptionsConfigItem("VoiceCraft", "edit_mode", "replace all", OptionsValidator(["replace half", "replace all"]))

    # GPT_SoVITS
    slice_mode = OptionsConfigItem("GPT_SoVITS", "slice_mode", "Slice once every 4 sentences", OptionsValidator(["No Slice", "Slice by English punct", "Slice by every punct", "Slice once every 4 sentences", "Slice once every 2 sentences"]))
    low_vram_gpt_sovits = ConfigItem("GPT_SoVITS", "low_vram", False, BoolValidator())
    top_p_gpt_sovits = RangeConfigItem("GPT_SoVITS", "top_p", 100, RangeValidator(0.0, 100))
    top_k_gpt_sovits = RangeConfigItem("GPT_SoVITS", "top_k", 15, RangeValidator(0, 100))
    temperature_gpt_sovits = RangeConfigItem("GPT_SoVITS", "model_temperature", 75, RangeValidator(1, 100))
    speed_gpt_sovits = RangeConfigItem("GPT_SoVITS", "speed", 100, RangeValidator(1, 200))

    # StyleTTS2
    style_beta = RangeConfigItem("StyleTTS2", "beta", 20, RangeValidator(0, 100))
    style_alpha = RangeConfigItem("StyleTTS2", "alpha", 20, RangeValidator(0, 100))
    style_embedding_scale = RangeConfigItem("StyleTTS2", "embedding_scale", 1, RangeValidator(1, 3))
    style_diffusion_steps = RangeConfigItem("StyleTTS2", "diffusion_steps", 100, RangeValidator(1, 500))

    # theme
    themeColor = ColorConfigItem("QFluentWidgets", "ThemeColor", '#FFB642', restart=True)
    dpiScale = OptionsConfigItem(
        "MainWindow", "DpiScale", "Auto", OptionsValidator([1, 1.1, 1.25, 1.5, 1.75, 2, "Auto"]), restart=True)
    # main window
    enableAcrylicBackground = ConfigItem(
        "MainWindow", "EnableAcrylicBackground", False, BoolValidator())
    minimizeToTray = ConfigItem(
        "MainWindow", "MinimizeToTray", False, BoolValidator())
    playBarColor = ColorConfigItem("MainWindow", "PlayBarColor", "#225C7F")
    language = OptionsConfigItem(
        "MainWindow", "Language", Language.AUTO, OptionsValidator(Language), LanguageSerializer(), restart=True)

    def resetToDefault(self):
        self.resetVoiceCraft()
        self.resetXtts()
        self.resetGPT()
        self.resetStyleTTS()
        self.resetRvc()
        self.resetMainSettings()

    def resetMainSettings(self):
        self.set(self.download_configs, self.download_configs.defaultValue)
        self.set(self.check_for_updates, self.check_for_updates.defaultValue)
        self.set(self.api_only_mode, self.api_only_mode.defaultValue)

    def resetXtts(self):
        self.set(self.speed, self.speed.defaultValue)
        self.set(self.model_temperature, self.model_temperature.defaultValue)
        self.set(self.model_repetition, self.model_repetition.defaultValue)
        self.set(self.deepspeed_enabled, self.deepspeed_enabled.defaultValue)
        self.set(self.low_vram, self.low_vram.defaultValue)

    def resetVoiceCraft(self):
        self.set(self.mode, self.mode.defaultValue)
        self.set(self.stop_repetition, self.stop_repetition.defaultValue)
        self.set(self.sample_batch_size, self.sample_batch_size.defaultValue)
        self.set(self.seed, self.seed.defaultValue)
        self.set(self.kvcache, self.kvcache.defaultValue)
        self.set(self.left_margin, self.left_margin.defaultValue)
        self.set(self.right_margin, self.right_margin.defaultValue)
        self.set(self.top_p, self.top_p.defaultValue)
        self.set(self.top_k, self.top_k.defaultValue)
        self.set(self.codec_audio_sr, self.codec_audio_sr.defaultValue)
        self.set(self.codec_sr, self.codec_sr.defaultValue)
        self.set(self.silence_tokens, self.silence_tokens.defaultValue)
        self.set(self.split_text, self.split_text.defaultValue)
        self.set(self.smart_transcript, self.smart_transcript.defaultValue)
        self.set(self.voicecraft_temperature, self.voicecraft_temperature.defaultValue)
        self.set(self.edit_mode, self.edit_mode.defaultValue)

    def resetRvc(self):
        self.set(self.rvc_pitch, self.rvc_pitch.defaultValue)
        self.set(self.rvc_hop_length, self.rvc_hop_length.defaultValue)
        self.set(self.rvc_training_data_size, self.rvc_training_data_size.defaultValue)
        self.set(self.rvc_index_influence, self.rvc_index_influence.defaultValue)
        self.set(self.rvc_volume_envelope, self.rvc_volume_envelope.defaultValue)
        self.set(self.rvc_protect, self.rvc_protect.defaultValue)
        self.set(self.rvc_filter_radius, self.rvc_filter_radius.defaultValue)
        self.set(self.rvc_autotune, self.rvc_autotune.defaultValue)
        self.set(self.rvc_split_audio, self.rvc_split_audio.defaultValue)
        self.set(self.rvc_pitch_extraction, self.rvc_pitch_extraction.defaultValue)
        self.set(self.rvc_embedder_model, self.rvc_embedder_model.defaultValue)

    def resetGPT(self):
        self.set(self.slice_mode, self.slice_mode.defaultValue)
        self.set(self.low_vram_gpt_sovits, self.low_vram_gpt_sovits.defaultValue)
        self.set(self.top_p_gpt_sovits, self.top_p_gpt_sovits.defaultValue)
        self.set(self.top_k_gpt_sovits, self.top_k_gpt_sovits.defaultValue)
        self.set(self.temperature_gpt_sovits, self.temperature_gpt_sovits.defaultValue)
        self.set(self.speed_gpt_sovits, self.speed_gpt_sovits.defaultValue)

    def resetStyleTTS(self):
        self.set(self.style_beta, self.style_beta.defaultValue)
        self.set(self.style_alpha, self.style_alpha.defaultValue)
        self.set(self.style_embedding_scale, self.style_embedding_scale.defaultValue)
        self.set(self.style_diffusion_steps, self.style_diffusion_steps.defaultValue)


YEAR = 2024
AUTHOR = "Bryant21"
VERSION = '1.0.3'
NEXUS_URL = "https://www.nexusmods.com/fallout4/mods/86525"
HELP_URL = "https://github.com/falltalk/falltalk4"
FEEDBACK_URL = "https://github.com/falltalk/falltalk4/issues"
RELEASE_URL = "https://github.com/falltalk/falltalk4/releases/latest"
KOFI_URL = "https://ko-fi.com/bryant21"
DISCORD_URL = "https://discord.gg/FgKrxdnQdG"
HUGGING_FACE = "https://huggingface.co/falltalk/falltalk4"
REPO = "falltalk/falltalk4"

cfg = Config()
qconfig.load('config/config.json', cfg)

DISCLAIMER = """
    By accessing and using the FallTalk, you hereby agree to the following terms and conditions:

    Public Disclosure of AI Synthesis: You are obligated to clearly inform any and all end-users that the speech content they are interacting with has been synthesized using the FallTalk AI models. This disclosure should be made in a manner that is prominent and easily understandable.

    Permitted Use: You agree to use the FallTalk AI models exclusively for the following purposes:

       • Personal Use: Utilizing the FallTalk AI models for personal, non-commercial projects and activities that do not involve the distribution or sharing of synthesized speech content with others.
       • Research: Conducting academic or scientific research in the field of artificial intelligence, speech synthesis, or related disciplines.
       • Non-Commercial Mod Creation: Developing and distributing modifications (mods) for the game Fallout 4 that are available to the public free of charge.
    
    Prohibited Use: You are expressly prohibited from using the FallTalk AI models for any commercial purposes, including but not limited to:

       • Selling or licensing the synthesized speech content.
       • Incorporating the synthesized speech into any commercial product or service.
       • Creating or distributing any pornographic or adult material.

    Compliance with Laws and Regulations: You agree to comply with all applicable laws, regulations, and ethical standards in your use of the FallTalk models. This includes, but is not limited to, laws concerning intellectual property, privacy, and consumer protection. We assume no responsibility for any illegal use of the codebase.
"""
