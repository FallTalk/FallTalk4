from PySide6.QtGui import QFont
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QFileDialog
from qfluentwidgets import FluentIcon as FIF, RangeSettingCard, TextEdit, PrimaryPushButton, ConfigItem, SwitchSettingCard, ConfigValidator, PushSettingCard

from audio.audio_player import StandardAudioPlayerBar
from src.config.config import cfg, FileValidator
from src.utils.icons import FallTalkIcons
from src.ui.cards import RangeSettingCardScaled, RadioSettingCard, TextSettingCard, SpinSettingCard
from widgets import FallTalkWidget


class AudioGenWidget(FallTalkWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent, text="Sound Generator", vertical=True)
        self.parent = parent

        self.text_input = TextEdit()
        font = QFont()
        font.setPointSize(12)
        self.text_input.setFont(font)
        self.addToFrame(self.text_input)

        self.text_input.setPlaceholderText("""
The first time you generate, a 5GB model must downloaded. It is recommended you have 6GB of VRAM, but it can work on less or CPU mode, just slowly.

dog barking

sirenes of an emergency vehicule

footsteps in a corridor

A baby is crying in a huge room.

Sine wave with low pitch.

Wooden table tapping sound followed by water pouring.

gun reloading

Two swords fighting

Audio model is licensed under CC-By-NC license for non commercial use        
        """)

        self.generate_button = PrimaryPushButton("Generate")
        self.generate_button.setIcon(FIF.SEND)
        self.generate_button.clicked.connect(self.parent.generate_fx)

        self.media_player = StandardAudioPlayerBar(self)
        self.media_player.setVolume(100)

        self.output_name = ConfigItem("fx", "output_name", None, ConfigValidator())

        self.autoplay = SwitchSettingCard(
            FIF.PLAY,
            self.tr('Autoplay'),
            self.tr('Automatically Play Generated Audio'),
            cfg.auto_play,
        )
        self.output_name_card = TextSettingCard(
            self.output_name,
            FIF.SAVE_AS,
            self.tr('Output Name'),
            self.tr('Name of Generated WAV file'),
            placeholder="Random"
        )

        self.parse_mode_card = RadioSettingCard(
            cfg.parse_mode,
            FIF.CUT,
            self.tr('Generation Mode'),
            self.tr('Split on commas to generate multiple layers.'),
            texts=["Split on Comma", "Single Command"],
            parent=self
        )

        self.duration_card = SpinSettingCard(
            cfg.fx_duration,
            FIF.STOP_WATCH,
            self.tr('Duration in Seconds'),
            self.tr('Max 120'),
            step=5
        )

        self.temperature_card = RangeSettingCardScaled(
            cfg.music_temperature,
            FIF.FRIGID,
            self.tr('Temperature'),
            self.tr('Randomness, 1 = balanced, 0 = disabled'),
        )

        self.a_d_ = QGroupBox()
        self.a_d_.setStyleSheet("border: none")
        self.a_d__layout = QHBoxLayout()
        self.a_d__layout.setContentsMargins(0, 0, 0, 0)
        self.a_d__layout.addWidget(self.autoplay, 3)
        self.a_d__layout.addWidget(self.duration_card, 3)
        self.a_d_.setLayout(self.a_d__layout)

        self.r_and_sub = QGroupBox()
        self.r_and_sub.setStyleSheet("border: none")
        self.r_and_sub_layout = QHBoxLayout()
        self.r_and_sub_layout.setContentsMargins(0, 0, 0, 0)
        self.r_and_sub_layout.addWidget(self.output_name_card, 3)
        self.r_and_sub_layout.addWidget(self.temperature_card, 3)
        self.r_and_sub.setLayout(self.r_and_sub_layout)

        self.addToFrame(self.parse_mode_card)
        self.addToFrame(self.r_and_sub)
        self.addToFrame(self.a_d_)
        self.addToFrame(self.generate_button)
        self.addToFrame(self.media_player)


class MusicGenWidget(FallTalkWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent, text="Music Generator", vertical=True)
        self.parent = parent

        self.text_input = TextEdit()
        font = QFont()
        font.setPointSize(12)
        self.text_input.setFont(font)
        self.addToFrame(self.text_input)

        self.text_input.setPlaceholderText("""
The first time you generate, a 10GB model must downloaded. It is recommended you have 12GB of VRAM, but it can work on less or CPU mode, just slowly.

Include some level of details on the instruments present, along with some intended use case (e.g. adding "perfect for a commercial") can sometimes help. You can control BPM and Time Signatures 4/4, 3/4, 5/4, 2/4, etc, or leave it to the AI. Here are some example prompts:

Violins and synths that inspire awe at the finiteness of life and the universe.

An 80s driving pop song with heavy drums and synth pads in the background

a light and cheerily EDM track, with syncopated drums, aery pads, and strong emotions bpm: 130

3/4 105bpm piano only baroque

Audio model is licensed under CC-By-NC license for non commercial use        
        """)

        self.generate_button = PrimaryPushButton("Generate")
        self.generate_button.setIcon(FIF.SEND)
        self.generate_button.clicked.connect(self.parent.generate_music)

        self.media_player = StandardAudioPlayerBar(self)
        self.media_player.setVolume(100)

        self.output_name = ConfigItem("TTS", "output_name", None, ConfigValidator())

        self.autoplay = SwitchSettingCard(
            FIF.PLAY,
            self.tr('Autoplay'),
            self.tr('Automatically Play Generated Audio'),
            cfg.auto_play,
        )

        self.temperature_card = RangeSettingCardScaled(
            cfg.music_temperature,
            FIF.FRIGID,
            self.tr('Temperature'),
            self.tr('Randomness, 1 = balanced, 0 = disabled'),
        )

        self.extend_stride_card = RangeSettingCard(
            cfg.extend_stride,
            FIF.SKIP_FORWARD,
            self.tr('Extended Strike'),
            self.tr('Higher Number = Faster, Lower = Better Quality'),
        )

        self.output_name_card = TextSettingCard(
            self.output_name,
            FIF.SAVE_AS,
            self.tr('Output Name'),
            self.tr('Name of Generated WAV file'),
            placeholder="Random"
        )

        self.duration_card = SpinSettingCard(
            cfg.music_duration,
            FIF.STOP_WATCH,
            self.tr('Duration in Seconds'),
            self.tr('Max 300'),
            step=5
        )

        self.parse_mode_card = RadioSettingCard(
            cfg.parse_mode,
            FIF.CUT,
            self.tr('Generation Mode'),
            self.tr('Split on commas to generate multiple musical layers.'),
            texts=["Split on Comma", "Single Command"],
            parent=self
        )

        self.mode_card = RadioSettingCard(
            cfg.audio_mode,
            FallTalkIcons.VOICE_SQUARE.icon(stroke=True),
            self.tr('Mode'),
            self.tr('Which model should we use? Changing causes loading in next generation'),
            texts=["Mono (5 GB)", "Stereo (12 GB)", "Song (20 GB)"],
            parent=self
        )

        self.ref_file = ConfigItem("audioGen", "ref", "Please Select a File", FileValidator())

        self.ref_file_card = PushSettingCard(
            self.tr('Select File'),
            FIF.DOCUMENT,
            self.tr("Reference Audio to use for Melody"),
            self.ref_file.value,
        )

        self.ref_file_card.clicked.connect(self.__onOutputFolderCardClicked)

        self.f_and_sub = QGroupBox()
        self.f_and_sub.setStyleSheet("border: none")
        self.f_and_sub_layout = QHBoxLayout()
        self.f_and_sub_layout.setContentsMargins(0, 0, 0, 0)
        self.f_and_sub_layout.addWidget(self.output_name_card, 3)
        self.f_and_sub_layout.addWidget(self.ref_file_card, 3)
        self.f_and_sub.setLayout(self.f_and_sub_layout)

        self.r_and_sub = QGroupBox()
        self.r_and_sub.setStyleSheet("border: none")
        self.r_and_sub_layout = QHBoxLayout()
        self.r_and_sub_layout.setContentsMargins(0, 0, 0, 0)
        self.r_and_sub_layout.addWidget(self.autoplay, 3)
        self.r_and_sub_layout.addWidget(self.duration_card, 3)
        self.r_and_sub.setLayout(self.r_and_sub_layout)

        self.t_and_s = QGroupBox()
        self.t_and_s.setStyleSheet("border: none")
        self.t_and_s_layout = QHBoxLayout()
        self.t_and_s_layout.setContentsMargins(0, 0, 0, 0)
        self.t_and_s_layout.addWidget(self.extend_stride_card, 3)
        self.t_and_s_layout.addWidget(self.temperature_card, 3)
        self.t_and_s.setLayout(self.t_and_s_layout)

        self.addToFrame(self.mode_card)
        self.addToFrame(self.parse_mode_card)
        self.addToFrame(self.f_and_sub)
        self.addToFrame(self.r_and_sub)
        self.addToFrame(self.t_and_s)
        self.addToFrame(self.generate_button)
        self.addToFrame(self.media_player)

    def __onOutputFolderCardClicked(self):
        allowed_file_types = "WAV files (*.wav);;MP3 files (*.mp3)"
        folder = QFileDialog.getOpenFileName(
            self, self.tr("Choose CSV or Text File"), "./", allowed_file_types)
        if not folder or folder[0] == "":
            return

        self.ref_file.value = folder[0]
        self.ref_file_card.setContent(folder[0])


