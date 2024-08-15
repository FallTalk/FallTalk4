from textwrap import dedent
from typing import Union

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QWidget, QLabel, QGroupBox, QHBoxLayout
from qfluentwidgets import (
    FluentIcon as FIF, SettingCardGroup, RangeSettingCard, isDarkTheme, ExpandSettingCard
)
from qfluentwidgets import ScrollArea, ExpandLayout

from falltalk.config import cfg, DISCLAIMER
from icons import FallTalkIcons, FallTalkStrokeIcons


class FaqSettingCard(ExpandSettingCard):
    def __init__(self, icon: Union[str, QIcon, FIF], title: str, content: str = None, parent=None, text: str = None, height=100):
        super().__init__(icon, title, content, parent)
        self.viewLayout.setSpacing(5)
        self.viewLayout.setContentsMargins(5, 5, 5, 5)
        self.options_group_box = QGroupBox(parent=self)
        self.options_group_box.setStyleSheet("border: none")
        self.options_box = QHBoxLayout(self)
        self.label = QLabel()

        self.label.setText(dedent(text))
        self.label.setWordWrap(True)
        self.label.setOpenExternalLinks(True)
        self.label.setMaximumHeight(height)
        self.label.setMinimumHeight(height)

        self.options_box.addWidget(self.label)
        self.options_group_box.setLayout(self.options_box)

        self.viewLayout.addWidget(self.options_group_box)
        self._adjustViewSize()


class FAQPage(ScrollArea):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.settings_group = SettingCardGroup(self.tr('Getting Started'), self.scroll_widget)
        self.tips_group = SettingCardGroup(self.tr('Tips and Tricks'), self.scroll_widget)
        self.about_group = SettingCardGroup(self.tr('About'), self.scroll_widget)
        self.models_group = SettingCardGroup(self.tr('Models'), self.scroll_widget)

        self.why = FaqSettingCard(
            FallTalkIcons.VAULT_BOY.icon(),
            self.tr('Why did you make this?'),
            text=
            """
            Originally, this started out as just a fun little project to get to understand how AI works, so I could further my career. One day while playing Fallout 4, I came across a mod that I had installed with absolutely terrible AI voices in it. This gave me an idea, what if I combined my little project to understand how AI works, and provided the community with a way to make better AI voices.
            
            I mean after all how hard could it be.
            """,
            height=150

        )

        self.eginesuse = FaqSettingCard(
            FIF.DEVELOPER_TOOLS,
            self.tr('Which Engine Should I use?'),
            text="""
            <p>That really depends on what you are trying to accomplish. Overall, I would recommend using GPT-SoVITs as the default to get started, but feel free to explore the others as each has its own strengths and weaknesses.</p>
            <ul>
                <li>RVC: Voice Cloning using your own voice! This allows you to achieve better results in some cases, as you can mimic the original cadence and speech patterns. Works really well with Robots like Mr Handy.</li>
                <li>GPT SoVITs: Extremely fast and easy to train, making it a great all-rounder. The default engine for version 1.0</li>
                <li>VoiceCraft: Has something the others don't: Audio Editing. Allows you to edit an existing audio file, changing a few words within a larger phrase. The can often lead to better quality than just creating new audio outright, but requires a powerful graphics card to run.</li>
                <li>XTTSv2: Offers decent quality, though it may sometimes lack emotion. It excels at generating large amounts of text, making it ideal for narration or long speeches.</li>
                <li>StyleTTSv2: Incredibly fast and, as the name suggests, is designed to mimic the tone and style of reference audio. It is often considered the best in the text-to-speech space, but requires a high-end GPU like the RTX 4090 (or two) for fine-tuning a voice.</li>
            </ul>  
            """,
            height=225
        )

        self.rvc = FaqSettingCard(
            FIF.VOLUME,
            self.tr('What is RVC?'),
            text="""
            Retrieval-based Voice Conversion (RVC) converts the voice of a source speaker into the voice of a target speaker while preserving the linguistic content. 
            
            The source voice can be anything from audio generated from a text to speech engine (TTS) or your own voice.
            
            It can thought of as an audio upscaler for TTS models, or can be used stand alone with your own voice.
            """,
            height=125

        )

        self.reference = FaqSettingCard(
            FIF.MIX_VOLUMES,
            self.tr('What is reference audio?'),
            text="""
            <p>This will be used as the base for making your new audio, is it very important in determining how the final audio sounds.
            
            Each Model has different reference requirements:</p>
            <ul>
                <li>GPT SoVITs (Fine Tuned): At least 5 seconds of reference audio More references can lead to a better result, assuming they all sound similar.</li>
                <li>GPT SoVITs (Untrained): 5-10 seconds. More than 10 seconds can lead to hallucinations.</li>
                <li>VoiceCraft: 3-16 seconds. Can work with longer, but can consume too much VRAM.</li>
                <li>XTTSv2: 10-15 seconds. Can give back hallucinations with less than 10 seconds, but does sometimes work</li>
                <li>StyleTTSv2: 5-10 seconds. </li>
            </ul>  
            """,
            height=125

        )

        self.gpu = FaqSettingCard(
            FallTalkIcons.GPU.icon(),
            self.tr("GPU or CPU?"),
            text="""
            The GPU will always be faster, but some engines like GPT-SoVITS and RVC see great performance on CPU. Other models like VoiceCraft really struggle with CPU only mode.
             
            Try the different settings to see if the CPU speed is acceptable for you.
            """
        )

        self.downmodels = FaqSettingCard(
            FIF.PEOPLE,
            self.tr("How do I get new models?"),
            text="""
            FallTalk will automatically download a config.json file each time it starts. This file is the master list of all models that can be downloaded at any given time. 
            You then click the download button to save and then use the models locally. I know hard drive space can be limited so I tried to make this as modular as possible.
            """
        )

        self.moremodels = FaqSettingCard(
            FIF.PEOPLE,
            self.tr("Why isn't [character name] trained?"),
            text="""
            Training each model takes a large of amount of time and power, and I am only one person. I will definitely add more model if there is demand, but I am limited by my hardware.
            I still like to play games, and when I am doing that I cannot train AI. There are cloud AI trainers, which are much faster but can add up to hundreds or thousands of dollars. 
            Ultimately the best plan is to try and use the power of the community to create more models.
            
            If you would like to help, there are more details below. You can also contribute to the ko-fi if you would like to support the creation of more models.
            """,
            height=155
        )

        self.api = FaqSettingCard(
            FIF.MEGAPHONE,
            self.tr('Upcoming Features or new Engines?'),
            text="""
            <p>I have a list of features that I would like to add, if given enough time.</p>
            <ul>
                <li>API Based Streaming: Allow on the fly generation of audio so it can be used ingame dynamically.</li>
                <li>Bulk Audio Creator: Allow uploading of a CSV with the engine, reference audi names, and text</li>
            </ul>             
            
            <p>There seem to be new voice cloning and text to speech engines every week. Here are a few I have been looking at:</p>
            <ul>
                <li><a href="https://github.com/fishaudio/fish-speech">Fish Speech</a>: Waiting on v1.3 to be released. Seems to have potential</li>
                <li><a href="https://github.com/FunAudioLLM/CosyVoice">CosyVoice</a>: Emotion controls look interesting</li>
                <li><a href="https://github.com/metavoiceio/metavoice-src">MetaVoice</a>: Good leaderboard scores, but not sure how easy to train.</li>
                <li><a href="https://github.com/myshell-ai/MeloTTS">MeloTTS</a>: Another with good scores, but doesnt take in reference audio so not sure how useful it can be</li>            
            </ul> 
            """,
            height=225

        )

        self.acknowledgements = FaqSettingCard(
            FIF.HEART,
            self.tr('Acknowledgements and Credits'),
            text="""
            <p>I would like to thank my wife for humoring me working and talking about AI all the time. Support like this is always needed.</p>
            <ul> 
                <li><a href="https://github.com/erew123">Erew123 creator AllTalk</a>: This project was originally a fork of AllTalk, all the work hes done is great. Getting deepspeed installed, rvc, and other features would not be possible without him.</li>
                <li><a href="https://www.nexusmods.com/skyrimspecialedition/mods/1756">BSA Browser CLI</a>: Allows us to extract the reference audio from Fallout 4</li>
                <li><a href="https://www.nexusmods.com/skyrimspecialedition/mods/17765">BowmoreLover Fuz Tools</a>: Allows us to extract the reference audio from the .fuz files</li>
                <li><a href="https://qfluentwidgets.com/">QT Fluent Widgets</a>: Look I am a backend developer, so this project was needed to make python look pretty.</li>
                <li><a href="https://github.com/oobabooga/text-generation-webui">Text Gen UI</a>: For the flash attention easy windows install</li>
                <li>Bethesda</a>: For the great games and modding capabilities.</li>
            </ul>             
            <p>And thank you to all the creators and contributors of the AI projects used.</p>
            """,
            height=225
        )

        self.ownmodels = FaqSettingCard(
            FIF.EDUCATION,
            self.tr('Can I train models?'),
            text="""
            <p>Yes, if you would like to help train models please join our Discord. As a warning, you will need some technical know-how with python, setting up envs, and general troubleshooting.
            
            Each engine's project have their own documentation and training requirements. The minimum amount of VRAM needed for training starts at 8 GB and goes up dramatically
            </p>
            <ul>
                <li><a href="https://github.com/IAHispano/Applio">RVC</a> : Very easy for anyone to train, with a simple .exe download install</li>
                <li><a href="https://github.com/erew123/alltalk_tts">XTTSv2 via AllTalk</a>: The creators of XTTS have closed, so this is currently the best repository for training.</li>
                <li><a href="https://github.com/jasonppy/VoiceCraft">VoiceCraft</a>: Takes a lot of VRAM, but we LoRA training scripts for this on the discord.</li>
                <li><a href="https://github.com/RVC-Boss/GPT-SoVITS">GPT So-VITS</a>: They have a super easy training UI that does all the work for you.</li>
                <li><a href="https://github.com/yl4579/StyleTTS2">StyleTTS2</a>: A monster for training, needs a 90 series card (3090, 4090) </li>
            </ul>  
            """,
            height=150
        )

        self.games = FaqSettingCard(
            FIF.GAME,
            self.tr('Can you add other games?'),
            text="""
            I would love to make SkyTalk, StarTalk, FallTalk3 etc, but ultimately I just do not have the time. If you are interesting in doing that, please join our discord and get started making a fork of the github repository.
            """
        )

        self.other = FaqSettingCard(
            FIF.TILES,
            self.tr('Is this better than ElevenLabs or xVASynth?'),
            text="""
            I think you can get quality comparable to the ElevenLabs with the right reference audio and settings. It's not perfect, but its free. As for xVASynth, it was great for the time but has become outdated.
            """
        )

        self.bugs = FaqSettingCard(
            FallTalkStrokeIcons.BUG.icon(),
            self.tr('Crashed / Encountered an Error'),
            text=
            """
            That can happen with software as complex as this, but we are ready for this. The good news is we have two options on how to get help.
            
            The first thing you need to do is locate your log file, which should be in "logs" folder where you installed FallTalk.
             
            Then you can Join the FallTalk discord and navigate to bug-reporting channel.
            
            If you are a bit more tech savy, you can open an issue on the GitHub repository with the log error trace, and steps to reproduce.
            """,
            height=150

        )

        self.disclaimer = FaqSettingCard(
            FIF.TILES,
            self.tr('Disclaimer for Use of FallTalk'),
            text=DISCLAIMER,
            height=400
        )

        self.text = FaqSettingCard(
            FIF.INFO,
            self.tr('Creating Text'),
            text="""                
                <p>There are a few import things to keep in mind when creating text to get the best results:</p>
               
                <ul>
                    <li>Length: Many of the models when trying to create a small amount of text. It's a good idea to try and match the length of the reference audios text.</li>
                    <li>Punctuation: Periods are your best friend when it comes to getting good results. Many of the engines internally need periods to know how to generate the voice, and adding them inserts pauses</li>
                    <li>Numbers: You may need to break up numbers to get them to sound correct. "2077" vs "20 77" for example.</li>
                </ul>
               """,
            height=175
        )

        self.params = FaqSettingCard(
            FIF.INFO,
            self.tr('Engine Parameters'),
            text="""
            All of the engines have many parameters, which can be adjusted for you. We have tried to set the "best" as defaults, but playing around with them can give better or worse results. 
            
            The reset button on the top of the page will revert to default, so play around until you fine what you like. There is also a reset all option in settings.
            """,
            height=125
        )

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
        self.settings_group.addSettingCard(self.why)
        self.settings_group.addSettingCard(self.eginesuse)
        self.settings_group.addSettingCard(self.reference)
        self.settings_group.addSettingCard(self.rvc)
        self.settings_group.addSettingCard(self.bugs)

        self.tips_group.addSettingCard(self.text)
        self.tips_group.addSettingCard(self.params)
        self.tips_group.addSettingCard(self.gpu)

        self.models_group.addSettingCard(self.ownmodels)
        self.models_group.addSettingCard(self.moremodels)
        self.models_group.addSettingCard(self.downmodels)

        self.about_group.addSettingCard(self.api)
        self.about_group.addSettingCard(self.games)
        self.about_group.addSettingCard(self.acknowledgements)
        self.about_group.addSettingCard(self.disclaimer)

        # add setting card group to layout
        self.expand_layout.setSpacing(28)
        self.expand_layout.setContentsMargins(15, 0, 15, 0)
        self.expand_layout.addWidget(self.settings_group)
        self.expand_layout.addWidget(self.tips_group)
        self.expand_layout.addWidget(self.models_group)
        self.expand_layout.addWidget(self.about_group)

    def __setQss(self):
        """ set style sheet """
        self.scroll_widget.setObjectName('scrollWidget')

        theme = 'dark' if isDarkTheme() else 'light'
        with open(f'resource/qss/{theme}/setting_interface.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def __connectSignalToSlot(self):
        pass
