from typing import Union, List

import torch
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QAction
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QListWidgetItem
from qfluentwidgets import FluentIcon as FIF, FluentIconBase, CommandBar, Action, TransparentDropDownPushButton, \
    setFont, MenuIndicatorType, qrouter, FluentTitleBar, NavigationInterface, NavigationItemPosition, \
    NavigationTreeWidget, BodyLabel, RoundMenu
from qfluentwidgets.components.widgets.menu import createCheckableMenuItemDelegate, MenuAnimationType
from qfluentwidgets.window.fluent_window import FluentWindowBase

from src.config.config import cfg
from src.enums.engine_type import EngineType
from src.utils.icons import FallTalkIcons


class CustomCheckableMenu(RoundMenu):
    """ Checkable menu """

    def __init__(self, title="", parent=None, indicatorType: MenuIndicatorType = MenuIndicatorType.CHECK):
        super().__init__(title, parent)
        self.view.setItemDelegate(createCheckableMenuItemDelegate(indicatorType))
        self.view.setObjectName('checkableListWidget')

    def _adjustItemText(self, item: QListWidgetItem, action: QAction):
        w = super()._adjustItemText(item, action)
        item.setSizeHint(QSize(w + 100, self.itemHeight))

    def exec(self, pos, ani=True, aniType=MenuAnimationType.DROP_DOWN):
        return super().exec(pos, ani, aniType)

class CustomCommandBar(CommandBar):

    def __init__(self, parent=None):
        super().__init__(parent)

    def _visibleWidgets(self) -> List[QWidget]:
        """ return the visible widgets in layout """
        # have enough spacing to show all widgets
        return self._widgets

class FallTalkFluentWindow(FluentWindowBase):
    """ Fluent window """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitleBar(FluentTitleBar(self))

        self.navigationInterface = NavigationInterface(self, showReturnButton=True)

        self.toolbar = QHBoxLayout()
        self.toolbar.stretch(1)
        self.widgetLayout = QVBoxLayout()
        self.label = BodyLabel(self.tr("Character Models"))
        self.label.setFixedWidth(125)

        self.character_label = BodyLabel(self.tr("Please Load Model"))
        self.character_label.setFixedWidth(150)

        self.reference_label = BodyLabel(self.tr("Reference:"))
        self.reference_time_label = BodyLabel(self.tr("00:00"))
        self.reference_time_label.setFixedWidth(45)

        self.rvc_action = Action(FallTalkIcons.VOICE_SQUARE.icon(stroke=True), self.tr('RVC\t\t0.5 GB VRAM'), checkable=True, checked=cfg.get(cfg.engine) == EngineType.RVC.value)
        self.gpt_sovits_action = Action(FallTalkIcons.G.icon(), self.tr('GPT_SoVITS\t\t4 GB VRAM'), checkable=True, checked=cfg.get(cfg.engine) == EngineType.GPT_SOVITS.value)
        self.styletts2_action = Action(FallTalkIcons.STYLE.icon(), self.tr('StyleTTS2\t\t16 GB VRAM'), checkable=True, checked=cfg.get(cfg.engine) == EngineType.STYLE_TTS2.value)
        self.fish_action = Action(FallTalkIcons.FISH.icon(), self.tr('FishSpeech\t\t5 GB VRAM'), checkable=True, checked=cfg.get(cfg.engine) == EngineType.FISH_SPEECH.value)
        self.f5_action = Action(FallTalkIcons.F5.icon(), self.tr('F5\t\t3 GB VRAM'), checkable=True, checked=cfg.get(cfg.engine) == EngineType.F5.value)
        self.dia_action = Action(FallTalkIcons.DIA.icon(stroke=True), self.tr('DIA\t\t10 GB VRAM'), checkable=True, checked=cfg.get(cfg.engine) == EngineType.DIA.value)
        self.llasa_action = Action(FallTalkIcons.LLAMA.icon(), self.tr('Llasa\t\t8 GB VRAM'), checkable=True, checked=cfg.get(cfg.engine) == EngineType.LLASA.value)
        self.orpheus_action = Action(FallTalkIcons.TRIANGLE.icon(), self.tr('Orpheus\t\t8 GB VRAM'), checkable=True, checked=cfg.get(cfg.engine) == EngineType.ORPHEUS.value)
        self.spark_action = Action(FallTalkIcons.SPARK.icon(), self.tr('Spark\t\t5 GB VRAM'), checkable=True, checked=cfg.get(cfg.engine) == EngineType.SPARK.value)
        self.csm_action = Action(FallTalkIcons.CSM.icon(), self.tr('CSM\t\t5 GB VRAM'), checkable=True, checked=cfg.get(cfg.engine) == EngineType.CSM.value)
        self.higgs_action = Action(FallTalkIcons.HIGGS.icon(), self.tr('Higgs\t\t24 GB VRAM'), checkable=True, checked=cfg.get(cfg.engine) == EngineType.HIGGS.value)
        self.chatterbox_action = Action(FallTalkIcons.CHATTERBOX.icon(stroke=True), self.tr('Chatterbox\t\t6.5 GB VRAM'), checkable=True, checked=cfg.get(cfg.engine) == EngineType.CHATTERBOX.value)
        self.dmo_speech2_action = Action(FallTalkIcons.DMO2.icon(stroke=True), self.tr('DMO\t\t5 GB VRAM'), checkable=True, checked=cfg.get(cfg.engine) == EngineType.DMOSPEECH2.value)
        self.vibe_action = Action(FallTalkIcons.MICROSOFT.icon(stroke=True), self.tr('MS Vibe\t\t7 GB VRAM'), checkable=True, checked=cfg.get(cfg.engine) == EngineType.VIBE.value)
        self.qwen_action = Action(FallTalkIcons.QWEN.icon(stroke=True), self.tr('Qwen3\t\t8 GB VRAM'), checkable=True, checked=cfg.get(cfg.engine) == EngineType.QWEN3_TTS.value)

        self.cpu_action = Action(FallTalkIcons.CPU.icon(), self.tr('CPU'), checkable=True, checked=cfg.get(cfg.device) == 'cpu')
        self.gpu_action = Action(FallTalkIcons.GPU.icon(), self.tr('GPU'), checkable=True, checked=cfg.get(cfg.device) == 'cuda')
        self.gpu2_action = Action(FallTalkIcons.GPU.icon(), self.tr('GPU 2'), checkable=True, checked=cfg.get(cfg.device) == 'cuda:1')
        #self.clean_action = Action(FIF.BROOM, self.tr('Clean'))

        self.update_action = Action(FallTalkIcons.IMPORTANT.icon(color=cfg.get(cfg.themeColor)), self.tr('Update Available'))
        self.new_models_action = Action(FallTalkIcons.NEW.icon(color=cfg.get(cfg.themeColor)), self.tr('New Models Added'))

        # initialize layout
        self.toolbar_1 = self.createCommandBar()
        self.toolbar.addWidget(self.toolbar_1, stretch=1)

        self.widgetLayout.addLayout(self.toolbar)

        self.hBoxLayout.addWidget(self.navigationInterface)
        self.hBoxLayout.addLayout(self.widgetLayout)
        self.hBoxLayout.setStretchFactor(self.widgetLayout, 1)

        self.widgetLayout.addWidget(self.stackedWidget)
        self.widgetLayout.setContentsMargins(0, 48, 0, 0)

        self.navigationInterface.displayModeChanged.connect(self.titleBar.raise_)
        self.titleBar.raise_()

    def _onCurrentInterfaceChanged(self, index: int):
        super()._onCurrentInterfaceChanged(index)
        self.label.setText(self.stackedWidget.currentWidget().title)

    def addSubInterface(self, interface: QWidget, icon: Union[FluentIconBase, QIcon, str], text: str,
                        position=NavigationItemPosition.TOP, parent=None, isTransparent=False) -> NavigationTreeWidget:

        if not interface.objectName():
            raise ValueError("The object name of `interface` can't be empty string.")
        if parent and not parent.objectName():
            raise ValueError("The object name of `parent` can't be empty string.")

        interface.setProperty("isStackedTransparent", isTransparent)
        self.stackedWidget.addWidget(interface)

        # add navigation item
        routeKey = interface.objectName()
        item = self.navigationInterface.addItem(
            routeKey=routeKey,
            icon=icon,
            text=text,
            onClick=lambda: self.switchTo(interface),
            position=position,
            tooltip=text,
            parentRouteKey=parent.objectName() if parent else None
        )

        # initialize selected item
        if self.stackedWidget.count() == 1:
            self.stackedWidget.currentChanged.connect(self._onCurrentInterfaceChanged)
            self.navigationInterface.setCurrentItem(routeKey)
            qrouter.setDefaultRouteKey(self.stackedWidget, routeKey)

        self._updateStackedBackground()

        return item

    def resizeEvent(self, e):
        self.titleBar.move(46, 0)
        self.titleBar.resize(self.width() - 46, self.titleBar.height())

    def createEngineMenu(self, pos=None):
        menu = CustomCheckableMenu(parent=self, indicatorType=MenuIndicatorType.RADIO)

        # Only add actions for enabled engines
        actions_to_add = []

        if EngineType.CHATTERBOX.enabled:
            actions_to_add.append(self.chatterbox_action)
        if EngineType.VIBE.enabled:
            actions_to_add.append(self.vibe_action)
        if EngineType.QWEN3_TTS.enabled:
            actions_to_add.append(self.qwen_action)
        if EngineType.HIGGS.enabled:
            actions_to_add.append(self.higgs_action)
        if EngineType.GPT_SOVITS.enabled:
            actions_to_add.append(self.gpt_sovits_action)
        if EngineType.F5.enabled:
            actions_to_add.append(self.f5_action)
        if EngineType.RVC.enabled:
            actions_to_add.append(self.rvc_action)
        if EngineType.DIA.enabled:
            actions_to_add.append(self.dia_action)
        if EngineType.LLASA.enabled:
            actions_to_add.append(self.llasa_action)
        if EngineType.ORPHEUS.enabled:
            actions_to_add.append(self.orpheus_action)
        if EngineType.SPARK.enabled:
            actions_to_add.append(self.spark_action)
        if EngineType.CSM.enabled:
            actions_to_add.append(self.csm_action)
        if EngineType.FISH_SPEECH.enabled:
            actions_to_add.append(self.fish_action)
        if EngineType.DMOSPEECH2.enabled:
            actions_to_add.append(self.dmo_speech2_action)
        if EngineType.STYLE_TTS2.enabled:
            actions_to_add.append(self.styletts2_action)

        menu.addActions(actions_to_add)
        if pos is not None:
            menu.exec(pos, ani=True)
        return menu

    def createDeviceMenu(self, pos=None):
        menu = CustomCheckableMenu(parent=self, indicatorType=MenuIndicatorType.RADIO)
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                menu.addActions([
                    self.cpu_action,
                    self.gpu_action,
                    self.gpu2_action,
                    #self.clean_action
                ])
            else:
                menu.addActions([
                    self.cpu_action,
                    self.gpu_action,
                    #self.clean_action
                ])
        else:
            menu.addActions([
                self.cpu_action
            ])
        if pos is not None:
            menu.exec(pos, ani=True)
        return menu

    def createCommandBar_2(self):
        bar = CustomCommandBar(self)
        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        # add custom widget

        return bar

    def __reset(self):
        if cfg.get(cfg.engine) == EngineType.GPT_SOVITS.value:
            cfg.resetGPT()
        elif cfg.get(cfg.engine) == EngineType.STYLE_TTS2.value:
            cfg.resetStyleTTS()
        elif cfg.get(cfg.engine) == EngineType.F5.value:
            cfg.resetF5()
        elif cfg.get(cfg.engine) == EngineType.DIA.value:
            cfg.resetDIA()
        elif cfg.get(cfg.engine) == EngineType.FISH_SPEECH.value:
            cfg.resetDIA()
        elif cfg.get(cfg.engine) == EngineType.LLASA.value:
            cfg.resetLlasa()
        elif cfg.get(cfg.engine) == EngineType.ORPHEUS.value:
            cfg.resetOrpheus()
        elif cfg.get(cfg.engine) == EngineType.SPARK.value:
            cfg.resetSpark()
        elif cfg.get(cfg.engine) == EngineType.CSM.value:
            cfg.resetCSM()
        elif cfg.get(cfg.engine) == EngineType.HIGGS.value:
            cfg.resetHiggs()
        elif cfg.get(cfg.engine) == EngineType.CHATTERBOX.value:
            cfg.resetChatterbox()
        elif cfg.get(cfg.engine) == EngineType.DMOSPEECH2.value:
            cfg.resetDMSpeech2()
        elif cfg.get(cfg.engine) == EngineType.VIBE.value:
            cfg.resetVibe()
        elif cfg.get(cfg.engine) == EngineType.QWEN3_TTS.value:
            cfg.resetQwen()

        cfg.resetRvc()

    def createCommandBar(self):
        bar = CommandBar(self)

        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        bar.addWidget(self.label)
        bar.addSeparator()
        bar.addWidget(self.character_label)
        bar.addSeparator()
        bar.addWidget(self.reference_label)
        bar.addWidget(self.reference_time_label)
        bar.addSeparator()
        reset = Action(FIF.ROTATE, self.tr('Reset Generation Settings'))
        bar.addActions([reset])
        bar.addSeparator()
        reset.triggered.connect(self.__reset)
        button = TransparentDropDownPushButton(self.tr('Engine'), self, FIF.DEVELOPER_TOOLS)
        button.setMenu(self.createEngineMenu())
        button.setFixedHeight(34)
        setFont(button, 12)
        bar.addWidget(button)
        bar.addSeparator()

        cpu_button = TransparentDropDownPushButton(self.tr('Device'), self, FallTalkIcons.GPU.icon())
        cpu_button.setMenu(self.createDeviceMenu())
        cpu_button.setFixedHeight(34)
        setFont(cpu_button, 12)
        bar.addWidget(cpu_button)

        # spacer = QWidget()
        # spaceLayout = QVBoxLayout()
        # spaceLayout.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        # spacer.setLayout(spaceLayout)
        # bar.addWidget(StretchingWidget())

        # bar.addActions([
        #     Action(FIF.ADD, self.tr('Add')),
        #     Action(FIF.ROTATE, self.tr('Rotate')),
        #     Action(FIF.ZOOM_IN, self.tr('Zoom in')),
        #     Action(FIF.ZOOM_OUT, self.tr('Zoom out')),
        # ])
        # bar.addSeparator()
        # bar.addActions([
        #     Action(FIF.EDIT, self.tr('Edit'), checkable=True),
        #     Action(FIF.INFO, self.tr('Info')),
        #     Action(FIF.DELETE, self.tr('Delete')),
        #     Action(FIF.SHARE, self.tr('Share'))
        # ])

        # add custom widget
        # button = TransparentDropDownPushButton(self.tr('Sort'), self, FIF.SCROLL)
        # button.setMenu(self.createCheckableMenu())
        # button.setFixedHeight(34)
        # setFont(button, 12)
        # bar.addWidget(button)

        # bar.addHiddenActions([
        #     Action(FIF.SETTING, self.tr('Settings'), shortcut='Ctrl+I'),
        # ])
        return bar
