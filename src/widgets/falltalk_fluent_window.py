from typing import Union

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QWidget
from qfluentwidgets import FluentIconBase, NavigationItemPosition, NavigationInterface, FluentTitleBar
from qfluentwidgets.window.fluent_window import FluentWindowBase
from qfluentwidgets import CheckableMenu, MenuIndicatorType, Action, CommandBar

class CustomCommandBar(CommandBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

    def _visibleWidgets(self):
        return [a for a in self.actions() if not a.isSeparator() and a.isVisible()]

class FallTalkFluentWindow(FluentWindowBase):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setTitleBar(FluentTitleBar(self))
        self.navigationInterface = NavigationInterface(self)
        self.navigationInterface.setExpandWidth(250)
        self.stackedWidget = QWidget(self)
        self.initLayout()

        self.toolbar_1 = CustomCommandBar(self)
        self.titleBar.customTitleBar.addWidget(self.toolbar_1)

        self.engine_menu = CheckableMenu(self.toolbar_1)
        self.engine_menu.setIndicatorType(MenuIndicatorType.CHECK)

        self.device_menu = CheckableMenu(self.toolbar_1)
        self.device_menu.setIndicatorType(MenuIndicatorType.CHECK)

        self.rvc_action = Action(self.tr('RVC'), self.toolbar_1)
        self.rvc_action.setCheckable(True)
        self.xtts_action = Action(self.tr('XTTS v2'), self.toolbar_1)
        self.xtts_action.setCheckable(True)
        self.styletts2_action = Action(self.tr('StyleTTS2'), self.toolbar_1)
        self.styletts2_action.setCheckable(True)
        self.dia_action = Action(self.tr('DIA'), self.toolbar_1)
        self.dia_action.setCheckable(True)
        self.f5_action = Action(self.tr('F5'), self.toolbar_1)
        self.f5_action.setCheckable(True)
        self.fish_action = Action(self.tr('FishSpeech'), self.toolbar_1)
        self.fish_action.setCheckable(True)
        self.orpheus_action = Action(self.tr('Orpheus'), self.toolbar_1)
        self.orpheus_action.setCheckable(True)
        self.llasa_action = Action(self.tr('Llasa'), self.toolbar_1)
        self.llasa_action.setCheckable(True)
        self.gpt_sovits_action = Action(self.tr('GPT-SoVITS'), self.toolbar_1)
        self.gpt_sovits_action.setCheckable(True)

        self.cpu_action = Action(self.tr('CPU'), self.toolbar_1)
        self.cpu_action.setCheckable(True)
        self.gpu_action = Action(self.tr('GPU 0'), self.toolbar_1)
        self.gpu_action.setCheckable(True)
        self.gpu2_action = Action(self.tr('GPU 1'), self.toolbar_1)
        self.gpu2_action.setCheckable(True)

        self.update_action = Action(self.tr('Update Available'), self.toolbar_1)
        self.new_models_action = Action(self.tr('New Models Available'), self.toolbar_1)

        self.character_label = Action(self.tr('Please Load Model'), self.toolbar_1)
        self.character_label.setEnabled(False)

    def _onCurrentInterfaceChanged(self, index: int):
        widget = self.stackedWidget.widget(index)
        self.stackedWidget.setCurrentWidget(widget)

    def addSubInterface(self, interface: QWidget, icon: Union[FluentIconBase, QIcon, str], text: str, position=NavigationItemPosition.TOP, parent=None, isTransparent=False):
        """Add sub interface to window

        Parameters
        ----------
        interface: QWidget
            the sub interface to be added

        icon: FluentIconBase | QIcon | str
            the icon of navigation item

        text: str
            the text of navigation item

        position: NavigationItemPosition
            the position of navigation item

        parent: str
            the text of parent item

        isTransparent: bool
            whether to use transparent background
        """
        self.stackedWidget.addWidget(interface)
        self.navigationInterface.addItem(
            routeKey=interface.objectName(),
            icon=icon,
            text=text,
            onClick=lambda: self._onCurrentInterfaceChanged(self.stackedWidget.indexOf(interface)),
            position=position,
            tooltip=text,
            parentRouteKey=parent,
            isTransparent=isTransparent
        )

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.titleBar.move(0, 0)

    def createEngineMenu(self, pos=None):
        self.engine_menu.clear()
        self.engine_menu.addAction(self.rvc_action)
        self.engine_menu.addAction(self.xtts_action)
        self.engine_menu.addAction(self.styletts2_action)
        self.engine_menu.addAction(self.dia_action)
        self.engine_menu.addAction(self.f5_action)
        self.engine_menu.addAction(self.fish_action)
        self.engine_menu.addAction(self.orpheus_action)
        self.engine_menu.addAction(self.llasa_action)
        self.engine_menu.addAction(self.gpt_sovits_action)
        return self.engine_menu

    def createDeviceMenu(self, pos=None):
        self.device_menu.clear()
        self.device_menu.addAction(self.cpu_action)
        self.device_menu.addAction(self.gpu_action)
        self.device_menu.addAction(self.gpu2_action)
        return self.device_menu

    def createCommandBar_2(self):
        self.toolbar_2 = CustomCommandBar(self)
        self.titleBar.customTitleBar.addWidget(self.toolbar_2)
        return self.toolbar_2

    def __reset(self):
        self.toolbar_1.clear()
        self.toolbar_1.addAction(self.character_label)
        self.toolbar_1.addSeparator()

        self.engine_button = self.toolbar_1.addDropDownAction('Engine', self.createEngineMenu)
        self.engine_button.setToolTip('Select Engine')
        self.engine_button.setIcon(QIcon('resource/icons/engine.svg'))

        self.device_button = self.toolbar_1.addDropDownAction('Device', self.createDeviceMenu)
        self.device_button.setToolTip('Select Device')
        self.device_button.setIcon(QIcon('resource/icons/device.svg'))

    def createCommandBar(self):
        self.__reset()
        return self.toolbar_1