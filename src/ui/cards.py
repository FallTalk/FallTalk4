# coding:utf-8
from textwrap import dedent
from typing import Union

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon, QPainter, QColor
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QButtonGroup, QGroupBox, QSizePolicy, QVBoxLayout, QFrame
from qfluentwidgets import (ConfigItem, OptionsConfigItem, RangeConfigItem, SettingCard, FluentIconBase, SpinBox,
                            DoubleSpinBox, BodyLabel, ComboBox, LineEdit, Slider, RadioButton, qconfig,
                            ExpandSettingCard, FluentIcon as FIF, isDarkTheme)


class TextAreaCard(ExpandSettingCard):
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

class StyledSettingCard(SettingCard):

    def __init__(self, icon: Union[str, QIcon, FluentIconBase], title, content, parent=None):
        super().__init__(icon, title, content, parent)
        self.setContentsMargins(0, 0, 10, 0)


class TextCard(QFrame):
    def __init__(self, text: str, parent=None, height=100):
        super().__init__(parent)

        # Label setup
        self.label = QLabel(self)
        self.label.setText(dedent(text).strip())
        self.label.setWordWrap(True)
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.label.setContentsMargins(0, 0, 0, 0)

        # Layout setup
        self.vBoxLayout = QVBoxLayout(self)
        self.vBoxLayout.setContentsMargins(12, 12, 12, 12)
        self.vBoxLayout.setSpacing(0)
        self.vBoxLayout.addWidget(self.label)

        self.setMaximumHeight(height)

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(255, 255, 255, 170 if not isDarkTheme() else 13))
        painter.setPen(QColor(0, 0, 0, 19 if not isDarkTheme() else 50))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 6, 6)


class SpinSettingCard(StyledSettingCard):

    def __init__(self, configItem: RangeConfigItem, icon: Union[str, QIcon, FluentIconBase], title, content, parent=None, step=10):
        super().__init__(icon, title, content, parent)
        self.configItem = configItem
        sb = SpinBox()
        sb.setMinimumWidth(150)
        sb.setSingleStep(step)
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


class ComboBoxWordsCard(SettingCard):
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
        selected_word = self.configItem.itemData(value)
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
        self.valueLabel.setText(f" {value['start']} {value['end']}")
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


