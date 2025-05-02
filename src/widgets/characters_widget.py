import json
import os

from PySide6.QtWidgets import QVBoxLayout, QStackedWidget, QGroupBox, QHeaderView, QAbstractItemView, QWidget
from qfluentwidgets import (
    FluentIcon as FIF, SearchLineEdit, TableView, PushButton, SegmentedWidget,
    TransparentDropDownPushButton, Action
)

from src.widgets.custom_message_box import CustomMessageBox
from src.widgets.falltalk_widget import FallTalkWidget
from src.widgets.table_models import CharacterTableModel


class CharactersWidget(FallTalkWidget):
    """
    Widget for managing character models.
    """
    def __init__(self, parent=None):
        super().__init__(text="Character Models", parent=parent, vertical=True)
        
        self.segmented_widget = SegmentedWidget(self)
        self.segmented_widget.setObjectName("charactersSegmentedWidget")
        
        self.stackedWidget = QStackedWidget(self)
        self.stackedWidget.setObjectName("charactersStackedWidget")
        
        # Create trained characters widget
        self.trained_widget = QGroupBox(self)
        self.trained_layout = QVBoxLayout(self.trained_widget)
        
        self.trained_search = SearchLineEdit(self)
        self.trained_search.setPlaceholderText(self.tr("Search trained characters"))
        self.trained_search.textChanged.connect(self.apply_filter)
        
        self.trained_table = TableView(self)
        self.trained_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.trained_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.trained_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.trained_table.horizontalHeader().sectionClicked.connect(self.on_header_clicked)
        self.trained_table.doubleClicked.connect(self.select_row)
        
        self.trained_layout.addWidget(self.trained_search)
        self.trained_layout.addWidget(self.trained_table)
        
        # Create untrained characters widget
        self.untrained_widget = QGroupBox(self)
        self.untrained_layout = QVBoxLayout(self.untrained_widget)
        
        self.untrained_search = SearchLineEdit(self)
        self.untrained_search.setPlaceholderText(self.tr("Search untrained characters"))
        self.untrained_search.textChanged.connect(self.apply_filter)
        
        self.untrained_table = TableView(self)
        self.untrained_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.untrained_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.untrained_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.untrained_table.horizontalHeader().sectionClicked.connect(self.on_header_clicked)
        self.untrained_table.doubleClicked.connect(self.select_row)
        
        self.untrained_layout.addWidget(self.untrained_search)
        self.untrained_layout.addWidget(self.untrained_table)
        
        # Create custom characters widget
        self.custom_widget = QGroupBox(self)
        self.custom_layout = QVBoxLayout(self.custom_widget)
        
        self.custom_search = SearchLineEdit(self)
        self.custom_search.setPlaceholderText(self.tr("Search custom characters"))
        self.custom_search.textChanged.connect(self.apply_filter)
        
        self.custom_table = TableView(self)
        self.custom_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.custom_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.custom_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.custom_table.horizontalHeader().sectionClicked.connect(self.on_header_clicked)
        self.custom_table.doubleClicked.connect(self.select_row)
        
        self.add_custom_button = PushButton(self.tr("Add Custom"), self, FIF.ADD)
        self.add_custom_button.clicked.connect(self.add_custom)
        
        self.custom_layout.addWidget(self.custom_search)
        self.custom_layout.addWidget(self.custom_table)
        self.custom_layout.addWidget(self.add_custom_button)
        
        # Add widgets to stacked widget
        self.addSubInterface(self.trained_widget, "trainedWidget", "Trained")
        self.addSubInterface(self.untrained_widget, "untrainedWidget", "Untrained")
        self.addSubInterface(self.custom_widget, "customWidget", "Custom")
        
        self.segmented_widget.setCurrentItem("trainedWidget")
        self.segmented_widget.currentItemChanged.connect(self.onCurrentIndexChanged)
        
        self.main_layout.addWidget(self.segmented_widget)
        self.main_layout.addWidget(self.stackedWidget)

    def onCurrentIndexChanged(self, index):
        """Handle change of current tab"""
        if index == "trainedWidget":
            self.stackedWidget.setCurrentWidget(self.trained_widget)
        elif index == "untrainedWidget":
            self.stackedWidget.setCurrentWidget(self.untrained_widget)
        elif index == "customWidget":
            self.stackedWidget.setCurrentWidget(self.custom_widget)

    def verify(self, obj):
        """Verify if an object is valid"""
        if obj is None:
            return False
        if isinstance(obj, str) and obj == "":
            return False
        if isinstance(obj, list) and len(obj) == 0:
            return False
        if isinstance(obj, dict) and len(obj) == 0:
            return False
        return True

    def add_custom(self):
        """Add a custom character"""
        dialog = CustomMessageBox(self)
        dialog.yesButton.setText(self.tr("Add"))
        dialog.yesButton.setEnabled(False)
        
        if dialog.exec():
            name = dialog.name_input.text()
            display_name = dialog.display_name_input.text()
            index_path = dialog.index_card.configItem.text()
            model_path = dialog.path_card.configItem.text()
            ckpt_path = dialog.ckpt_card.configItem.text()
            
            # Create model directory
            os.makedirs(f"models/{name}/RVC", exist_ok=True)
            
            # Create custom model entry
            custom_model = {
                "name": name,
                "display_name": display_name,
                "RVC": {
                    "index": index_path,
                    "model": model_path,
                    "ckpt": ckpt_path
                }
            }
            
            # Load existing custom models or create new list
            if os.path.exists('config/custom_models.json'):
                with open('config/custom_models.json', 'r', encoding="utf-8") as file:
                    custom_models = json.load(file)
            else:
                custom_models = []
            
            # Add new model and save
            custom_models.append(custom_model)
            with open('config/custom_models.json', 'w', encoding="utf-8") as file:
                json.dump(custom_models, file)
            
            # Reload models
            self.parent().load_models_config()

    def find_versions(self, directory_path, character_model_name, model_type):
        """Find versions of a model"""
        versions = []
        if os.path.exists(directory_path):
            for file in os.listdir(directory_path):
                if file.endswith(".pth") and character_model_name in file and model_type in file:
                    versions.append(file)
        return versions

    def clear(self):
        """Clear all tables"""
        # Clear trained table
        if hasattr(self, 'trained_table') and self.trained_table.model():
            self.trained_table.setModel(None)
        
        # Clear untrained table
        if hasattr(self, 'untrained_table') and self.untrained_table.model():
            self.untrained_table.setModel(None)
        
        # Clear custom table
        if hasattr(self, 'custom_table') and self.custom_table.model():
            self.custom_table.setModel(None)

    def loadTrained(self, parent, trained_characters):
        """Load trained characters into the table"""
        headers = ["Name", "Engine", "Actions"]
        data = []
        
        for character in trained_characters:
            if not self.verify(character):
                continue
            
            # Create actions button
            actions_button = TransparentDropDownPushButton(self.tr('Actions'), self.trained_table)
            
            # Create actions menu
            load_action = Action(FIF.PLAY, self.tr('Load'))
            load_action.triggered.connect(lambda checked=False, c=character: parent.load_trained_model(c['name'], c, c['RVC']))
            
            download_action = Action(FIF.DOWNLOAD, self.tr('Download'))
            download_action.triggered.connect(lambda checked=False, c=character: parent.download_model(c['name'], c[parent.cfg.get(parent.cfg.engine)], c['RVC']))
            
            update_action = Action(FIF.UPDATE, self.tr('Update'))
            update_action.triggered.connect(lambda checked=False, c=character: parent.update_model(c['name'], c[parent.cfg.get(parent.cfg.engine)], c['RVC']))
            
            delete_action = Action(FIF.DELETE, self.tr('Delete'))
            delete_action.triggered.connect(lambda checked=False, c=character, d=character['display_name']: parent.delete_model(c['name'], c[parent.cfg.get(parent.cfg.engine)], d))
            
            # Add actions to button
            actions_button.addAction(load_action)
            actions_button.addAction(download_action)
            actions_button.addAction(update_action)
            actions_button.addAction(delete_action)
            
            # Add row to data
            data.append([character['display_name'], parent.cfg.get(parent.cfg.engine).value, actions_button])
        
        # Create model and set it to table
        model = CharacterTableModel(data, headers, self)
        self.trained_table.setModel(model)
        
        # Set row height for buttons
        for row in range(len(data)):
            self.trained_table.setRowHeight(row, 40)
            if data[row][2] is not None:
                self.trained_table.setIndexWidget(model.index(row, 2), data[row][2])

    def loadCustom(self, parent, custom_characters):
        """Load custom characters into the table"""
        if not custom_characters:
            return
            
        headers = ["Name", "Actions"]
        data = []
        
        for name, character in custom_characters.items():
            if not self.verify(character):
                continue
            
            # Create actions button
            actions_button = TransparentDropDownPushButton(self.tr('Actions'), self.custom_table)
            
            # Create actions menu
            load_action = Action(FIF.PLAY, self.tr('Load'))
            load_action.triggered.connect(lambda checked=False, c=character: parent.load_custom_model(c['name'], c, c['RVC']))
            
            delete_action = Action(FIF.DELETE, self.tr('Delete'))
            delete_action.triggered.connect(lambda checked=False, c=character, d=character['display_name']: parent.delete_custom_model(c['name'], d))
            
            # Add actions to button
            actions_button.addAction(load_action)
            actions_button.addAction(delete_action)
            
            # Add row to data
            data.append([character['display_name'], actions_button])
        
        # Create model and set it to table
        model = CharacterTableModel(data, headers, self)
        self.custom_table.setModel(model)
        
        # Set row height for buttons
        for row in range(len(data)):
            self.custom_table.setRowHeight(row, 40)
            if data[row][1] is not None:
                self.custom_table.setIndexWidget(model.index(row, 1), data[row][1])

    def loadUntrained(self, parent, untrained_characters):
        """Load untrained characters into the table"""
        headers = ["Name", "Actions"]
        data = []
        
        for character in untrained_characters:
            if not self.verify(character):
                continue
            
            # Create actions button
            actions_button = TransparentDropDownPushButton(self.tr('Actions'), self.untrained_table)
            
            # Create actions menu
            load_action = Action(FIF.PLAY, self.tr('Load Base Model'))
            load_action.triggered.connect(lambda checked=False, c=character: parent.load_base_model(c, c['RVC']))
            
            # Add actions to button
            actions_button.addAction(load_action)
            
            # Add row to data
            data.append([character['display_name'], actions_button])
        
        # Create model and set it to table
        model = CharacterTableModel(data, headers, self)
        self.untrained_table.setModel(model)
        
        # Set row height for buttons
        for row in range(len(data)):
            self.untrained_table.setRowHeight(row, 40)
            if data[row][1] is not None:
                self.untrained_table.setIndexWidget(model.index(row, 1), data[row][1])

    def on_header_clicked(self, index):
        """Handle header click for sorting"""
        pass

    def apply_filter(self):
        """Apply search filter to tables"""
        # Get current tab
        current_tab = self.segmented_widget.currentItem()
        
        # Apply filter based on current tab
        if current_tab == "trainedWidget":
            search_text = self.trained_search.text().lower()
            for row in range(self.trained_table.model().rowCount()):
                if search_text in self.trained_table.model().data(self.trained_table.model().index(row, 0)).lower():
                    self.trained_table.showRow(row)
                else:
                    self.trained_table.hideRow(row)
        elif current_tab == "untrainedWidget":
            search_text = self.untrained_search.text().lower()
            for row in range(self.untrained_table.model().rowCount()):
                if search_text in self.untrained_table.model().data(self.untrained_table.model().index(row, 0)).lower():
                    self.untrained_table.showRow(row)
                else:
                    self.untrained_table.hideRow(row)
        elif current_tab == "customWidget":
            search_text = self.custom_search.text().lower()
            for row in range(self.custom_table.model().rowCount()):
                if search_text in self.custom_table.model().data(self.custom_table.model().index(row, 0)).lower():
                    self.custom_table.showRow(row)
                else:
                    self.custom_table.hideRow(row)

    def addSubInterface(self, widget: QWidget, objectName, text):
        """Add sub interface to widget"""
        widget.setObjectName(objectName)
        self.stackedWidget.addWidget(widget)
        self.segmented_widget.addItem(
            routeKey=objectName,
            text=text,
            onClick=lambda: self.stackedWidget.setCurrentWidget(widget)
        )