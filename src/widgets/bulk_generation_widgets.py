from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget, QGroupBox, QHeaderView, QAbstractItemView, QFileDialog
from qfluentwidgets import (
    FluentIcon as FIF, SearchLineEdit, TableView, PushButton, SegmentedWidget,
    SwitchSettingCard
)

from src.widgets.falltalk_widget import FallTalkWidget
from src.config.config import  cfg
from src.widgets.table_models import CustomTableModel
from src.ui.cards import TextSettingCard, SpinSettingCard, ComboBoxSettingsCard


class BulkLipFuzWidget(QWidget):
    """
    Widget for bulk lip synchronization file generation.
    """
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create lip directory card
        self.lip_dir = TextSettingCard(
            cfg.lip_dir,
            FIF.FOLDER,
            self.tr('Lip Directory'),
            self.tr('Directory containing lip files to process'),
            parent=self
        )
        self.lip_dir.clicked.connect(self.__onFolderCardClicked)
        
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
        
        # Create threads card
        self.threads_card = SpinSettingCard(
            cfg.threads,
            FIF.SPEED_HIGH,
            self.tr('Threads'),
            self.tr('Number of threads to use for processing'),
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
        
        # Add widgets to layout
        self.main_layout.addWidget(self.lip_dir)
        self.main_layout.addWidget(self.switches_group)
        self.main_layout.addWidget(self.threads_card)

    def __onFolderCardClicked(self):
        """Handle folder card click to select lip directory"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Lip Directory"),
            ""
        )
        if folder_path:
            self.lip_dir.configItem.setText(folder_path)

class BulkGenerationRVCWidget(QWidget):
    """
    Widget for bulk RVC generation.
    """
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create RVC directory card
        self.rvc_dir = TextSettingCard(
            cfg.rvc_dir,
            FIF.FOLDER,
            self.tr('RVC Directory'),
            self.tr('Directory containing audio files to process'),
            parent=self
        )
        self.rvc_dir.clicked.connect(self.__onFolderCardClicked)
        
        # Create character card
        self.character_card = ComboBoxSettingsCard(
            cfg.bulk_character,
            FIF.PEOPLE,
            self.tr('Character'),
            self.tr('Character to use for RVC'),
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
        
        # Create use existing lip switch
        self.use_existing_lip = SwitchSettingCard(
            cfg.use_existing_lip,
            FIF.FOLDER,
            self.tr('Use Existing Lip'),
            self.tr('Use existing lip files if available'),
            parent=self
        )
        
        # Create threads card
        self.threads_card = SpinSettingCard(
            cfg.threads,
            FIF.SPEED_HIGH,
            self.tr('Threads'),
            self.tr('Number of threads to use for processing'),
            parent=self
        )
        
        # Create switches group
        self.switches_group = QGroupBox()
        self.switches_group.setStyleSheet("border: none")
        self.switches_layout = QHBoxLayout()
        self.switches_layout.setContentsMargins(0, 0, 0, 0)
        self.switches_layout.addWidget(self.replace_existing)
        self.switches_layout.addWidget(self.include_subdir)
        self.switches_layout.addWidget(self.use_existing_lip)
        self.switches_group.setLayout(self.switches_layout)
        
        # Add widgets to layout
        self.main_layout.addWidget(self.rvc_dir)
        self.main_layout.addWidget(self.character_card)
        self.main_layout.addWidget(self.switches_group)
        self.main_layout.addWidget(self.threads_card)

    def __onFolderCardClicked(self):
        """Handle folder card click to select RVC directory"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select RVC Directory"),
            ""
        )
        if folder_path:
            self.rvc_dir.configItem.setText(folder_path)

class BulkGenerationTableWidget(QWidget):
    """
    Widget for bulk generation from a table of data.
    """
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create search field
        self.search = SearchLineEdit(self)
        self.search.setPlaceholderText(self.tr("Search bulk generation data"))
        self.search.textChanged.connect(self.apply_filter)
        
        # Create bulk table
        self.bulk_table = TableView(self)
        self.bulk_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.bulk_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.bulk_table.setSelectionMode(QAbstractItemView.SingleSelection)
        
        # Create buttons
        self.button_layout = QHBoxLayout()
        self.load_button = PushButton(self.tr("Load CSV"), self, FIF.FOLDER)
        self.clear_button = PushButton(self.tr("Clear"), self, FIF.DELETE)
        
        self.button_layout.addStretch(1)
        self.button_layout.addWidget(self.load_button)
        self.button_layout.addWidget(self.clear_button)
        self.button_layout.addStretch(1)
        
        # Add widgets to layout
        self.main_layout.addWidget(self.search)
        self.main_layout.addWidget(self.bulk_table)
        self.main_layout.addLayout(self.button_layout)
        
        # Set up model
        headers = ["Character", "Text", "Output Name", "Selected"]
        data = []
        model = CustomTableModel(data, headers, self)
        self.bulk_table.setModel(model)

    def apply_filter(self):
        """Apply search filter to table"""
        search_text = self.search.text().lower()
        for row in range(self.bulk_table.model().rowCount()):
            show_row = False
            for col in range(self.bulk_table.model().columnCount() - 1):  # Skip the "Selected" column
                cell_text = self.bulk_table.model().data(self.bulk_table.model().index(row, col))
                if cell_text and search_text in cell_text.lower():
                    show_row = True
                    break
            
            if show_row:
                self.bulk_table.showRow(row)
            else:
                self.bulk_table.hideRow(row)

class BulkGenerationWidget(FallTalkWidget):
    """
    Main widget for bulk generation functionality.
    """
    def __init__(self, parent=None):
        super().__init__(text="Bulk Generation", parent=parent, vertical=True)
        
        self.segmented_widget = SegmentedWidget(self)
        self.segmented_widget.setObjectName("bulkGenerationSegmentedWidget")
        
        self.stackedWidget = QStackedWidget(self)
        self.stackedWidget.setObjectName("bulkGenerationStackedWidget")
        
        # Create sub-widgets
        self.bulk_csv_widget = BulkGenerationTableWidget(self)
        self.bulk_rvc_widget = BulkGenerationRVCWidget(self)
        self.bulk_fuz_widget = BulkLipFuzWidget(self)
        
        # Add widgets to stacked widget
        self.addSubInterface(self.bulk_csv_widget, "bulkCsvWidget", "CSV")
        self.addSubInterface(self.bulk_rvc_widget, "bulkRvcWidget", "RVC")
        self.addSubInterface(self.bulk_fuz_widget, "bulkFuzWidget", "FUZ")
        
        self.segmented_widget.setCurrentItem("bulkCsvWidget")
        self.segmented_widget.currentItemChanged.connect(self.onCurrentIndexChanged)
        
        # Create output directory card
        self.output_dir = TextSettingCard(
            cfg.output_dir,
            FIF.FOLDER,
            self.tr('Output Directory'),
            self.tr('Directory to save generated files'),
            parent=self
        )
        self.output_dir.clicked.connect(self.__onOutputFolderCardClicked)
        
        # Create generate button
        self.generate_button = PushButton(self.tr("Generate"), self, FIF.PLAY)
        self.generate_button.clicked.connect(lambda: self.parent().bulk_inference())
        
        self.button_layout = QHBoxLayout()
        self.button_layout.addStretch(1)
        self.button_layout.addWidget(self.generate_button)
        self.button_layout.addStretch(1)
        
        # Add widgets to layout
        self.main_layout.addWidget(self.segmented_widget)
        self.main_layout.addWidget(self.stackedWidget)
        self.main_layout.addWidget(self.output_dir)
        self.main_layout.addLayout(self.button_layout)

    def onCurrentIndexChanged(self, index):
        """Handle change of current tab"""
        if index == "bulkCsvWidget":
            self.stackedWidget.setCurrentWidget(self.bulk_csv_widget)
        elif index == "bulkRvcWidget":
            self.stackedWidget.setCurrentWidget(self.bulk_rvc_widget)
        elif index == "bulkFuzWidget":
            self.stackedWidget.setCurrentWidget(self.bulk_fuz_widget)

    def addSubInterface(self, widget: QWidget, objectName, text):
        """Add sub interface to widget"""
        widget.setObjectName(objectName)
        self.stackedWidget.addWidget(widget)
        self.segmented_widget.addItem(
            routeKey=objectName,
            text=text,
            onClick=lambda: self.stackedWidget.setCurrentWidget(widget)
        )

    def populate_character_card(self):
        """Populate the character card with available characters"""
        if hasattr(self, 'bulk_rvc_widget') and hasattr(self.bulk_rvc_widget, 'character_card'):
            self.bulk_rvc_widget.character_card.comboBox.clear()
            
            if self.parent().characters_data:
                for character_name, character in self.parent().characters_data.items():
                    if 'display_name' in character:
                        self.bulk_rvc_widget.character_card.comboBox.addItem(character['display_name'], character)

    def __onShowGettingStarted(self):
        """Show getting started information"""
        pass

    def __onOutputFolderCardClicked(self):
        """Handle output folder card click to select output directory"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Output Directory"),
            ""
        )
        if folder_path:
            self.output_dir.configItem.setText(folder_path)
            cfg.set(cfg.output_dir, folder_path)

    def clear(self):
        """Clear bulk generation data"""
        if hasattr(self, 'bulk_csv_widget') and hasattr(self.bulk_csv_widget, 'bulk_table'):
            self.bulk_csv_widget.bulk_table.setModel(None)
            headers = ["Character", "Text", "Output Name", "Selected"]
            data = []
            model = CustomTableModel(data, headers, self)
            self.bulk_csv_widget.bulk_table.setModel(model)