from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QStackedWidget, QGroupBox, QHeaderView, QAbstractItemView, QSizePolicy, QSpacerItem, QWidget
from PySide6.QtCore import Qt
from qfluentwidgets import (
    FluentIcon as FIF, SearchLineEdit, TableView, PushButton, SegmentedWidget,
    TransparentDropDownPushButton, Action
)

from src.widgets.falltalk_widget import FallTalkWidget
from src.widgets.table_models import CustomReferencesModel
import os
import soundfile as sf

class ReferencesWidget(FallTalkWidget):
    """
    Widget for managing reference audio files.
    """
    def __init__(self, parent=None):
        super().__init__(text="Reference Audio", parent=parent, vertical=True)
        
        self.reference_audio = []
        self.reference_audio_length = 0
        
        self.segmented_widget = SegmentedWidget(self)
        self.segmented_widget.setObjectName("referencesSegmentedWidget")
        
        self.stackedWidget = QStackedWidget(self)
        self.stackedWidget.setObjectName("referencesStackedWidget")
        
        # Create references widget
        self.references_widget = QGroupBox(self)
        self.references_layout = QVBoxLayout(self.references_widget)
        
        self.references_search = SearchLineEdit(self)
        self.references_search.setPlaceholderText(self.tr("Search references"))
        self.references_search.textChanged.connect(self.apply_filter)
        
        self.references_table = TableView(self)
        self.references_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.references_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.references_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.references_table.doubleClicked.connect(self.on_reference_select)
        
        self.button_layout = QHBoxLayout()
        self.select_button = PushButton(self.tr("Select"), self, FIF.ACCEPT)
        self.select_button.clicked.connect(self.select_row)
        self.remove_button = PushButton(self.tr("Remove"), self, FIF.DELETE)
        self.remove_button.clicked.connect(self.remove_row)
        
        self.button_layout.addStretch(1)
        self.button_layout.addWidget(self.select_button)
        self.button_layout.addWidget(self.remove_button)
        self.button_layout.addStretch(1)
        
        self.references_layout.addWidget(self.references_search)
        self.references_layout.addWidget(self.references_table)
        self.references_layout.addLayout(self.button_layout)
        
        # Create custom references widget
        self.custom_references_widget = QGroupBox(self)
        self.custom_references_layout = QVBoxLayout(self.custom_references_widget)
        
        self.custom_references_search = SearchLineEdit(self)
        self.custom_references_search.setPlaceholderText(self.tr("Search custom references"))
        self.custom_references_search.textChanged.connect(self.apply_filter)
        
        self.custom_references_table = TableView(self)
        self.custom_references_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.custom_references_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.custom_references_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.custom_references_table.doubleClicked.connect(self.on_custom_reference_select)
        
        self.custom_button_layout = QHBoxLayout()
        self.custom_select_button = PushButton(self.tr("Select"), self, FIF.ACCEPT)
        self.custom_select_button.clicked.connect(self.select_row)
        
        self.custom_button_layout.addStretch(1)
        self.custom_button_layout.addWidget(self.custom_select_button)
        self.custom_button_layout.addStretch(1)
        
        self.custom_references_layout.addWidget(self.custom_references_search)
        self.custom_references_layout.addWidget(self.custom_references_table)
        self.custom_references_layout.addLayout(self.custom_button_layout)
        
        # Add widgets to stacked widget
        self.addSubInterface(self.references_widget, "referencesWidget", "References")
        self.addSubInterface(self.custom_references_widget, "customReferencesWidget", "Custom")
        
        self.segmented_widget.setCurrentItem("referencesWidget")
        self.segmented_widget.currentItemChanged.connect(self.onCurrentIndexChanged)
        
        self.main_layout.addWidget(self.segmented_widget)
        self.main_layout.addWidget(self.stackedWidget)
        
        # Add spacer at the bottom
        self.main_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

    def addDataToReferencesTable(self, selected_model):
        """Add data to references table"""
        headers = ["Name", "Length", "Count", "Path"]
        data = []
        
        if selected_model and 'references' in selected_model:
            for reference in selected_model['references']:
                if os.path.exists(reference):
                    try:
                        audio, sr = sf.read(reference)
                        length = len(audio) / sr
                        data.append([os.path.basename(reference), self._formatTime(length), 0, reference])
                    except Exception as e:
                        print(f"Error reading audio file: {e}")
        
        model = CustomReferencesModel(data, headers, self)
        self.references_table.setModel(model)

    def apply_filter(self):
        """Apply search filter to tables"""
        # Get current tab
        current_tab = self.segmented_widget.currentItem()
        
        # Apply filter based on current tab
        if current_tab == "referencesWidget":
            search_text = self.references_search.text().lower()
            for row in range(self.references_table.model().rowCount()):
                if search_text in self.references_table.model().data(self.references_table.model().index(row, 0)).lower():
                    self.references_table.showRow(row)
                else:
                    self.references_table.hideRow(row)
        elif current_tab == "customReferencesWidget":
            search_text = self.custom_references_search.text().lower()
            for row in range(self.custom_references_table.model().rowCount()):
                if search_text in self.custom_references_table.model().data(self.custom_references_table.model().index(row, 0)).lower():
                    self.custom_references_table.showRow(row)
                else:
                    self.custom_references_table.hideRow(row)

    def addSubInterface(self, widget: QWidget, objectName, text):
        """Add sub interface to widget"""
        widget.setObjectName(objectName)
        self.stackedWidget.addWidget(widget)
        self.segmented_widget.addItem(
            routeKey=objectName,
            text=text,
            onClick=lambda: self.stackedWidget.setCurrentWidget(widget)
        )

    def load_files_from_folder(self, folder_path):
        """Load audio files from a folder"""
        headers = ["Name", "Length", "Path"]
        data = []
        
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith((".wav", ".mp3", ".ogg")):
                    file_path = os.path.join(folder_path, file)
                    try:
                        audio, sr = sf.read(file_path)
                        length = len(audio) / sr
                        data.append([file, self._formatTime(length), file_path])
                    except Exception as e:
                        print(f"Error reading audio file: {e}")
        
        model = CustomReferencesModel(data, headers, self)
        self.custom_references_table.setModel(model)

    def onCurrentIndexChanged(self, index):
        """Handle change of current tab"""
        self.stackedWidget.setCurrentWidget(self.stackedWidget.findChild(QWidget, index))

    def on_custom_reference_select(self, index):
        """Handle custom reference selection"""
        row = index.row()
        model = self.custom_references_table.model()
        
        if row >= 0 and row < model.rowCount():
            file_path = model.data(model.index(row, 2))
            
            if file_path in self.reference_audio:
                self.decrease(file_path)
            else:
                self.reference_audio.append(file_path)
                self.increase(file_path)
                
                # Calculate total length
                try:
                    audio, sr = sf.read(file_path)
                    self.reference_audio_length += len(audio) / sr
                except Exception as e:
                    print(f"Error reading audio file: {e}")
            
            # Update UI
            if self.parent().tts_engine:
                if self.parent().tts_engine.engine_name == "RVC":
                    self.parent().rvc_widget.onReferenceSelect()
                elif self.parent().tts_engine.engine_name == "XTTS":
                    self.parent().xtts_widget.onReferenceSelect()
                elif self.parent().tts_engine.engine_name == "GPT_SoVITS":
                    self.parent().gpt_sovits_widget.onReferenceSelect()
                elif self.parent().tts_engine.engine_name == "StyleTTS2":
                    self.parent().styletts2_widget.onReferenceSelect()
                elif self.parent().tts_engine.engine_name == "DIA":
                    self.parent().dia_widget.onReferenceSelect()
                elif self.parent().tts_engine.engine_name == "Orpheus":
                    self.parent().orpheus_widget.onReferenceSelect()
                elif self.parent().tts_engine.engine_name == "Llasa":
                    self.parent().llasa_widget.onReferenceSelect()
                elif self.parent().tts_engine.engine_name == "F5":
                    self.parent().f5_widget.onReferenceSelect()
                elif self.parent().tts_engine.engine_name == "FishSpeech":
                    self.parent().fish_widget.onReferenceSelect()

    def on_reference_select(self, index):
        """Handle reference selection"""
        row = index.row()
        model = self.references_table.model()
        
        if row >= 0 and row < model.rowCount():
            file_path = model.data(model.index(row, 3))
            
            if file_path in self.reference_audio:
                self.decrease(file_path)
                
                # Update count in table
                count = int(model.data(model.index(row, 2)))
                if count > 0:
                    model._data[row][2] = count - 1
                    model.dataChanged.emit(model.index(row, 2), model.index(row, 2))
                
                # Calculate total length
                try:
                    audio, sr = sf.read(file_path)
                    self.reference_audio_length -= len(audio) / sr
                except Exception as e:
                    print(f"Error reading audio file: {e}")
            else:
                self.reference_audio.append(file_path)
                self.increase(file_path)
                
                # Update count in table
                count = int(model.data(model.index(row, 2)))
                model._data[row][2] = count + 1
                model.dataChanged.emit(model.index(row, 2), model.index(row, 2))
                
                # Calculate total length
                try:
                    audio, sr = sf.read(file_path)
                    self.reference_audio_length += len(audio) / sr
                except Exception as e:
                    print(f"Error reading audio file: {e}")
            
            # Update UI
            if self.parent().tts_engine:
                if self.parent().tts_engine.engine_name == "RVC":
                    self.parent().rvc_widget.onReferenceSelect()
                elif self.parent().tts_engine.engine_name == "XTTS":
                    self.parent().xtts_widget.onReferenceSelect()
                elif self.parent().tts_engine.engine_name == "GPT_SoVITS":
                    self.parent().gpt_sovits_widget.onReferenceSelect()
                elif self.parent().tts_engine.engine_name == "StyleTTS2":
                    self.parent().styletts2_widget.onReferenceSelect()
                elif self.parent().tts_engine.engine_name == "DIA":
                    self.parent().dia_widget.onReferenceSelect()
                elif self.parent().tts_engine.engine_name == "Orpheus":
                    self.parent().orpheus_widget.onReferenceSelect()
                elif self.parent().tts_engine.engine_name == "Llasa":
                    self.parent().llasa_widget.onReferenceSelect()
                elif self.parent().tts_engine.engine_name == "F5":
                    self.parent().f5_widget.onReferenceSelect()
                elif self.parent().tts_engine.engine_name == "FishSpeech":
                    self.parent().fish_widget.onReferenceSelect()

    def clear(self):
        """Clear references"""
        self.reference_audio = []
        self.reference_audio_length = 0
        
        # Clear references table
        if hasattr(self, 'references_table') and self.references_table.model():
            for row in range(self.references_table.model().rowCount()):
                self.references_table.model()._data[row][2] = 0
            self.references_table.model().dataChanged.emit(
                self.references_table.model().index(0, 2),
                self.references_table.model().index(self.references_table.model().rowCount() - 1, 2)
            )
        
        # Clear custom references table
        if hasattr(self, 'custom_references_table') and self.custom_references_table.model():
            self.custom_references_table.model().selected_rows.clear()
            self.custom_references_table.model().dataChanged.emit(
                self.custom_references_table.model().index(0, 0),
                self.custom_references_table.model().index(
                    self.custom_references_table.model().rowCount() - 1,
                    self.custom_references_table.model().columnCount() - 1
                )
            )

    def select_row(self):
        """Select the current row"""
        # Get current tab
        current_tab = self.segmented_widget.currentItem()
        
        # Handle selection based on current tab
        if current_tab == "referencesWidget":
            selected_indexes = self.references_table.selectedIndexes()
            if selected_indexes:
                row = selected_indexes[0].row()
                model = self.references_table.model()
                
                if row >= 0 and row < model.rowCount():
                    file_path = model.data(model.index(row, 3))
                    
                    if file_path in self.reference_audio:
                        self.decrease(file_path)
                        
                        # Update count in table
                        count = int(model.data(model.index(row, 2)))
                        if count > 0:
                            model._data[row][2] = count - 1
                            model.dataChanged.emit(model.index(row, 2), model.index(row, 2))
                    else:
                        self.reference_audio.append(file_path)
                        self.increase(file_path)
                        
                        # Update count in table
                        count = int(model.data(model.index(row, 2)))
                        model._data[row][2] = count + 1
                        model.dataChanged.emit(model.index(row, 2), model.index(row, 2))
        elif current_tab == "customReferencesWidget":
            selected_indexes = self.custom_references_table.selectedIndexes()
            if selected_indexes:
                row = selected_indexes[0].row()
                model = self.custom_references_table.model()
                
                if row >= 0 and row < model.rowCount():
                    model.toggle_selection(row)
                    file_path = model.data(model.index(row, 2))
                    
                    if file_path in self.reference_audio:
                        self.decrease(file_path)
                    else:
                        self.reference_audio.append(file_path)
                        self.increase(file_path)

    def increase(self, file_path):
        """Increase reference count and length"""
        try:
            audio, sr = sf.read(file_path)
            self.reference_audio_length += len(audio) / sr
        except Exception as e:
            print(f"Error reading audio file: {e}")

    def remove_row(self):
        """Remove the selected row from references"""
        # Get current tab
        current_tab = self.segmented_widget.currentItem()
        
        # Handle removal based on current tab
        if current_tab == "referencesWidget":
            selected_indexes = self.references_table.selectedIndexes()
            if selected_indexes:
                row = selected_indexes[0].row()
                model = self.references_table.model()
                
                if row >= 0 and row < model.rowCount():
                    file_path = model.data(model.index(row, 3))
                    
                    if file_path in self.reference_audio:
                        self.reference_audio.remove(file_path)
                        
                        # Update count in table
                        model._data[row][2] = 0
                        model.dataChanged.emit(model.index(row, 2), model.index(row, 2))
                        
                        # Calculate total length
                        try:
                            audio, sr = sf.read(file_path)
                            self.reference_audio_length -= len(audio) / sr
                        except Exception as e:
                            print(f"Error reading audio file: {e}")
        elif current_tab == "customReferencesWidget":
            selected_indexes = self.custom_references_table.selectedIndexes()
            if selected_indexes:
                row = selected_indexes[0].row()
                model = self.custom_references_table.model()
                
                if row >= 0 and row < model.rowCount():
                    file_path = model.data(model.index(row, 2))
                    
                    if file_path in self.reference_audio:
                        self.reference_audio.remove(file_path)
                        model.toggle_selection(row)
                        
                        # Calculate total length
                        try:
                            audio, sr = sf.read(file_path)
                            self.reference_audio_length -= len(audio) / sr
                        except Exception as e:
                            print(f"Error reading audio file: {e}")

    def decrease(self, file_path):
        """Decrease reference count and length"""
        if file_path in self.reference_audio:
            self.reference_audio.remove(file_path)
            
            # Calculate total length
            try:
                audio, sr = sf.read(file_path)
                self.reference_audio_length -= len(audio) / sr
            except Exception as e:
                print(f"Error reading audio file: {e}")

    def _formatTime(self, time: float):
        """Format time in seconds to MM:SS format"""
        minutes = int(time // 60)
        seconds = int(time % 60)
        return f"{minutes:02d}:{seconds:02d}"