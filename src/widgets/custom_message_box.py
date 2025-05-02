from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QFileDialog
from qfluentwidgets import MessageBoxBase, FluentIcon as FIF

from ui.cards import TextSettingCard


class CustomMessageBox(MessageBoxBase):
    """
    Custom message box with additional settings.
    """
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.titleLabel.setText(self.tr('Add Custom Character'))
        
        # Create layout for content
        self.vBoxLayout = QVBoxLayout(self.widget)
        self.vBoxLayout.setContentsMargins(20, 20, 20, 20)
        
        # Add title label
        self.vBoxLayout.addWidget(self.titleLabel)
        
        # Add name input
        self.name_label = QLabel(self.tr("Character Name:"), self)
        self.name_input = QLineEdit(self)
        self.name_input.setPlaceholderText(self.tr("Enter character name"))
        
        self.name_layout = QHBoxLayout()
        self.name_layout.addWidget(self.name_label)
        self.name_layout.addWidget(self.name_input)
        
        self.vBoxLayout.addLayout(self.name_layout)
        
        # Add display name input
        self.display_name_label = QLabel(self.tr("Display Name:"), self)
        self.display_name_input = QLineEdit(self)
        self.display_name_input.setPlaceholderText(self.tr("Enter display name"))
        
        self.display_name_layout = QHBoxLayout()
        self.display_name_layout.addWidget(self.display_name_label)
        self.display_name_layout.addWidget(self.display_name_input)
        
        self.vBoxLayout.addLayout(self.display_name_layout)
        
        # Add index path card
        self.index_card = TextSettingCard(
            None,
            FIF.FOLDER,
            self.tr('Index Path'),
            self.tr('Path to the index file'),
            parent=self
        )
        self.index_card.clicked.connect(self.__onIndexCardClicked)
        
        self.vBoxLayout.addWidget(self.index_card)
        
        # Add model path card
        self.path_card = TextSettingCard(
            None,
            FIF.FOLDER,
            self.tr('Model Path'),
            self.tr('Path to the model file'),
            parent=self
        )
        self.path_card.clicked.connect(self.__onPathCardClicked)
        
        self.vBoxLayout.addWidget(self.path_card)
        
        # Add checkpoint path card
        self.ckpt_card = TextSettingCard(
            None,
            FIF.FOLDER,
            self.tr('Checkpoint Path'),
            self.tr('Path to the checkpoint file'),
            parent=self
        )
        self.ckpt_card.clicked.connect(self.__onCkptCardClicked)
        
        self.vBoxLayout.addWidget(self.ckpt_card)
        
        # Add buttons
        self.vBoxLayout.addLayout(self.hBoxLayout)
        
        # Set size
        self.widget.setFixedSize(500, 400)

    def enableYesButton(self):
        """Enable the Yes button if all fields are filled"""
        if (self.name_input.text() and 
            self.display_name_input.text() and 
            self.index_card.configItem.text() and 
            self.path_card.configItem.text() and 
            self.ckpt_card.configItem.text()):
            self.yesButton.setEnabled(True)
        else:
            self.yesButton.setEnabled(False)

    def __onIndexCardClicked(self):
        """Handle index card click to select index file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Index File"),
            "",
            self.tr("Index Files (*.index)")
        )
        if file_path:
            self.index_card.configItem.setText(file_path)
            self.enableYesButton()

    def __onPathCardClicked(self):
        """Handle path card click to select model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Model File"),
            "",
            self.tr("Model Files (*.pth)")
        )
        if file_path:
            self.path_card.configItem.setText(file_path)
            self.enableYesButton()

    def __onCkptCardClicked(self):
        """Handle checkpoint card click to select checkpoint file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Checkpoint File"),
            "",
            self.tr("Checkpoint Files (*.pth)")
        )
        if file_path:
            self.ckpt_card.configItem.setText(file_path)
            self.enableYesButton()