"""
Widget module for FallTalk application.
This module contains all the widgets used in the application.
"""
from __future__ import annotations

from src.widgets.bulk_generation_widgets import (
    BulkLipFuzWidget, BulkGenerationRVCWidget,
    BulkGenerationTableWidget, BulkGenerationWidget
)
from src.widgets.characters_widget import CharactersWidget
from src.widgets.custom_message_box import CustomMessageBox
from src.widgets.drawer import RightDrawer
from src.widgets.ez_voice_creator_widget import EzVoiceCreatorWidget
from src.widgets.falltalk_fluent_window import FallTalkFluentWindow, CustomCommandBar
# Import all widgets to make them available from the module
from src.widgets.falltalk_widget import FallTalkWidget
from src.widgets.faq_widget import FaqWidget
from src.widgets.generation_widget import GenerationWidget
from src.widgets.references_widget import ReferencesWidget
from src.widgets.rvc_widgets import (
    BaseRVCWidget, RVCMicrophoneWidget, RVCFileWidget,
    RVCEdgeTTSWidget, RVCElevenLabsWidget, RVCWidget
)
from src.widgets.settings_widget import SettingsWidget
from src.widgets.table_models import (
    TableModel, CharacterTableModel,
    CustomTableModel, CustomReferencesModel
)
from src.widgets.upscale_widget import UpscaleWidget

__all__ = [
    'FallTalkWidget',
    'FallTalkFluentWindow',
    'CustomCommandBar',
    'RightDrawer',
    'FaqWidget',
    'SettingsWidget',
    'GenerationWidget',
    'BaseRVCWidget',
    'RVCMicrophoneWidget',
    'RVCFileWidget',
    'RVCEdgeTTSWidget',
    'RVCElevenLabsWidget',
    'RVCWidget',
    'TableModel',
    'CharacterTableModel',
    'CustomTableModel',
    'CustomReferencesModel',
    'CustomMessageBox',
    'CharactersWidget',
    'ReferencesWidget',
    'UpscaleWidget',
    'BulkLipFuzWidget',
    'BulkGenerationRVCWidget',
    'BulkGenerationTableWidget',
    'BulkGenerationWidget',
    'EzVoiceCreatorWidget',
]
