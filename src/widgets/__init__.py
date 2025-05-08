"""
Widget module for FallTalk application.
This module contains all the widgets used in the application.
"""
from __future__ import annotations

# Import all widgets to make them available from the module
from src.widgets.falltalk_widget import FallTalkWidget
from src.widgets.falltalk_fluent_window import FallTalkFluentWindow, CustomCommandBar
from src.widgets.faq_widget import FaqWidget
from src.widgets.settings_widget import SettingsWidget
from src.widgets.generation_widget import GenerationWidget
from src.widgets.xtts_widget import XttsWidget
from src.widgets.styletts2_widget import StyleTTS2Widget
from src.widgets.f5_widget import F5Widget
from src.widgets.fish_widget import FishWidget
from src.widgets.orpheus_widget import OrpheusWidget
from src.widgets.llasa_widget import LlasaWidget
from src.widgets.dia_widget import DIAWidget
from src.widgets.gpt_sovits_widget import GPT_SoVITSWidget
from src.widgets.spark_widget import SparkWidget
from src.widgets.rvc_widgets import (
    BaseRVCWidget, RVCMicrophoneWidget, RVCFileWidget, 
    RVCEdgeTTSWidget, RVCElevenLabsWidget, RVCWidget
)
from src.widgets.table_models import (
    TableModel, CharacterTableModel, 
    CustomTableModel, CustomReferencesModel
)
from src.widgets.custom_message_box import CustomMessageBox
from src.widgets.characters_widget import CharactersWidget
from src.widgets.references_widget import ReferencesWidget
from src.widgets.upscale_widget import UpscaleWidget
from src.widgets.bulk_generation_widgets import (
    BulkLipFuzWidget, BulkGenerationRVCWidget, 
    BulkGenerationTableWidget, BulkGenerationWidget
)
from src.widgets.ez_voice_creator_widget import EzVoiceCreatorWidget

__all__ = [
    'FallTalkWidget',
    'FallTalkFluentWindow',
    'CustomCommandBar',
    'FaqWidget',
    'SettingsWidget',
    'GenerationWidget',
    'XttsWidget',
    'StyleTTS2Widget',
    'F5Widget',
    'FishWidget',
    'OrpheusWidget',
    'LlasaWidget',
    'DIAWidget',
    'GPT_SoVITSWidget',
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
    'EzVoiceCreatorWidget'
]