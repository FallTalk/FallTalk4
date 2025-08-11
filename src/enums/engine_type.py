import os
from enum import Enum

class EngineType(Enum):
    F5 = ("F5", True, True, ["v1"], "v1", 15, 3, False, True)
    RVC = ("RVC", False, False, ["1", "2"], "2", 15, 3, False, True)
    ORPHEUS = ("Orpheus", False, True, ["3b-0.1"], "3b-0.1", 15, 3, True, True)
    DIA = ("DIA", True, True, ["3b-0.1"], "3b-0.1", 15, 3, False, False)
    FISH_SPEECH = ("FishSpeech", True, True, ["s1-mini"], "s1-mini", 30, 5, False, True)
    LLASA = ("Llasa", False, True, ["1b", "3b"], "1b", 15, 3, True, False)
    XTTS_V2 = ("XTTS v2", True, False, ["1"], "2", 10, 3, False, True)
    GPT_SOVITS = ("GPT SoVITS", True, True, ["1", "2", "v2ProPlus"], "v2ProPlus", 10, 3, False, True)
    STYLE_TTS2 = ("Style TTS2", True, False, ["1"], "1", 15, 3, False, False)
    SPARK = ("Spark", False, False, ["0.5"], "0.5", 15, 3, True, True)
    MegaTTS3 = ("Mega TTS3", True, True, ["3"], "3", 15, 3, True, False)
    CSM = ("CSM", False, True, ["1b"], "1b", 15, 3, True, True)
    CHATTERBOX = ("Chatterbox", False, True, ["0.5B"], "0.5B", 30, 10, True, True)
    INDEX = ("Index", False, True, ["v1.5", "v2"], "v1.5", 15, 3, True, False)
    HIGGS = ("Higgs", True, True, ["v2"], "v2", 15, 3, True, False)
    DMOSPEECH2 = ("DMO Speech 2", True, True, ["v2"], "v2", 15, 3, True, True)

    def __new__(cls, value, needs_reference_when_trained, needs_transcription, supported_versions, version, max_reference_length, min_reference_length, loads_from_dir, enabled):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.needs_reference_when_trained = needs_reference_when_trained
        obj.needs_transcription = needs_transcription
        obj.supported_versions = supported_versions
        obj.version = version
        obj.max_reference_length = max_reference_length
        obj.min_reference_length = min_reference_length
        obj.loads_from_dir = loads_from_dir
        obj.enabled = enabled
        return obj

    @classmethod
    def _missing_(cls, value):
        # Allow direct string comparison
        for member in cls:
            if member.value == value:
                return member
        return None

    @property
    def value(self):
        return self._value_

    def is_version_supported(self, version):
        """Check if a given version is supported by this engine type"""
        return version in self.supported_versions

    def get_shared_model_path(self, shared_model, version):
        """Get the correct model path based on a version"""
        base_path = os.path.join("shared", f"{shared_model}", f"{self.value}")
        if version == "1":
            return base_path
        return os.path.join(base_path, f"{version}")

    def get_model_path(self, character_name, version):
        """Get the correct model path based on a version"""
        base_path = os.path.join(character_name, self.value)
        if version == "1":
            return base_path
        return os.path.join(str(base_path), version)
