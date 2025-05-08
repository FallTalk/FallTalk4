from enum import Enum

class EngineType(Enum):
    F5 = ("F5", True, True, ["v1"], "v1", 15, 3)
    RVC = ("RVC", False, False, ["1", "2"], "2", 15, 3)
    ORPHEUS = ("Orpheus", True, True, ["1"], "1", 15, 3)
    DIA = ("DIA", True, True, ["1"], "1", 15, 3)
    FISH_SPEECH = ("FishSpeech", True, True, ["1.5"], "1.5", 30, 10)
    LLASA = ("Llasa", True, True, ["1"], "1", 15, 3)
    XTTS_V2 = ("XTTSv2", True, False, ["1"], "2", 10, 3)
    GPT_SOVITS = ("GPT_SoVITS", True, True, ["1", "2", "3", "4"], "4", 10, 3)
    STYLE_TTS2 = ("StyleTTS2", True, False, ["1"], "1", 15, 3)
    SPARK = ("Spark", True, False, ["1"], "1", 15, 3)
    MegaTTS3 = ("MegaTTS3", True, True, ["3"], "3", 15, 3)

    def __new__(cls, value, needs_reference_when_trained, needs_transcription, supported_versions, version, max_reference_length, min_reference_length):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.needs_reference_when_trained = needs_reference_when_trained
        obj.needs_transcription = needs_transcription
        obj.supported_versions = supported_versions
        obj.version = version
        obj.max_reference_length = max_reference_length
        obj.min_reference_length = min_reference_length
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

    def get_model_path(self, character_name, version):
        """Get the correct model path based on version"""
        base_path = f"{character_name}/{self.value}"
        if version == "1":
            return base_path
        return f"{base_path}/{version}"