from enum import Enum

class EngineType(Enum):
    F5 = ("F5", True, True, ["v1"], "v1", 15, 3, False)
    RVC = ("RVC", False, False, ["1", "2"], "2", 15, 3, False)
    ORPHEUS = ("Orpheus", False, True, ["3b"], "3b", 15, 3, True)
    DIA = ("DIA", True, True, ["0.1"], "0.1", 15, 3, False)
    FISH_SPEECH = ("FishSpeech", True, True, ["o1"], "o1", 30, 10, False)
    LLASA = ("Llasa", False, True, ["1b", "3b"], "1b", 15, 3, True)
    XTTS_V2 = ("XTTSv2", True, False, ["1"], "2", 10, 3, False)
    GPT_SOVITS = ("GPT_SoVITS", False, True, ["1", "2", "v2ProPlus"], "v2ProPlus", 10, 3, False)
    STYLE_TTS2 = ("StyleTTS2", True, False, ["1"], "1", 15, 3, False)
    SPARK = ("Spark", False, False, ["0.5"], "0.5", 15, 3, True)
    MegaTTS3 = ("MegaTTS3", True, True, ["3"], "3", 15, 3, True)
    CSM = ("CSM", False, True, ["1b"], "1b", 15, 3, True)
    CHATTERBOX = ("Chatterbox", False, True, ["v1"], "v1", 15, 3, True)

    def __new__(cls, value, needs_reference_when_trained, needs_transcription, supported_versions, version, max_reference_length, min_reference_length, loads_from_dir):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.needs_reference_when_trained = needs_reference_when_trained
        obj.needs_transcription = needs_transcription
        obj.supported_versions = supported_versions
        obj.version = version
        obj.max_reference_length = max_reference_length
        obj.min_reference_length = min_reference_length
        obj.loads_from_dir = loads_from_dir
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