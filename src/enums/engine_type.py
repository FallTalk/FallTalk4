from enum import Enum

class EngineType(Enum):
    F5 = ("F5", True, True)  # (name, needs_reference_when_trained, needs_transcription)
    RVC = ("RVC", False, False)
    ORPHEUS = ("Orpheus", True, True)
    DIA = ("DIA", True, True)
    FISH_SPEECH = ("FishSpeech", True, True)
    LLASA = ("Llasa", True, True)
    XTTS_V2 = ("XTTSv2", True, False)
    GPT_SOVITS = ("GPT_SoVITS", True, True)
    STYLE_TTS2 = ("StyleTTS2", True, False)

    def __new__(cls, value, needs_reference_when_trained, needs_transcription):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.needs_reference_when_trained = needs_reference_when_trained
        obj.needs_transcription = needs_transcription
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