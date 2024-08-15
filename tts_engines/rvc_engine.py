from tts_engine import tts_engine


class RVC_Engine(tts_engine):
    def __init__(self):
        super().__init__()
        self.engine_name = 'RVC'

    def load_model(self):
        pass

    def load_base_model(self):
        pass

    def clean(self):
        super().clean()

    def unload_model(self):
        pass
