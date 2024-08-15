import io
import os
import random

import numpy as np
import torch
import torchaudio

from tts_engines import whisper_engine
from tts_engines.voicecraft import voicecraft
from tts_engines.voicecraft.data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)

from falltalk import falltalkutils

from falltalk.config import cfg
from tts_engines.tts_engine import tts_engine


def seed_everything(seed):
    if seed != -1:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class VoiceCraft_Engine(tts_engine):
    def __init__(self):
        super().__init__()
        self.phn2num = None
        self.text_tokenizer = TextTokenizer(backend="es2peak")
        self.audio_tokenizer = None
        self.ckpt = None
        self.voicecraft_model = None
        self.engine_name = 'VoiceCraft'
        self.device = cfg.get(cfg.device)

    def combine_weights(self, ckpt):
        lora_weights = torch.load(self.model_path, map_location="cpu")
        for name in lora_weights:
            falltalkutils.logger.debug(f" lora name in weights {name}")
        # Create a new state dictionary for the model with the LoRA weights applied
        new_state_dict = {}
        for name, param in ckpt["model"].items():
            # falltalkutils.logger.debug(f" name found in weights {name}")
            if name.endswith('.weight'):
                # falltalkutils.logger.debug(f"checking for lora weights module.{name.replace('.weight', '.lora.A')}")
                lora_A_name = f"module.{name.replace('.weight', '.lora.A')}"
                lora_B_name = f"module.{name.replace('.weight', '.lora.B')}"
                if lora_A_name in lora_weights and lora_B_name in lora_weights:
                    # Apply the LoRA weights to the corresponding parameters in the model
                    A = lora_weights[lora_A_name]
                    B = lora_weights[lora_B_name]
                    if A is not None and B is not None:
                        falltalkutils.logger.debug(f"LoRA weights A.shape={A.shape}, B.shape={B.shape}, param.shape={param.shape} added to {name}")
                        if A.shape[0] == param.shape[1] and B.shape[1] == param.shape[0]:
                            lora_weight = (A @ B).T
                            ckpt["model"][name] = param + lora_weight
                        else:
                            raise ValueError(f"Shape mismatch for {name}: A.shape={A.shape}, B.shape={B.shape}, param.shape={param.shape}")

        falltalkutils.logger.debug("applied lora weights to model")
        del lora_weights

    def clean(self):
        super().clean()
        del self.audio_tokenizer
        del self.text_tokenizer
        self.unload_model()

    def unload_model(self):
        del self.voicecraft_model
        del self.ckpt
        super().basic_unload_model()
        self.ckpt = None
        self.voicecraft_model = None

    def load_base_model(self):
        self.load_model()

    def load_model(self):
        voicecraft_name = "830M_TTSEnhanced.pth"
        ckpt_fn = os.path.abspath(f"models/VoiceCraft/{voicecraft_name}")
        encodec_fn = os.path.abspath("models/VoiceCraft/encodec_4cb2048_giga.th")

        self.ckpt = torch.load(ckpt_fn, map_location="cpu")
        self.voicecraft_model = voicecraft.VoiceCraft(self.ckpt["config"])

        if not self.is_base:
            self.combine_weights(self.ckpt)

        self.voicecraft_model.load_state_dict(self.ckpt["model"])

        if False:
            import deepspeed

            self.ds_engine = deepspeed.init_inference(
                model=self.voicecraft_model.half(),  # Transformers models
                mp_size=1,  # Number of GPU
                dtype=torch.float32,  # desired data type of output
                replace_method="auto",  # Lets DS autmatically identify the layer to replace
                replace_with_kernel_inject=True,  # replace the model with the kernel injector
            )
            self.ds_engine.module.eval()
        else:
            self.voicecraft_model.to(self.device)
            self.voicecraft_model.eval()

        self.phn2num = self.ckpt['phn2num']
        if not self.audio_tokenizer:
            self.audio_tokenizer = AudioTokenizer(signature=encodec_fn, device=self.device)

    @torch.no_grad()
    def generate_audio(self, text=None, voice=None, language=None, output_file=None, streaming=None, transcribe_state=None, prompt_end_time=None, edit_start_time=None, edit_end_time=None):
        self.inference(text=text, voice=voice, language="en", output_file=output_file, transcribe_state=transcribe_state, prompt_end_time=prompt_end_time, edit_start_time=edit_start_time, edit_end_time=edit_end_time)
        if cfg.get(cfg.rvc_enabled) and self.rvc_model:
            self.run_rvc(output_file)

    @torch.no_grad()
    def inference(self, text=None, voice=None, language=None, output_file=None, streaming=None, transcribe_state=None, prompt_end_time=None, edit_start_time=None, edit_end_time=None):
        seed_everything(cfg.get(cfg.seed))
        transcript = whisper_engine.replace_numbers_with_words(text).replace("  ", " ").replace("  ", " ")  # replace numbers with words, so that the phonemizer can do a better job

        if cfg.get(cfg.mode) == "Long TTS":
            if cfg.get(cfg.split_text) == "Newline":
                sentences = transcript.split('\n')
            else:
                from nltk.tokenize import sent_tokenize
                sentences = sent_tokenize(transcript.replace("\n", " "))
        else:
            sentences = [transcript.replace("\n", " ")]

        info = torchaudio.info(voice)
        audio_dur = info.num_frames / info.sample_rate

        codec_sr = cfg.get(cfg.codec_sr) * 1.0
        left_margin = cfg.get(cfg.left_margin) / 1000.0
        right_margin = cfg.get(cfg.right_margin) / 1000.0

        audio_tensors = []
        inference_transcript = ""
        for sentence in sentences:
            decode_config = {"top_k": cfg.get(cfg.top_k), "top_p": cfg.get(cfg.top_p) / 100.0, "temperature": cfg.get(cfg.voicecraft_temperature) / 100.0, "stop_repetition": cfg.get(cfg.stop_repetition),
                             "kvcache": cfg.get(cfg.kvcache), "codec_audio_sr": cfg.get(cfg.codec_audio_sr), "codec_sr": codec_sr,
                             "silence_tokens": cfg.get(cfg.silence_tokens), "sample_batch_size": cfg.get(cfg.sample_batch_size)}

            falltalkutils.logger.debug(f"{decode_config}")
            if cfg.get(cfg.mode) != "edit":
                falltalkutils.logger.debug(f"Starting TTS")
                from tts_engines.voicecraft.inference_tts_scale import inference_one_sample

                if cfg.get(cfg.smart_transcript):
                    target_transcript = ""
                    for word in transcribe_state["words_info"]:
                        if word["end"] < prompt_end_time:
                            falltalkutils.logger.debug(f'word: -{word["word"]}-')
                            append = word["word"] + (" " if word["word"][-1] != " " else "")
                            falltalkutils.logger.debug(f'append: -{append}-')
                            target_transcript += word["word"] + (" " if word["word"][-1] != " " else "")
                        elif (word["start"] + word["end"]) / 2 < prompt_end_time:
                            # include part of the word it it's big, but adjust prompt_end_time
                            target_transcript += word["word"] + (" " if word["word"][-1] != " " else "")
                            prompt_end_time = word["end"]
                            break
                        else:
                            break
                    target_transcript += f" {sentence}"
                else:
                    target_transcript = sentence

                # target_transcript = re.sub(r'\s+', ' ', target_transcript)

                falltalkutils.logger.debug(f"target_transcript {target_transcript}")

                inference_transcript += target_transcript + "\n"

                prompt_end_frame = int(min(audio_dur, prompt_end_time) * info.sample_rate)

                _, gen_audio = inference_one_sample(self.voicecraft_model,
                                                    self.ckpt["config"],
                                                    self.phn2num,
                                                    self.text_tokenizer, self.audio_tokenizer,
                                                    voice, target_transcript, self.device, decode_config,
                                                    prompt_end_frame)
            else:
                falltalkutils.logger.debug(f"Starting Edit")
                from tts_engines.voicecraft.inference_speech_editing_scale import inference_one_sample

                if cfg.get(cfg.smart_transcript):
                    target_transcript = ""
                    for word in transcribe_state["words_info"]:
                        if word["start"] < edit_start_time:
                            target_transcript += word["word"] + (" " if word["word"][-1] != " " else "")
                        else:
                            break
                    target_transcript += f" {sentence}"
                    for word in transcribe_state["words_info"]:
                        if word["end"] > edit_end_time:
                            target_transcript += (" " if word["word"][-1] != " " else "") + word["word"]
                else:
                    target_transcript = sentence

                # target_transcript = re.sub(r'\s+', ' ', target_transcript)

                falltalkutils.logger.debug(f"target_transcript {target_transcript}")

                inference_transcript += target_transcript + "\n"

                morphed_span = (max(edit_start_time - left_margin, 1 / codec_sr), min(edit_end_time + right_margin, audio_dur))
                mask_interval = [[round(morphed_span[0] * codec_sr), round(morphed_span[1] * codec_sr)]]
                mask_interval = torch.LongTensor(mask_interval)

                falltalkutils.logger.debug(f"mask_interval {mask_interval}")

                _, gen_audio = inference_one_sample(self.voicecraft_model,
                                                    self.ckpt["config"],
                                                    self.phn2num,
                                                    self.text_tokenizer, self.audio_tokenizer,
                                                    voice, target_transcript, mask_interval, self.device, decode_config)
            gen_audio = gen_audio[0].cpu()
            audio_tensors.append(gen_audio)

        self.get_output_audio(audio_tensors, cfg.get(cfg.codec_audio_sr), output_file)

    def get_output_audio(self, audio_tensors, codec_audio_sr, output_file):
        result = torch.cat(audio_tensors, 1)
        buffer = io.BytesIO()
        torchaudio.save(buffer, result, int(codec_audio_sr), format="wav")
        buffer.seek(0)
        with open(output_file, 'wb') as f:
            f.write(buffer.getbuffer())

        buffer.close()
