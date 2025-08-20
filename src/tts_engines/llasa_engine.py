import collections
import os
import typing

import librosa
import omegaconf
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from xcodec2.modeling_xcodec2 import XCodec2Model

from src.config.config import cfg
from src.enums.engine_type import EngineType
from src.tts_engines.tts_engine import tts_engine
from src.utils import torch_utils
from src.utils.filesystem_utils import get_app_root

torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])
torch.serialization.add_safe_globals([typing.Any])
torch.serialization.add_safe_globals([list])
torch.serialization.add_safe_globals([collections.defaultdict])
torch.serialization.add_safe_globals([dict])
torch.serialization.add_safe_globals([int])

class LlasaEngine(tts_engine):

    def __init__(self):
        super().__init__()
        self.engine_type = EngineType.LLASA
        self.engine_name = self.engine_type.value
        print(f"Setting Up {self.engine_name} Engine")
        self.device = cfg.get(cfg.device)
        self.codec_model = None
        self.tokenizer = None



    def load_model(self):
        print("Loading Llasa Model")
        self.codec_model = XCodec2Model.from_pretrained(os.path.abspath(os.path.join(get_app_root(), 'models', 'Llasa', 'xcodec2')))

        if self.is_base:
            self.tokenizer = AutoTokenizer.from_pretrained(str(os.path.abspath(os.path.join(get_app_root(), 'models', 'Llasa', str(cfg.get(cfg.llasa_mode))))), torch_dtype=torch_utils.get_compute_dtype())
            self.model = AutoModelForCausalLM.from_pretrained(str(os.path.abspath(os.path.join(get_app_root(), 'models', 'Llasa', str(cfg.get(cfg.llasa_mode))))), torch_dtype=torch_utils.get_compute_dtype())
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)

        if self.device == 'cpu':
            self.codec_model.eval().cpu()
            self.model.eval().cpu()
        else:
            self.codec_model.eval().cpu()
            self.model.eval().cuda()




    def unload_model(self):
        self.basic_unload_model()
        del self.tokenizer
        self.tokenizer = None

    def ids_to_speech_tokens(self, speech_ids):
        speech_tokens_str = []
        for speech_id in speech_ids:
            speech_tokens_str.append(f"<|s_{speech_id}|>")
        return speech_tokens_str

    def extract_speech_ids(self, speech_tokens_str):
        speech_ids = []
        for token_str in speech_tokens_str:
            if token_str.startswith('<|s_') and token_str.endswith('|>'):
                num_str = token_str[4:-2]

                num = int(num_str)
                speech_ids.append(num)
            else:
                print(f"Unexpected token: {token_str}")
        return speech_ids

    @torch.no_grad()
    def inference(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False, speaker=None):
        text = f"{speaker}: " + text if speaker else text
        prompt_wav = None
        speech_ids_prefix = None
        if voice:
            # only 16khz speech support!
            prompt_wav, sr = librosa.load(voice, sr=16000)
            prompt_wav = torch.from_numpy(prompt_wav).float().unsqueeze(0)
            input_text = transcript + text
            # Encode the prompt wav
            vq_code_prompt = self.codec_model.encode_code(input_waveform=prompt_wav)
            print("Prompt Vq Code Shape:", vq_code_prompt.shape)
            vq_code_prompt = vq_code_prompt[0, 0, :]
            # Convert int 12345 to token <|s_12345|>
            speech_ids_prefix = self.ids_to_speech_tokens(vq_code_prompt)
            formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

            # Tokenize the text and the speech prefix
            chat = [
                {"role": "user", "content": "Convert the text to speech:" + formatted_text},
                {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)}
            ]
        else:
            input_text = text
            formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"
            # Tokenize the text and the speech prefix
            chat = [
                {"role": "user", "content": "Convert the text to speech:" + formatted_text},
                {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}
            ]

        input_ids = self.tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            return_tensors='pt',
            continue_final_message=True
        )

        input_ids = input_ids.to(self.device)
        speech_end_id =  self.tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

        # Generate the speech autoregressively
        outputs =  self.model.generate(
            input_ids,
            max_length=2048,  # We trained our model with a max length of 2048
            eos_token_id=speech_end_id,
            do_sample=True,
            top_p=1,
            temperature=0.8,
        )
        if voice:
            # Extract the speech tokens
            generated_ids = outputs[0][input_ids.shape[1] - len(speech_ids_prefix):-1]

            speech_tokens = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Convert  token <|s_23456|> to int 23456
            speech_tokens = self.extract_speech_ids(speech_tokens)

            speech_tokens = torch.tensor(speech_tokens).cpu().unsqueeze(0).unsqueeze(0)

            # Decode the speech tokens to speech waveform
            gen_wav = self.codec_model.decode_code(speech_tokens)

            # if only need the generated part
            gen_wav = gen_wav[:,:,prompt_wav.shape[1]:]

        else:
            generated_ids = outputs[0][input_ids.shape[1]:-1]

            speech_tokens = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Convert  token <|s_23456|> to int 23456
            speech_tokens = self.extract_speech_ids(speech_tokens)

            speech_tokens = torch.tensor(speech_tokens).cpu().unsqueeze(0).unsqueeze(0)

            # Decode the speech tokens to speech waveform
            gen_wav = self.codec_model.decode_code(speech_tokens)

        return gen_wav[0, 0, :].cpu().numpy(), 16000
        # sf.write(output_file, , )
