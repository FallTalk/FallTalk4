import os
import sys

from src.config.config import cfg
from src.enums.engine_type import EngineType
from src.tts_engines.tts_engine import tts_engine
from src.utils.audio_utils import load_audio
from src.utils.filesystem_utils import get_app_root, get_app_code_root

sys.path.append(os.path.abspath(os.path.join(get_app_code_root(), 'third_party', 'orpheus', 'orpheus_tts_pypi', 'orpheus_tts')))

from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from src.utils import torch_utils



class OrpheusEngine(tts_engine):

    def __init__(self):
        super().__init__()
        print("Setting Up Orpheus Engine")
        self.engine_type = EngineType.ORPHEUS
        self.engine_name = self.engine_type.value
        self.device = cfg.get(cfg.device)
        self.snac_model = None
        self.tokenizer = None
        self.model = None

    def load_model(self):
        self.snac_model = SNAC.from_pretrained( os.path.join(get_app_root(),'models', 'Orpheus', '3b-0.1', 'snac'))

        if self.is_base:
            self.model = AutoModelForCausalLM.from_pretrained(os.path.join(get_app_root(), 'models', 'Orpheus', '3b-0.1'), torch_dtype=torch_utils.get_compute_dtype())
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(get_app_root(), 'models', 'Orpheus', '3b-0.1'))
        else:
            self.model = AutoModelForCausalLM.from_pretrained(str(os.path.abspath(self.model_path)), torch_dtype=torch_utils.get_compute_dtype())
            self.tokenizer = AutoTokenizer.from_pretrained(str(os.path.abspath(self.model_path)))

        self.model.to(self.device)



    def unload_model(self):
        self.basic_unload_model()
        del self.snac_model
        del self.tokenizer
        self.tokenizer = None
        self.snac_model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def tokenise_audio(self, waveform):
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(dtype=torch.float32)

        waveform = waveform.unsqueeze(0)

        with torch.inference_mode():
            codes = self.snac_model.encode(waveform)

        all_codes = []
        for i in range(codes[0].shape[1]):
            all_codes.append(codes[0][0][i].item() + 128266)
            all_codes.append(codes[1][0][2 * i].item() + 128266 + 4096)
            all_codes.append(codes[2][0][4 * i].item() + 128266 + (2 * 4096))
            all_codes.append(codes[2][0][(4 * i) + 1].item() + 128266 + (3 * 4096))
            all_codes.append(codes[1][0][(2 * i) + 1].item() + 128266 + (4 * 4096))
            all_codes.append(codes[2][0][(4 * i) + 2].item() + 128266 + (5 * 4096))
            all_codes.append(codes[2][0][(4 * i) + 3].item() + 128266 + (6 * 4096))

        return all_codes

    def redistribute_codes(self, code_list):
        layer_1 = []
        layer_2 = []
        layer_3 = []
        for i in range((len(code_list) + 1) // 7):
            layer_1.append(code_list[7 * i])
            layer_2.append(code_list[7 * i + 1] - 4096)
            layer_3.append(code_list[7 * i + 2] - (2 * 4096))
            layer_3.append(code_list[7 * i + 3] - (3 * 4096))
            layer_2.append(code_list[7 * i + 4] - (4 * 4096))
            layer_3.append(code_list[7 * i + 5] - (5 * 4096))
            layer_3.append(code_list[7 * i + 6] - (6 * 4096))
        codes = [torch.tensor(layer_1).unsqueeze(0),
                 torch.tensor(layer_2).unsqueeze(0),
                 torch.tensor(layer_3).unsqueeze(0)]
        audio_hat = self.snac_model.decode(codes)
        return audio_hat


    def inference(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False, speaker=None, start_time=None, end_time=None):
        reference_mode = voice is not None and transcript is not None
        processed_prompts = [f"{speaker}: " + text if speaker else text]

        if reference_mode:

            # Reference audio tokens
            audio_tokens = self.tokenise_audio(load_audio(voice, 24000))


            # Tokenize the reference transcript
            prompt_tokked = self.tokenizer(transcript, return_tensors="pt")
            input_ids = prompt_tokked["input_ids"]

            # Build reference prefix tokens
            start_tokens = torch.tensor([[128259]], dtype=torch.int64)  # SOH
            end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)  # EOT, EOH, etc.
            final_tokens = torch.tensor([[128258, 128262]], dtype=torch.int64)  # EOS, etc.

            # Combine reference components
            reference_prefix = torch.cat([
                start_tokens,
                input_ids,
                end_tokens,
                torch.tensor([audio_tokens]),
                final_tokens
            ], dim=1)

        # Tokenize all prompts
        all_input_ids = []
        for prompt in processed_prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            if reference_mode:
                # For reference mode: [reference_prefix] + [prompt]
                full_input = torch.cat([
                    reference_prefix,
                    start_tokens,
                    input_ids,
                    end_tokens
                ], dim=1)
            else:
                # For normal mode: [prompt]
                start_token = torch.tensor([[128259]], dtype=torch.int64)  # SOH
                end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)  # EOT, EOH
                full_input = torch.cat([start_token, input_ids, end_tokens], dim=1)

            all_input_ids.append(full_input)

        # Pad all sequences to same length
        max_length = max(input_ids.shape[1] for input_ids in all_input_ids)
        all_padded_tensors = []
        all_attention_masks = []

        for input_ids in all_input_ids:
            padding = max_length - input_ids.shape[1]
            padded_tensor = torch.cat([
                torch.full((1, padding), 128263, dtype=torch.int64),  # Padding token
                input_ids
            ], dim=1)

            attention_mask = torch.cat([
                torch.zeros((1, padding), dtype=torch.int64),
                torch.ones((1, input_ids.shape[1]), dtype=torch.int64)
            ], dim=1)

            all_padded_tensors.append(padded_tensor)
            all_attention_masks.append(attention_mask)

        # Combine all batches
        all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
        all_attention_masks = torch.cat(all_attention_masks, dim=0)

        input_ids = all_padded_tensors.to(self.device)
        attention_mask = all_attention_masks.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=cfg.get(cfg.orpehus_max_new_tokens),
                do_sample=True,
                temperature=float(cfg.get(cfg.orpehus_temperature) / 100.0),
                # top_k=40,
                top_p=(cfg.get(cfg.fish_top_p) / 100.0),
                repetition_penalty=(cfg.get(cfg.orpehus_repetition) / 10.0),
                num_return_sequences=1,
                eos_token_id=128258,
            )

        # @title Convert output to speech
        token_to_find = 128257
        token_to_remove = 128258

        # Check if the token exists in the tensor
        token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

        if len(token_indices[1]) > 0:
            last_occurrence_idx = token_indices[1][-1].item()
            cropped_tensor = generated_ids[:, last_occurrence_idx + 1:]
        else:
            cropped_tensor = generated_ids

        mask = cropped_tensor != token_to_remove
        processed_rows = []
        for row in cropped_tensor:
            # Apply the mask to each row
            masked_row = row[row != token_to_remove]
            processed_rows.append(masked_row)

        code_lists = []
        for row in processed_rows:
            # row is a 1D tensor with its own length
            row_length = row.size(0)
            new_length = (row_length // 7) * 7  # largest multiple of 7 that fits in this row
            trimmed_row = row[:new_length]
            trimmed_row = [t - 128266 for t in trimmed_row]
            code_lists.append(trimmed_row)


        my_samples = []
        for code_list in code_lists:
            samples = self.redistribute_codes(code_list)
            my_samples.append(samples)

        # Combine all samples (assumes each is 1D or has shape [1, T])
        combined = torch.cat([s.detach().squeeze().cpu() for s in my_samples], dim=-1)

        # Convert to numpy and save
        return combined.numpy(), 24000
        # sf.write(output_file, combined.numpy(), samplerate=24000)
