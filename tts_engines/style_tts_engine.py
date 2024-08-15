import soundfile as sf
from nltk.tokenize import word_tokenize
from phonemizer.backend import EspeakBackend

import falltalkutils
from falltalk.config import cfg
from tts_engine import tts_engine
from tts_engines.styletts2.Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from tts_engines.styletts2.Utils.PLBERT.util import load_plbert
from tts_engines.styletts2.models import *
from tts_engines.styletts2.text_utils import TextCleaner
from tts_engines.styletts2.utils import *


class StyleTTS2_Engine(tts_engine):
    def __init__(self):
        super().__init__()
        print("Setting Up StyleTTS2 Engine")
        self.engine_name = 'StyleTTS2'
        print(f"setting up TextCleaner")
        self.textclenaer = TextCleaner()
        print(f"setting up MelSpectrogram")
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        self.mean, self.std = -4, 4
        print(f"setting up EspeakBackend")
        self.global_phonemizer = EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)
        print(f"global_phonemizer { self.global_phonemizer}")
        self.sampler = None
        self.model_params = None
        self.text_aligner = None
        self.plbert = None
        self.pitch_extractor = None
        self.device = cfg.get(cfg.device)

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask

    def preprocess(self, wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def compute_style(self, path):
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = self.preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))
        del mel_tensor
        return torch.cat([ref_s, ref_p], dim=1)

    def clean(self):
        del self.sampler
        del self.to_mel
        del self.text_aligner
        del self.plbert
        del self.pitch_extractor

        self.unload_model()

    def unload_model(self):
        super().basic_unload_model()

    def load_base_model(self):
        self.load_model()

    def load_model(self):
        falltalkutils.logger.debug(f"Loading {self.model_path}")

        config = yaml.safe_load(open("models/StyleTTS2/Models/Vokan/config.yml"))

        # load pretrained ASR model
        self.text_aligner = load_ASR_models('models/StyleTTS2/ASR/epoch_00080.pth', 'models/StyleTTS2/ASR/config.yml')

        # load pretrained F0 model
        self.pitch_extractor = load_F0_models('models/StyleTTS2/JDC/bst.t7')

        # load BERT model
        self.plbert = load_plbert('models/StyleTTS2/PLBERT/')

        self.model_params = recursive_munch(config['model_params'])
        self.model = build_model(self.model_params, self.text_aligner, self.pitch_extractor, self.plbert)
        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(self.device) for key in self.model]

        if self.is_base:
            params_whole = torch.load("models/StyleTTS2/Models/Vokan/epoch_2nd_00012.pth", map_location='cpu')
        else:
            params_whole = torch.load(self.model_path, map_location='cpu')

        params = params_whole['net']

        for key in self.model:
            if key in params:
                print('%s loaded' % key)
                try:
                    self.model[key].load_state_dict(params[key])
                except:
                    from collections import OrderedDict
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    self.model[key].load_state_dict(new_state_dict, strict=False)
        #             except:
        #                 _load(params[key], model[key])
        _ = [self.model[key].eval() for key in self.model]

        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),  # empirical parameters
            clamp=False
        )

    def generate_audio(self, text=None, voice=None, language=None, output_file=None, streaming=False):
        alpha = cfg.get(cfg.style_alpha) / 100.0
        beta = cfg.get(cfg.style_beta) / 100.0
        diffusion_steps = cfg.get(cfg.style_diffusion_steps)
        embedding_scale = cfg.get(cfg.style_embedding_scale)
        self.inference(text=text, ref_s=self.compute_style(voice), output_file=output_file, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)
        rvc_enabled = cfg.get(cfg.rvc_enabled)
        if rvc_enabled and self.rvc_model:
            self.run_rvc(output_file)

    def inference(self, text, ref_s, output_file, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1):
        text = text.strip()
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        tokens = self.textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(self.device),
                                  embedding=bert_dur,
                                  embedding_scale=embedding_scale,
                                  features=ref_s,  # reference from the same speaker as the embedding
                                  num_steps=diffusion_steps).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(d_en,
                                                  s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))


        wav_tensor = out.squeeze().cpu().numpy()[..., :-100]  # weird pulse at the end of the model, need to be fixed later
        #wav_tensor = out.squeeze().cpu().numpy()[..., :-50]  # weird pulse at the end of the model, need to be fixed later
        sf.write(output_file, wav_tensor, 24000)
