import os
import os
import pickle
import re
import traceback
from time import time as ttime

import LangSegment
import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from falltalk import falltalkutils
from falltalk.config import cfg
from tts_engines.GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from tts_engines.GPT_SoVITS.feature_extractor import cnhubert
from tts_engines.GPT_SoVITS.module.mel_processing import spectrogram_torch
from tts_engines.GPT_SoVITS.module.models import SynthesizerTrn
from tts_engines.GPT_SoVITS.text import cleaned_text_to_sequence
from tts_engines.GPT_SoVITS.text.cleaner import clean_text
from falltalkutils import load_audio
from tts_engines.tts_engine import tts_engine


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


splits = {"，", "。", "？", "！", ",", "...", ".", "?", "!", "~", ":", "：", "—", "…", }
punctuation = {"!", "?", "…", ",", ".", "-"}

dict_language = {
    "all_zh": "all_zh",  # 全部按中文识别
    "en": "en",  # 全部按英文识别#######不变
    "all_ja": "all_ja",  # 全部按日文识别
    "zh": "zh",  # 按中英混合识别####不变
    "ja": "ja",  # 按日英混合识别####不变
    "auto": "auto",  # 多语种启动切分识别语种
}

dict_language_v2 = {
    "all_zh": "all_zh",#全部按中文识别
    "en": "en",#全部按英文识别#######不变
    "all_ja": "all_ja",#全部按日文识别
    "all_yue": "all_yue",#全部按中文识别
    "all_ko": "all_ko",#全部按韩文识别
    "zh": "zh",#按中英混合识别####不变
    "ja": "ja",#按日英混合识别####不变
    "yue": "yue",#按粤英混合识别####不变
    "ko": "ko",#按韩英混合识别####不变
    "auto": "auto",#多语种启动切分识别语种
    "auto_yue": "auto_yue",#多语种启动切分识别语种
}


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp, num):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), num))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut4(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip(".").split(".")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut5(inp):
    inp = inp.strip("\n")
    punds = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def process_text(texts):
    _text = []
    if all(text in [None, " ", "\n", ""] for text in texts):
        raise ValueError("Please enter valid text")
    for text in texts:
        if text in [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text


def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def replace_consecutive_punctuation(text):
    punctuations = ''.join(re.escape(p) for p in punctuation)
    pattern = f'([{punctuations}])([{punctuations}])+'
    result = re.sub(pattern, r'\1', text)
    return result


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec


def clean_text_inf(text, language, version):
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'utils':
            module = 'tts_engines.GPT_SoVITS.utils'  # Replace with the correct module path
        return super().find_class(module, name)


class GPT_SoVITS_Engine(tts_engine):
    def __init__(self):
        super().__init__()
        self.engine_name = 'GPT_SoVITS'
        self.device = cfg.get(cfg.device)
        self.t2s_model = None
        self.hz = None
        self.dict_s1 = None
        self.max_sec = None
        self.config = None
        self.bert_model = None
        self.tokenizer = None
        self.dict_s2 = None
        self.hps = None
        self.vq_model = None
        self.ssl_model = None
        self.version = None
        self.is_half = cfg.get(cfg.low_vram_gpt_sovits)
        self.dtype = torch.float16 if self.is_half else torch.float32

    def clean(self):
        super().clean()
        del self.bert_model
        del self.ssl_model
        del self.tokenizer
        del self.dict_s2
        del self.hps
        del self.dict_s1
        del self.hz
        self.unload_model()

    def unload_model(self):
        del self.vq_model
        del self.t2s_model
        self.vq_model = None
        self.t2s_model = None
        super().basic_unload_model()


    @torch.no_grad()
    def generate_audio(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False):
        self.inference(text=text, voice=voice, transcript=transcript, language=language, output_file=output_file, streaming=streaming)
        if cfg.get(cfg.rvc_enabled) and self.rvc_model:
            # del self.bert_model
            # del self.ssl_model
            # self.ssl_model = None
            # self.bert_model = None
            self.run_rvc(output_file)

    @torch.no_grad()
    def inference(self, text=None, transcript=None, voice=None, language='en', output_file=None, streaming=False):
        falltalkutils.logger.debug("Generating Audio...")
        # Synthesize audio

        self.loadSsl()
        self.loadBert()

        if self.is_base:
            last_sampling_rate, last_audio_data = self.get_tts_wav(prompt_text=transcript,
                                                                   prompt_language=language,
                                                                   text=text,
                                                                   ref_wav_path=voice,
                                                                   text_language=language,
                                                                   top_p=(cfg.get(cfg.top_p_gpt_sovits) / 100.0),
                                                                   top_k=cfg.get(cfg.top_k_gpt_sovits),
                                                                   how_to_cut=cfg.get(cfg.slice_mode),
                                                                   temperature=(cfg.get(cfg.temperature_gpt_sovits) / 100.0),
                                                                   speed=(cfg.get(cfg.speed_gpt_sovits) / 100.0))
        else:
            last_sampling_rate, last_audio_data = self.get_tts_wav(inp_refs=voice,
                                                                   prompt_text=transcript,
                                                                   prompt_language=language,
                                                                   text=text,
                                                                   ref_wav_path=None,
                                                                   ref_free=True,
                                                                   text_language=language,
                                                                   top_p=(cfg.get(cfg.top_p_gpt_sovits) / 100.0),
                                                                   top_k=cfg.get(cfg.top_k_gpt_sovits),
                                                                   how_to_cut=cfg.get(cfg.slice_mode),
                                                                   temperature=(cfg.get(cfg.temperature_gpt_sovits) / 100.0),
                                                                   speed=(cfg.get(cfg.speed_gpt_sovits) / 100.0))

        sf.write(output_file, last_audio_data, last_sampling_rate)
        print(f"Audio saved to {output_file}")

    def get_tts_wav(self, ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut="No Slice", top_k=20, top_p=0.6, temperature=0.6, ref_free=False, speed=1.0, inp_refs=None):
        if inp_refs is None:
            inp_refs = []
        if prompt_text is None or len(prompt_text) == 0:
            ref_free = True
        t0 = ttime()
        prompt_language = dict_language_v2[prompt_language]
        text_language = dict_language_v2[text_language]
        if not ref_free:
            prompt_text = prompt_text.strip("\n")
            if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
            print("input reference text:", prompt_text)
        text = text.strip("\n")
        text = replace_consecutive_punctuation(text)
        #if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text

        print("Actual input target text:", text)
        zero_wav = np.zeros(
            int(self.hps.data.sampling_rate * 0.3),
            dtype=np.float16 if self.is_half else np.float32,
        )
        if not ref_free:
            with torch.no_grad():
                wav16k, sr = librosa.load(ref_wav_path, sr=16000)
                wav16k = torch.from_numpy(wav16k)
                zero_wav_torch = torch.from_numpy(zero_wav)
                if self.is_half:
                    wav16k = wav16k.half().to(self.device)
                    zero_wav_torch = zero_wav_torch.half().to(self.device)
                else:
                    wav16k = wav16k.to(self.device)
                    zero_wav_torch = zero_wav_torch.to(self.device)
                wav16k = torch.cat([wav16k, zero_wav_torch])
                ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))[
                    "last_hidden_state"
                ].transpose(
                    1, 2
                )  # .float()
                codes = self.vq_model.extract_latent(ssl_content)
                prompt_semantic = codes[0, 0]
                prompt = prompt_semantic.unsqueeze(0).to(self.device)

        t1 = ttime()

        if (how_to_cut == "Slice once every 2 sentences"):
            text = cut1(text, 2)
        if (how_to_cut == "Slice once every 4 sentences"):
            text = cut1(text, 4)
        elif (how_to_cut == "Cut per 50 characters"):
            text = cut2(text)
        elif (how_to_cut == "Segment by Chinese period (。)."):
            text = cut3(text)
        elif (how_to_cut == "Slice by English punct"):
            text = cut4(text)
        elif (how_to_cut == "Slice by every punct"):
            text = cut5(text)
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        print("Actual input target text (after sentence segmentation):", text)
        texts = text.split("\n")
        texts = process_text(texts)
        texts = merge_short_text_in_array(texts, 5)
        audio_opt = []
        if not ref_free:
            phones1, bert1, norm_text1 = self.get_phones_and_bert(prompt_text, prompt_language, self.version)

        for text in texts:
            # 解决输入目标文本的空行导致报错的问题
            if (len(text.strip()) == 0):
                continue
            if (text[-1] not in splits): text += "。" if text_language != "en" else "."
            print("Actual input target text (each sentence):", text)
            phones2, bert2, norm_text2 = self.get_phones_and_bert(text, text_language, self.version)
            print("Text processed by the frontend (each sentence):", norm_text2)
            if not ref_free:
                bert = torch.cat([bert1, bert2], 1)
                all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
            else:
                bert = bert2
                all_phoneme_ids = torch.LongTensor(phones2).to(self.device).unsqueeze(0)

            bert = bert.to(self.device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)

            t2 = ttime()
            with torch.no_grad():
                # pred_semantic = t2s_model.model.infer(
                pred_semantic, idx = self.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    None if ref_free else prompt,
                    bert,
                    # prompt_phone_len=ph_offset,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=self.hz * self.max_sec,
                )
            t3 = ttime()
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
            refers = []
            if inp_refs and len(inp_refs) > 0:
                for path in inp_refs:
                    try:
                        refer = get_spepc(self.hps, path).to(self.dtype).to(self.device)
                        refers.append(refer)
                    except:
                        traceback.print_exc()

            if len(refers) == 0 and ref_wav_path:
                refers = [get_spepc(self.hps, ref_wav_path).to(self.dtype).to(self.device)]

            audio = (self.vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(self.device).unsqueeze(0), refers, speed=speed).detach().cpu().numpy()[0, 0])
            max_audio = np.abs(audio).max()
            if max_audio > 1: audio /= max_audio
            audio_opt.append(audio)
            audio_opt.append(zero_wav)
            t4 = ttime()
        print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        return self.hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
            np.int16
        )

    def change_gpt_weights(self, gpt_path):
        falltalkutils.logger.debug(f"change_gpt_weights {gpt_path}")
        self.hz = 50
        self.dict_s1 = torch.load(gpt_path, map_location="cpu")
        self.config = self.dict_s1["config"]
        self.max_sec = self.config["data"]["max_sec"]
        self.t2s_model = Text2SemanticLightningModule(self.config, "****", is_train=False)
        self.t2s_model.load_state_dict(self.dict_s1["weight"])
        if self.is_half:
            self.t2s_model = self.t2s_model.half()
        self.t2s_model = self.t2s_model.to(self.device)
        self.t2s_model.eval()
        total = sum([param.nelement() for param in self.t2s_model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
        with open("./gweight.txt", "w", encoding="utf-8") as f: f.write(gpt_path)

    def change_sovits_weights(self, sovits_path):
        falltalkutils.logger.debug(f"change_sovits_weights {sovits_path}")
        self.dict_s2 = torch.load(sovits_path, map_location="cpu")
        self.hps = self.dict_s2["config"]
        self.hps = DictToAttrRecursive(self.hps)
        self.hps.model.semantic_frame_rate = "25hz"
        if self.dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
            self.hps.model.version = "v1"
        else:
            self.hps.model.version = "v2"
        self.version = self.hps.model.version
        self.vq_model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model
        )
        if not self.is_base:
            del self.vq_model.enc_q
        if self.is_half:
            self.vq_model = self.vq_model.half().to(self.device)
        else:
            self.vq_model = self.vq_model.to(self.device)
        self.vq_model.eval()
        print(self.vq_model.load_state_dict(self.dict_s2["weight"], strict=False))

    def get_bert_feature(self, text, word2ph):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T

    def get_phones_and_bert(self, text, language, version, final=False):
        bert = None
        phones = None
        norm_text = None

        if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
            language = language.replace("all_", "")
            if language == "en":
                LangSegment.setfilters(["en"])
                formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
            else:
                # 因无法区别中日文汉字,以用户输入为准
                formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            if language == "zh":
                pass
                if re.search(r'[A-Za-z]', formattext):
                    pass
                    # formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                    # formattext = chinese.mix_text_normalize(formattext)
                    # return self.get_phones_and_bert(formattext, "zh", version)
                else:
                    phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
                    bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
            elif language == "yue" and re.search(r'[A-Za-z]', formattext):
                pass
                # formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                # formattext = chinese.mix_text_normalize(formattext)
                # return self.get_phones_and_bert(formattext, "yue", version)
            else:
                phones, word2ph, norm_text = clean_text_inf(formattext, language, self.version)
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=torch.float16 if self.is_half else torch.float32,
                ).to(self.device)
        elif language in {"zh", "ja", "auto"}:
            textlist = []
            langlist = []
            LangSegment.setfilters(["zh", "ja", "en", "ko"])
            if language == "auto":
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "ko":
                        langlist.append("zh")
                        textlist.append(tmp["text"])
                    else:
                        langlist.append(tmp["lang"])
                        textlist.append(tmp["text"])
            else:
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        # 因无法区别中日文汉字,以用户输入为准
                        langlist.append(language)
                    textlist.append(tmp["text"])
            print(textlist)
            print(langlist)
            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, self.version)
                bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = ''.join(norm_text_list)

        if not final and len(phones) < 6:
            return self.get_phones_and_bert("." + text, language, version, final=True)

        return phones, bert.to(self.dtype), norm_text

    def get_bert_inf(self, phones, word2ph, norm_text, language):
        language = language.replace("all_", "")
        if language == "zh":
            bert = self.get_bert_feature(norm_text, word2ph).to(self.device)  # .to(dtype)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if self.is_half else torch.float32,
            ).to(self.device)

        return bert

    def load_base_model(self):
        self.load_model()


    def loadSsl(self):
        if not self.ssl_model:
            cnhubert.cnhubert_base_path = "models/GPT_SoVITS/chinese-hubert-base"
            self.ssl_model = cnhubert.get_model()
            if self.is_half:
                self.ssl_model = self.ssl_model.half().to(self.device)
            else:
                self.ssl_model = self.ssl_model.to(self.device)

    def loadBert(self):
        if not self.bert_model:
            bert_path = "models/GPT_SoVITS/chinese-roberta-wwm-ext-large"
            self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
            self.bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
            if self.is_half:
                self.bert_model = self.bert_model.half().to(self.device)
            else:
                self.bert_model = self.bert_model.to(self.device)

    def load_model(self):
        if not self.is_base:
            self.change_sovits_weights(os.path.abspath(self.model_path))
            self.change_gpt_weights(os.path.abspath(self.get_model(self.engine_name, "ckpt")))
        else:
            self.change_sovits_weights(os.path.abspath("models/GPT_SoVITS/v2/s2G2333k.pth"))
            self.change_gpt_weights(os.path.abspath("models/GPT_SoVITS/v2/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"))
