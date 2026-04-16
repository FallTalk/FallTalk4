# coding: utf-8
import configparser
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.utils.filesystem_utils import get_app_root


def find_fallout4_exe():
    drives = [d + ':\\' for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
    common_paths = [
        '\\Program Files (x86)\\Steam\\steamapps\\common\\Fallout 4\\',
        '\\Program Files\\Steam\\steamapps\\common\\Fallout 4\\',
        '\\Program Files (x86)\\Fallout 4\\',
        '\\Program Files\\Fallout 4\\',
        '\\Steam\\steamapps\\common\\Fallout 4\\',
        '\\Steam Games\\steamapps\\common\\Fallout 4\\',
    ]
    for drive in drives:
        if not os.path.exists(drive):
            continue
        for common_path in common_paths:
            exe = os.path.join(drive, common_path.lstrip('\\'), 'Fallout4.exe')
            if os.path.isfile(exe):
                return os.path.dirname(exe)
    return "fallout4.exe not found"


class ConfigKey:
    def __init__(self, key: str, default: Any, type_=None):
        self.key = key
        self.default = default
        self.type_ = type_


class Config:
    # --- General ---
    engine = ConfigKey("engine", "F5", str)
    load_engine_art_start = ConfigKey("load_engine_art_start", False, bool)
    auto_update_models = ConfigKey("auto_update_models", False, bool)
    device = ConfigKey("device", "cuda", str)
    seed = ConfigKey("seed", -1, int)
    check_for_updates = ConfigKey("check_for_updates", True, bool)
    download_configs = ConfigKey("download_configs", True, bool)
    auto_play = ConfigKey("auto_play", True, bool)
    disableSSLVerify = ConfigKey("disableSSLVerify", False, bool)
    api_only_mode = ConfigKey("api_only_mode", False, bool)
    accepted_disclaimer = ConfigKey("accepted_disclaimer", False, bool)
    accepts_custom_disclaimer = ConfigKey("accepts_custom_disclaimer", False, bool)
    first_start = ConfigKey("first_start", True, bool)

    # --- Paths ---
    fallout_4_directory = ConfigKey("fallout_4_directory", "fallout4.exe not found", str)
    fallout_4_directory_check = ConfigKey("fallout_4_directory_check", True, bool)
    custom_references = ConfigKey("custom_references", os.path.join(get_app_root(), "references"), str)
    output_dir = ConfigKey("output_dir", "output", str)
    huggingface_cache_dir = ConfigKey("huggingface_cache_dir", None)
    huggingface_key = ConfigKey("huggingface_key", "", str)

    # --- Audio ---
    audio_output_device = ConfigKey("audio_output_device", -1, int)
    audio_input_device = ConfigKey("audio_input_device", -1, int)

    # --- Text processing ---
    max_text_size = ConfigKey("max_text_size", 200, int)
    min_chunk_size = ConfigKey("min_chunk_size", 50, int)
    lowercase_conversion = ConfigKey("lowercase_conversion", False, bool)
    whitespace_normalization = ConfigKey("whitespace_normalization", True, bool)
    dot_letter_fix = ConfigKey("dot_letter_fix", True, bool)
    inline_reference_removal = ConfigKey("inline_reference_removal", True, bool)
    pad_short_phrases = ConfigKey("pad_short_phrases", True, bool)

    # --- Bulk ---
    replace_existing = ConfigKey("replace_existing", False, bool)
    include_subdir = ConfigKey("include_subdir", True, bool)
    threads = ConfigKey("threads", 1, int)
    multigen_total = ConfigKey("multigen_total", 3, int)
    ez_total = ConfigKey("ez_total", 1, int)

    # --- Features ---
    rvc_enabled = ConfigKey("rvc_enabled", False, bool)
    apbwe_enabled = ConfigKey("apbwe_enabled", True, bool)
    xwm_enabled = ConfigKey("xwm_enabled", False, bool)
    keep_only_fuz = ConfigKey("keep_only_fuz", False, bool)
    use_existing_lip = ConfigKey("use_existing_lip", True, bool)

    # --- UI (imgui) ---
    font_global_scale = ConfigKey("font_global_scale", 1.0, float)
    theme_color = ConfigKey("theme_color", "#ffb642", str)
    nav_collapsed = ConfigKey("nav_collapsed", False, bool)

    # --- RVC ---
    rvc_pitch = ConfigKey("rvc_pitch", 0, int)
    rvc_hop_length = ConfigKey("rvc_hop_length", 128, int)
    rvc_training_data_size = ConfigKey("rvc_training_data_size", 0, int)
    rvc_index_influence = ConfigKey("rvc_index_influence", 75, int)
    rvc_volume_envelope = ConfigKey("rvc_volume_envelope", 100, int)
    rvc_protect = ConfigKey("rvc_protect", 33, int)
    rvc_filter_radius = ConfigKey("rvc_filter_radius", 3, int)
    rvc_autotune = ConfigKey("rvc_autotune", False, bool)
    rvc_split_audio = ConfigKey("rvc_split_audio", False, bool)
    rvc_pitch_extraction = ConfigKey("rvc_pitch_extraction", "rmvpe", str)
    rvc_mode = ConfigKey("rvc_mode", "Microphone", str)
    rvc_eleven_labs_key = ConfigKey("rvc_eleven_labs_key", "", str)
    rvc_embedder_model = ConfigKey("rvc_embedder_model", "contentvec", str)

    # --- F5 ---
    f5_mode = ConfigKey("f5_mode", "tts", str)
    f5_speed = ConfigKey("f5_speed", 1.0, float)
    f5_nfe_step = ConfigKey("f5_nfe_step", 32, int)
    f5_crossfade = ConfigKey("f5_crossfade", 0.15, float)
    f5_nfe = f5_nfe_step

    # --- GPT-SoVITS ---
    gpt_sovits_slice_mode = ConfigKey("gpt_sovits_slice_mode", "No slice", str)
    gpt_sovits_low_vram = ConfigKey("gpt_sovits_low_vram", False, bool)
    gpt_sovits_top_p = ConfigKey("gpt_sovits_top_p", 100, int)
    gpt_sovits_top_k = ConfigKey("gpt_sovits_top_k", 5, int)
    gpt_sovits_temperature = ConfigKey("gpt_sovits_temperature", 100, int)
    gpt_sovits_speed = ConfigKey("gpt_sovits_speed", 100, int)
    slice_mode = gpt_sovits_slice_mode
    low_vram_gpt_sovits = gpt_sovits_low_vram
    top_p_gpt_sovits = gpt_sovits_top_p
    top_k_gpt_sovits = gpt_sovits_top_k
    temperature_gpt_sovits = gpt_sovits_temperature
    speed_gpt_sovits = gpt_sovits_speed

    # --- StyleTTS2 ---
    styletts2_alpha = ConfigKey("styletts2_alpha", 20, int)
    styletts2_beta = ConfigKey("styletts2_beta", 20, int)
    styletts2_embedding_scale = ConfigKey("styletts2_embedding_scale", 1, int)
    styletts2_diffusion_steps = ConfigKey("styletts2_diffusion_steps", 100, int)
    style_alpha = styletts2_alpha
    style_beta = styletts2_beta
    style_embedding_scale = styletts2_embedding_scale
    style_diffusion_steps = styletts2_diffusion_steps

    # --- FishSpeech ---
    fish_use_torch_compile = ConfigKey("fish_use_torch_compile", False, bool)
    fish_temperature = ConfigKey("fish_temperature", 70, int)
    fish_repetition = ConfigKey("fish_repetition", 12, int)
    fish_top_p = ConfigKey("fish_top_p", 70, int)
    fish_max_length = ConfigKey("fish_max_length", 2048, int)
    fish_use_cache = ConfigKey("fish_use_cache", False, bool)
    fish_iterative_prompt = ConfigKey("fish_iterative_prompt", True, bool)

    # --- DIA ---
    dia_use_torch_compile = ConfigKey("dia_use_torch_compile", False, bool)
    dia_temperature = ConfigKey("dia_temperature", 135, int)
    dia_top_k = ConfigKey("dia_top_k", 45, int)
    dia_top_p = ConfigKey("dia_top_p", 95, int)

    # --- Llasa ---
    llasa_temperature = ConfigKey("llasa_temperature", 80, int)
    llasa_top_p = ConfigKey("llasa_top_p", 90, int)
    llasa_max_length = ConfigKey("llasa_max_length", 2048, int)
    llasa_mode = ConfigKey("llasa_mode", "1b", str)

    # --- Orpheus ---
    orpheus_temperature = ConfigKey("orpheus_temperature", 85, int)
    orpheus_top_p = ConfigKey("orpheus_top_p", 95, int)
    orpheus_repetition = ConfigKey("orpheus_repetition", 115, int)
    orpheus_top_k = ConfigKey("orpheus_top_k", 50, int)
    orpheus_max_new_tokens = ConfigKey("orpheus_max_new_tokens", 1200, int)
    orpehus_temperature = orpheus_temperature
    orpehus_top_p = orpheus_top_p
    orpehus_repetition = orpheus_repetition
    orpehus_top_k = orpheus_top_k
    orpehus_max_new_tokens = orpheus_max_new_tokens

    # --- Spark ---
    spark_top_p = ConfigKey("spark_top_p", 90, int)
    spark_temperature = ConfigKey("spark_temperature", 90, int)
    spark_top_k = ConfigKey("spark_top_k", 50, int)
    spark_max_new_tokens = ConfigKey("spark_max_new_tokens", 3000, int)

    # --- Chatterbox ---
    chatterbox_top_p = ConfigKey("chatterbox_top_p", 80, int)
    chatterbox_temperature = ConfigKey("chatterbox_temperature", 80, int)
    chatterbox_min_p = ConfigKey("chatterbox_min_p", 5, int)
    chatterbox_max_new_tokens = ConfigKey("chatterbox_max_new_tokens", 4096, int)
    chatterbox_exaggeration = ConfigKey("chatterbox_exaggeration", 50, int)
    chatterbox_repetition_penalty = ConfigKey("chatterbox_repetition_penalty", 100, int)
    chatterbox_cfg_weight = ConfigKey("chatterbox_cfg_weight", 50, int)

    # --- Higgs ---
    higgs_top_p = ConfigKey("higgs_top_p", 90, int)
    higgs_temperature = ConfigKey("higgs_temperature", 90, int)
    higgs_top_k = ConfigKey("higgs_top_k", 50, int)
    higgs_max_new_tokens = ConfigKey("higgs_max_new_tokens", 3000, int)
    higgs_ras_win_len = ConfigKey("higgs_ras_win_len", 15, int)
    higgs_ras_win_max_num_repeat = ConfigKey("higgs_ras_win_max_num_repeat", 4, int)

    # --- Vibe ---
    vibe_mode = ConfigKey("vibe_mode", "1.5B", str)
    vibe_cfg_scale = ConfigKey("vibe_cfg_scale", 10, int)
    vibe_inference_steps = ConfigKey("vibe_inference_steps", 50, int)
    vibe_temperature = ConfigKey("vibe_temperature", 100, int)
    vibe_do_sample = ConfigKey("vibe_do_sample", True, bool)
    vibe_top_p = ConfigKey("vibe_top_p", 90, int)
    vibe_top_k = ConfigKey("vibe_top_k", 50, int)
    vibe_dosmaple = vibe_do_sample

    # --- Qwen3TTS ---
    qwen_instruct = ConfigKey("qwen_instruct", "", str)
    qwen_language = ConfigKey("qwen_language", "English", str)
    qwen_model_version = ConfigKey("qwen_model_version", "1.7B-Base", str)

    # --- CSM ---
    csm_temperature = ConfigKey("csm_temperature", 80, int)
    csm_top_k = ConfigKey("csm_top_k", 50, int)

    # --- DMO Speech 2 ---
    dmo_temperature = ConfigKey("dmo_temperature", 80, int)
    dmo_teacher_steps = ConfigKey("dmo_teacher_steps", 16, int)
    dmo_teacher_stopping_time = ConfigKey("dmo_teacher_stopping_time", 7, int)
    dmo_student_start_step = ConfigKey("dmo_student_start_step", 1, int)
    dmo_top_k = ConfigKey("dmo_top_k", 50, int)
    dmo_speech2_temperature = dmo_temperature
    dmo_speech2_teacher_steps = dmo_teacher_steps
    dmo_speech2_teacher_stopping_time = dmo_teacher_stopping_time
    dmo_speech2_student_start_step = dmo_student_start_step

    # --- Chat ---
    chat_model = ConfigKey("chat_model", "Qwen/Qwen3-1.7B", str)

    def __init__(self):
        self._defaults: dict = {}
        self._data: dict = {}
        self._path = os.path.join(get_app_root(), "config", "settings.json")
        self._collect_defaults()
        self.load()

    def _collect_defaults(self):
        """Collect default values from all ConfigKey class attributes."""
        for attr_name in dir(self.__class__):
            attr = getattr(self.__class__, attr_name)
            if isinstance(attr, ConfigKey):
                self._defaults[attr.key] = attr.default

    def get(self, key: ConfigKey):
        val = self._data.get(key.key, key.default)
        if key.key == "f5_mode" and val == "generate":
            val = "tts"
        elif key.key == "f5_speed" and isinstance(val, (int, float)) and val > 3:
            val = float(val) / 10.0
        elif key.key == "f5_crossfade" and isinstance(val, (int, float)) and val > 1:
            val = float(val) / 100.0
        elif key.key == "vibe_mode" and val == "generate":
            val = "1.5B"
        if key.type_ is not None and val is not None:
            try:
                return key.type_(val)
            except (ValueError, TypeError):
                return key.default
        return val

    def set(self, key: ConfigKey, value):
        self._data[key.key] = value
        self.save()

    def reset(self):
        """Reset all settings to defaults."""
        self._data = dict(self._defaults)
        self.save()

    def load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path, 'r', encoding='utf-8') as f:
                    self._data = json.load(f)
                # Migrate nested sections from old Qt config if present
                if self._migrate_nested_sections():
                    self.save()
                return
            except (json.JSONDecodeError, OSError):
                pass

        # Attempt migration from old QSettings INI format
        old_path = os.path.join(os.path.dirname(self._path), "configv2.json")
        if os.path.exists(old_path):
            self._migrate_from_ini(old_path)
        else:
            self._data = dict(self._defaults)
        self.save()

    # Map from old nested (section, key) → new flat key
    _NESTED_KEY_MAP = {
        ("App", "accepts_disclaimer"): "accepted_disclaimer",
        ("App", "accepts_custom_disclaimer"): "accepts_custom_disclaimer",
        ("App", "fallout_4_directory"): "fallout_4_directory",
        ("App", "fallout_4_directory_check"): "fallout_4_directory_check",
        ("App", "first_start"): "first_start",
        ("App", "auto_play"): "auto_play",
        ("App", "check_for_updates"): "check_for_updates",
        ("App", "download_configs"): "download_configs",
        ("App", "output_dir"): "output_dir",
        ("App", "custom_references"): "custom_references",
        ("App", "huggingface_cache_dir"): "huggingface_cache_dir",
        ("App", "disable_ssl_verify"): "disableSSLVerify",
        ("App", "apbwe_enabled"): "apbwe_enabled",
        ("App", "rvc_enabled"): "rvc_enabled",
        ("App", "xwm_enabled"): "xwm_enabled",
        ("App", "keep_only_fuz"): "keep_only_fuz",
        ("App", "use_existing_lip"): "use_existing_lip",
        ("App", "themeColor"): "theme_color",
        ("App", "inline_reference_removal"): "inline_reference_removal",
        ("App", "dot_letter_fix"): "dot_letter_fix",
        ("App", "whitespace_normalization"): "whitespace_normalization",
        ("App", "lowercase_conversion"): "lowercase_conversion",
        ("App", "pad_short_phrases"): "pad_short_phrases",
        ("App", "max_text_size"): "max_text_size",
        ("App", "min_text_size"): "min_chunk_size",
        ("App", "api_only_mode"): "api_only_mode",
        ("APP", "huggingface_key"): "huggingface_key",
        ("TTS", "engine"): "engine",
        ("TTS", "device"): "device",
        ("TTS", "seed"): "seed",
        ("TTS", "auto_update_models"): "auto_update_models",
        ("TTS", "load_at_start"): "load_engine_art_start",
        ("Chat", "model"): "chat_model",
        ("Personalization", "themeColor"): "theme_color",
        ("bulk", "include_subdir"): "include_subdir",
        ("bulk", "replace_existing"): "replace_existing",
        ("bulk", "threads"): "threads",
        ("multigen", "total"): "multigen_total",
        ("ez", "total"): "ez_total",
        ("F5", "speed_factor"): "f5_speed",
        ("F5", "mode"): "f5_mode",
        ("F5", "nfe_step"): "f5_nfe_step",
        ("GPT_SoVITS", "slice_mode"): "gpt_sovits_slice_mode",
        ("GPT_SoVITS", "low_vram"): "gpt_sovits_low_vram",
        ("GPT_SoVITS", "model_temperature"): "gpt_sovits_temperature",
        ("GPT_SoVITS", "top_p"): "gpt_sovits_top_p",
        ("GPT_SoVITS", "top_k"): "gpt_sovits_top_k",
        ("GPT_SoVITS", "speed"): "gpt_sovits_speed",
        ("StyleTTS2", "alpha"): "styletts2_alpha",
        ("StyleTTS2", "beta"): "styletts2_beta",
        ("StyleTTS2", "embedding_scale"): "styletts2_embedding_scale",
        ("StyleTTS2", "diffusion_steps"): "styletts2_diffusion_steps",
        ("FishSpeech", "use_torch_compile"): "fish_use_torch_compile",
        ("FishSpeech", "model_temperature"): "fish_temperature",
        ("FishSpeech", "model_repetition"): "fish_repetition",
        ("FishSpeech", "top_p"): "fish_top_p",
        ("FishSpeech", "max_length"): "fish_max_length",
        ("FishSpeech", "use_memory_cache"): "fish_use_cache",
        ("FishSpeech", "iterative_prompt"): "fish_iterative_prompt",
        ("DIA", "model_temperature"): "dia_temperature",
        ("DIA", "top_k"): "dia_top_k",
        ("DIA", "top_p"): "dia_top_p",
        ("DIA", "use_torch_compile"): "dia_use_torch_compile",
        ("Llasa", "model_temperature"): "llasa_temperature",
        ("Llasa", "top_p"): "llasa_top_p",
        ("Llasa", "max_length"): "llasa_max_length",
        ("Llasa", "mode"): "llasa_mode",
        ("Orpehus", "model_temperature"): "orpheus_temperature",
        ("Orpehus", "top_p"): "orpheus_top_p",
        ("Orpehus", "model_repetition"): "orpheus_repetition",
        ("Orpehus", "top_k"): "orpheus_top_k",
        ("Orpehus", "max_new_tokens"): "orpheus_max_new_tokens",
        ("Spark", "model_temperature"): "spark_temperature",
        ("Spark", "top_p"): "spark_top_p",
        ("Spark", "top_k"): "spark_top_k",
        ("Spark", "max_new_tokens"): "spark_max_new_tokens",
        ("Chatterbox", "model_temperature"): "chatterbox_temperature",
        ("Chatterbox", "top_p"): "chatterbox_top_p",
        ("Chatterbox", "min_p"): "chatterbox_min_p",
        ("Chatterbox", "max_new_tokens"): "chatterbox_max_new_tokens",
        ("Chatterbox", "exaggeration"): "chatterbox_exaggeration",
        ("Chatterbox", "repetition_penalty"): "chatterbox_repetition_penalty",
        ("Chatterbox", "cfg_weight"): "chatterbox_cfg_weight",
        ("Higgs", "model_temperature"): "higgs_temperature",
        ("Higgs", "top_p"): "higgs_top_p",
        ("Higgs", "top_k"): "higgs_top_k",
        ("Higgs", "max_new_tokens"): "higgs_max_new_tokens",
        ("Higgs", "ras_win_len"): "higgs_ras_win_len",
        ("Higgs", "ras_win_max_num_repeat"): "higgs_ras_win_max_num_repeat",
        ("Vibe", "mode"): "vibe_mode",
        ("Vibe", "cfg_scale"): "vibe_cfg_scale",
        ("Vibe", "inference_steps"): "vibe_inference_steps",
        ("Vibe", "model_temperature"): "vibe_temperature",
        ("Vibe", "do_sample"): "vibe_do_sample",
        ("Vibe", "top_p"): "vibe_top_p",
        ("Vibe", "top_k"): "vibe_top_k",
        ("Qwen3TTS", "instruct"): "qwen_instruct",
        ("Qwen3TTS", "language"): "qwen_language",
        ("Qwen3TTS", "model_version"): "qwen_model_version",
        ("DMSpeech2", "model_temperature"): "dmo_temperature",
        ("DMSpeech2", "teacher_steps"): "dmo_teacher_steps",
        ("DMSpeech2", "teacher_stopping_time"): "dmo_teacher_stopping_time",
        ("DMSpeech2", "student_start_step"): "dmo_student_start_step",
        ("RVC", "rvc_pitch"): "rvc_pitch",
        ("RVC", "rvc_hop_length"): "rvc_hop_length",
        ("RVC", "rvc_training_data_size"): "rvc_training_data_size",
        ("RVC", "rvc_index_influence"): "rvc_index_influence",
        ("RVC", "rvc_volume_envelope_Rslider"): "rvc_volume_envelope",
        ("RVC", "rvc_protect"): "rvc_protect",
        ("RVC", "rvc_filter_radius"): "rvc_filter_radius",
        ("RVC", "rvc_autotune_checkbox"): "rvc_autotune",
        ("RVC", "rvc_split_audio"): "rvc_split_audio",
        ("RVC", "rvc_pitch_extraction"): "rvc_pitch_extraction",
        ("RVC", "rvc_mode"): "rvc_mode",
        ("RVC", "rvc_eleven_labs_key"): "rvc_eleven_labs_key",
        ("RVC", "rvc_embedding_model"): "rvc_embedder_model",
    }

    def _migrate_nested_sections(self) -> bool:
        """Flatten old Qt nested config sections into flat keys. Returns True if any migration happened."""
        # Detect nested sections (dicts that aren't list/None values)
        nested_sections = {k: v for k, v in self._data.items() if isinstance(v, dict)}
        if not nested_sections:
            return False

        migrated = False
        for (section, old_key), flat_key in self._NESTED_KEY_MAP.items():
            if section in nested_sections and old_key in nested_sections[section]:
                old_val = nested_sections[section][old_key]
                # Only migrate if the flat key still has its default value
                if flat_key in self._defaults and self._data.get(flat_key) == self._defaults[flat_key]:
                    self._data[flat_key] = old_val
                    migrated = True

        # Remove nested sections after migration
        if migrated:
            for section in nested_sections:
                del self._data[section]

        return migrated

    def _migrate_from_ini(self, ini_path: str):
        """Parse old qfluentwidgets QSettings INI config into self._data."""
        # The old file may already be JSON (from a previous migration).
        try:
            with open(ini_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._data = {**self._defaults, **data}
            return
        except (json.JSONDecodeError, OSError, ValueError):
            pass
        parser = configparser.ConfigParser()
        parser.read(ini_path, encoding='utf-8')
        self._data = dict(self._defaults)
        section = 'General'
        if parser.has_section(section):
            for key, value in parser.items(section):
                # Coerce bool strings
                if value.lower() == 'true':
                    self._data[key] = True
                elif value.lower() == 'false':
                    self._data[key] = False
                else:
                    # Try numeric, fall back to string
                    try:
                        self._data[key] = int(value)
                    except ValueError:
                        try:
                            self._data[key] = float(value)
                        except ValueError:
                            self._data[key] = value

    def save(self):
        """Atomic write to settings.json."""
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        tmp = self._path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, indent=2, default=str)
        os.replace(tmp, self._path)


cfg = Config()


YEAR = 2025
AUTHOR = "Bryant21"
VERSION = '2.2.0'
NEXUS_URL = "https://www.nexusmods.com/fallout4/mods/86525"
HELP_URL = "https://github.com/falltalk/falltalk4"
FEEDBACK_URL = "https://github.com/falltalk/falltalk4/issues"
RELEASE_URL = "https://github.com/falltalk/falltalk4/releases/latest"
KOFI_URL = "https://ko-fi.com/bryant21"
DISCORD_URL = "https://discord.gg/FgKrxdnQdG"
HUGGING_FACE = "https://huggingface.co/falltalk/falltalk4"
REPO = "falltalk/falltalk4"

DISCLAIMER = """
This code and the accompanying FallTalk AI models are provided subject to the terms and conditions of the End User License Agreement (EULA) of Zenimax Media, Inc., the original rights holder of the Fallout franchise. By using this code or the FallTalk AI models, you agree to comply with the following permitted and prohibited uses, as well as all terms outlined in the Zenimax Media EULA.

By accessing and using the FallTalk, you hereby agree to the following terms and conditions:

Public Disclosure of AI Synthesis: You are obligated to clearly inform any and all end-users that the speech content they are interacting with has been synthesized using the FallTalk AI models. This disclosure should be made in a manner that is prominent and easily understandable.

Permitted Use: You agree to use the FallTalk AI models exclusively for the following purposes:

    • Personal Use: Utilizing the FallTalk AI models for personal, non-commercial projects and activities that do not involve the distribution or sharing of synthesized speech content with others.
    • Research: Conducting academic or scientific research in the field of artificial intelligence, speech synthesis, or related disciplines.
    • Non-Commercial Mod Creation: Developing and distributing modifications (mods) for the game Fallout 4 that are available to the public free of charge.

Prohibited Use: You are expressly prohibited from using the FallTalk AI models for any commercial purposes, including but not limited to:

    • Selling or licensing the synthesized speech content.
    • Incorporating the synthesized speech into any commercial product or service.
    • Creating or distributing any pornographic or adult material.

Compliance with Laws and Regulations: You agree to comply with all applicable laws, regulations, and ethical standards in your use of the FallTalk models. This includes, but is not limited to, laws concerning intellectual property, privacy, and consumer protection. We assume no responsibility for any illegal use of the codebase.

Limitation of Liability: In no event shall the developers, contributors, or distributors of this modding tool be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this tool or the model(s), even if advised of the possibility of such damage.

Acknowledgment of Rights Holder: Zenimax Media, Inc. is the original rights holder of the Fallout franchise and all related intellectual property. This code and the FallTalk AI models are provided for the specific purposes outlined above, and any use outside of these parameters may violate Zenimax Media's intellectual property rights.

By clicking "Agree", you signify your acceptance of these terms and your commitment to abide by them. If you do not agree to these terms, you may not use this app.
"""

CUSTOM_DISCLAIMER = """
By proceeding with the import of custom models for use in Fallout 4 modding, you hereby acknowledge and agree to the following terms:

    •  No Infringement: You warrant that the use of the model(s) does not infringe upon the rights of any third party, including but not limited to privacy rights, publicity rights, and intellectual property rights.

    •  Indemnification: You agree to indemnify and hold harmless the developers, contributors, and distributors of this modding tool from any claims, damages, losses, or expenses (including attorney's fees) arising out of or in connection with your use of the model, including any claims that the model infringes upon the rights of any third party.

    •  Limitation of Liability: In no event shall the developers, contributors, or distributors of this modding tool be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this tool or the model(s), even if advised of the possibility of such damage.

By clicking "Agree", you signify your acceptance of these terms and your commitment to abide by them. If you do not agree to these terms, you must not proceed with the import of the model(s).
"""
