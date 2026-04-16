"""
Consolidated engine settings — replaces all 20 src/settings/*.py files.
Single draw_engine_settings(state) dispatches to per-engine imgui controls.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui_imgui.state import AppState

from src.config.config import cfg
from src.enums.engine_type import EngineType


def draw_engine_settings(state: AppState):
    """Render engine-specific settings. Called inside the drawer or Settings panel."""
    from imgui_bundle import imgui

    if state.engine_type is None:
        imgui.text("No engine loaded.")
        return

    match state.engine_type:
        case EngineType.GPT_SOVITS:
            _draw_gpt_sovits_settings()
        case EngineType.F5:
            _draw_f5_settings()
        case EngineType.FISH_SPEECH:
            _draw_fish_settings()
        case EngineType.STYLE_TTS2:
            _draw_styletts2_settings()
        case EngineType.DIA:
            _draw_dia_settings()
        case EngineType.LLASA:
            _draw_llasa_settings()
        case EngineType.ORPHEUS:
            _draw_orpheus_settings()
        case EngineType.SPARK:
            _draw_spark_settings()
        case EngineType.CSM:
            _draw_csm_settings()
        case EngineType.HIGGS:
            _draw_higgs_settings()
        case EngineType.CHATTERBOX:
            _draw_chatterbox_settings()
        case EngineType.DMOSPEECH2:
            _draw_dmo_settings()
        case EngineType.VIBE:
            _draw_vibe_settings()
        case EngineType.QWEN3_TTS:
            _draw_qwen_settings()
        case EngineType.RVC:
            _draw_rvc_settings()
        case _:
            imgui.text(f"No settings for {state.engine_type}")

    # Common text processing + LIP/FUZ settings (shared across all engines)
    _draw_text_processing_settings()
    _draw_lip_fuz_settings()


def draw_rvc_settings_panel():
    """Render the standalone RVC settings block for the Settings page."""
    _draw_rvc_settings()


def draw_engine_quick_tuning(state: AppState):
    """Render the primary engine controls directly on the generation page."""
    from imgui_bundle import imgui

    if state.engine_type is None:
        return

    drew_any = False
    imgui.text_disabled("Engine Controls")
    imgui.separator()

    match state.engine_type:
        case EngineType.F5:
            changed, val = imgui.slider_float("Speed##quick_f5", cfg.get(cfg.f5_speed), 0.1, 3.0, format="%.2f")
            if changed:
                cfg.set(cfg.f5_speed, val)
            _int_slider("NFE Steps##quick_f5", cfg.f5_nfe_step, 1, 128)
            changed, val = imgui.slider_float("Crossfade##quick_f5", cfg.get(cfg.f5_crossfade), 0.0, 1.0, format="%.2f")
            if changed:
                cfg.set(cfg.f5_crossfade, val)
            drew_any = True
        case EngineType.GPT_SOVITS:
            _scaled_slider("Temperature##quick", cfg.gpt_sovits_temperature, 0, 200)
            _int_slider("Top K##quick", cfg.gpt_sovits_top_k, 1, 100)
            _scaled_slider("Top P##quick", cfg.gpt_sovits_top_p, 0, 100)
            _scaled_slider("Speed##quick", cfg.gpt_sovits_speed, 10, 200)
            drew_any = True
        case EngineType.FISH_SPEECH:
            _scaled_slider("Temperature##quick", cfg.fish_temperature, 0, 200)
            _int_slider("Repetition##quick", cfg.fish_repetition, 1, 30)
            _scaled_slider("Top P##quick", cfg.fish_top_p, 0, 100)
            drew_any = True
        case EngineType.STYLE_TTS2:
            _scaled_slider("Alpha##quick_style", cfg.styletts2_alpha, 0, 100)
            _scaled_slider("Beta##quick_style", cfg.styletts2_beta, 0, 100)
            _int_slider("Diffusion Steps##quick_style", cfg.styletts2_diffusion_steps, 1, 500)
            _int_slider("Embedding Scale##quick_style", cfg.styletts2_embedding_scale, 1, 3)
            drew_any = True
        case EngineType.DIA:
            _scaled_slider("Temperature##quick", cfg.dia_temperature, 0, 200)
            _int_slider("Top K##quick", cfg.dia_top_k, 1, 100)
            _scaled_slider("Top P##quick", cfg.dia_top_p, 0, 100)
            drew_any = True
        case EngineType.LLASA:
            _combo("Model Size##quick_llasa", cfg.llasa_mode, ["1b", "3b", "8b"])
            _scaled_slider("Temperature##quick", cfg.llasa_temperature, 0, 200)
            _scaled_slider("Top P##quick", cfg.llasa_top_p, 0, 100)
            _int_slider("Max Length##quick_llasa", cfg.llasa_max_length, 128, 4096)
            drew_any = True
        case EngineType.ORPHEUS:
            _scaled_slider("Temperature##quick", cfg.orpheus_temperature, 0, 200)
            _int_slider("Repetition##quick", cfg.orpheus_repetition, 1, 200)
            _scaled_slider("Top P##quick_orpheus", cfg.orpheus_top_p, 0, 100)
            drew_any = True
        case EngineType.SPARK:
            _scaled_slider("Temperature##quick", cfg.spark_temperature, 0, 200)
            _int_slider("Top K##quick", cfg.spark_top_k, 1, 100)
            _scaled_slider("Top P##quick_spark", cfg.spark_top_p, 0, 100)
            drew_any = True
        case EngineType.CHATTERBOX:
            _scaled_slider("Temperature##quick", cfg.chatterbox_temperature, 0, 200)
            _scaled_slider("Repetition Penalty##quick", cfg.chatterbox_repetition_penalty, 0, 200, scale=10.0)
            _scaled_slider("CFG Weight##quick_cb", cfg.chatterbox_cfg_weight, 0, 100)
            drew_any = True
        case EngineType.HIGGS:
            _scaled_slider("Temperature##quick", cfg.higgs_temperature, 0, 200)
            _scaled_slider("Top P##quick", cfg.higgs_top_p, 0, 100)
            _scaled_slider("Top K##quick_higgs", cfg.higgs_top_k, 0, 100)
            drew_any = True
        case EngineType.VIBE:
            _scaled_slider("Temperature##quick", cfg.vibe_temperature, 0, 200)
            _scaled_slider("Top P##quick", cfg.vibe_top_p, 0, 100)
            _scaled_slider("CFG Scale##quick_vibe", cfg.vibe_cfg_scale, 0, 100)
            drew_any = True
        case EngineType.CSM:
            _scaled_slider("Temperature##quick", cfg.csm_temperature, 0, 200)
            _int_slider("Top K##quick", cfg.csm_top_k, 1, 100)
            drew_any = True
        case EngineType.DMOSPEECH2:
            _scaled_slider("Temperature##quick", cfg.dmo_temperature, 0, 200)
            _int_slider("Teacher Steps##quick_dmo", cfg.dmo_teacher_steps, 1, 50)
            drew_any = True
        case EngineType.QWEN3_TTS:
            _combo("Language##quick_qwen", cfg.qwen_language, ["Auto", "Chinese", "English", "Japanese", "Korean"])
            _combo("Model Version##quick_qwen", cfg.qwen_model_version, ["1.7B-Base", "0.6B-Base"])
            _text_input("Instruct##quick_qwen", cfg.qwen_instruct)
            drew_any = True
        case EngineType.RVC:
            _combo(
                "Pitch Extraction##quick_rvc",
                cfg.rvc_pitch_extraction,
                ["rmvpe", "crepe", "crepe-tiny", "fcpe", "hybrid[rmvpe+fcpe]"],
            )
            _int_slider("Pitch Adjustment##quick_rvc", cfg.rvc_pitch, -24, 24)
            _scaled_slider("Index Influence##quick_rvc", cfg.rvc_index_influence, 0, 100)
            _checkbox("Autotune##quick_rvc", cfg.rvc_autotune)
            drew_any = True
        case _:
            pass

    if not drew_any:
        imgui.text_disabled("No primary controls for this engine. Use the settings drawer for full controls.")


# -- Helper for scaled int sliders (stored as int, displayed as float / 100) --

def _scaled_slider(label: str, config_key, lo: float, hi: float, scale: float = 100.0):
    """Slider for int config keys that represent a scaled float value.
    e.g. stored as 80, displayed as 0.80, range 0.0 - 1.0 with scale=100.
    """
    from imgui_bundle import imgui
    raw = cfg.get(config_key)
    display_val = raw / scale
    changed, new_val = imgui.slider_float(label, display_val, lo / scale, hi / scale, format="%.2f")
    if changed:
        cfg.set(config_key, int(new_val * scale))


def _int_slider(label: str, config_key, lo: int, hi: int):
    from imgui_bundle import imgui
    changed, val = imgui.slider_int(label, cfg.get(config_key), lo, hi)
    if changed:
        cfg.set(config_key, val)


def _checkbox(label: str, config_key):
    from imgui_bundle import imgui
    changed, val = imgui.checkbox(label, cfg.get(config_key))
    if changed:
        cfg.set(config_key, val)


def _combo(label: str, config_key, options: list[str]):
    from imgui_bundle import imgui
    current = cfg.get(config_key)
    idx = options.index(current) if current in options else 0
    changed, new_idx = imgui.combo(label, idx, options)
    if changed:
        cfg.set(config_key, options[new_idx])


def _text_input(label: str, config_key, hint: str = ""):
    from imgui_bundle import imgui
    current = cfg.get(config_key) or ""
    imgui.set_next_item_width(-1)
    changed, new_val = imgui.input_text(label, current)
    if changed:
        cfg.set(config_key, new_val)



def _draw_f5_settings():
    from imgui_bundle import imgui
    imgui.text("F5 Settings")
    imgui.separator()
    # f5_speed is a float config
    changed, val = imgui.slider_float("Speed##f5", cfg.get(cfg.f5_speed), 0.1, 3.0, format="%.2f")
    if changed:
        cfg.set(cfg.f5_speed, val)
    _int_slider("NFE Steps##f5", cfg.f5_nfe_step, 1, 128)
    changed, val = imgui.slider_float("Crossfade##f5", cfg.get(cfg.f5_crossfade), 0.0, 1.0, format="%.2f")
    if changed:
        cfg.set(cfg.f5_crossfade, val)


def _draw_gpt_sovits_settings():
    from imgui_bundle import imgui
    imgui.text("GPT-SoVITS Settings")
    imgui.separator()
    _combo("Slice Mode##gpt", cfg.gpt_sovits_slice_mode,
           ["No slice", "Slice by English punct", "Slice by every punct",
            "Slice every 4 sentences", "Slice per 50 characters"])
    _checkbox("Low VRAM##gpt", cfg.gpt_sovits_low_vram)
    _scaled_slider("Top P##gpt", cfg.gpt_sovits_top_p, 0, 100)
    _int_slider("Top K##gpt", cfg.gpt_sovits_top_k, 1, 100)
    _scaled_slider("Temperature##gpt", cfg.gpt_sovits_temperature, 0, 200)
    _scaled_slider("Speed##gpt", cfg.gpt_sovits_speed, 10, 200)


def _draw_fish_settings():
    from imgui_bundle import imgui
    imgui.text("FishSpeech Settings")
    imgui.separator()
    _checkbox("Use Torch Compile##fish", cfg.fish_use_torch_compile)
    _scaled_slider("Temperature##fish", cfg.fish_temperature, 0, 200)
    _int_slider("Repetition##fish", cfg.fish_repetition, 1, 30)
    _scaled_slider("Top P##fish", cfg.fish_top_p, 0, 100)
    _int_slider("Max Length##fish", cfg.fish_max_length, 128, 4096)
    _checkbox("Use Memory Cache##fish", cfg.fish_use_cache)
    _checkbox("Iterative Prompt##fish", cfg.fish_iterative_prompt)


def _draw_styletts2_settings():
    from imgui_bundle import imgui
    imgui.text("StyleTTS2 Settings")
    imgui.separator()
    _scaled_slider("Alpha##style", cfg.styletts2_alpha, 0, 100)
    _scaled_slider("Beta##style", cfg.styletts2_beta, 0, 100)
    _int_slider("Diffusion Steps##style", cfg.styletts2_diffusion_steps, 1, 500)
    _int_slider("Embedding Scale##style", cfg.styletts2_embedding_scale, 1, 3)


def _draw_dia_settings():
    from imgui_bundle import imgui
    imgui.text("DIA Settings")
    imgui.separator()
    _scaled_slider("Temperature##dia", cfg.dia_temperature, 0, 200)
    _scaled_slider("Top P##dia", cfg.dia_top_p, 0, 100)
    _int_slider("Top K##dia", cfg.dia_top_k, 1, 100)


def _draw_llasa_settings():
    from imgui_bundle import imgui
    imgui.text("Llasa Settings")
    imgui.separator()
    _combo("Model Size##llasa", cfg.llasa_mode, ["1b", "3b", "8b"])
    _scaled_slider("Temperature##llasa", cfg.llasa_temperature, 0, 200)
    _scaled_slider("Top P##llasa", cfg.llasa_top_p, 0, 100)
    _int_slider("Max Length##llasa", cfg.llasa_max_length, 128, 4096)


def _draw_orpheus_settings():
    from imgui_bundle import imgui
    imgui.text("Orpheus Settings")
    imgui.separator()
    _scaled_slider("Temperature##orpheus", cfg.orpheus_temperature, 0, 200)
    _scaled_slider("Top P##orpheus", cfg.orpheus_top_p, 0, 100)
    _int_slider("Top K##orpheus", cfg.orpheus_top_k, 1, 100)
    _int_slider("Max New Tokens##orpheus", cfg.orpheus_max_new_tokens, 100, 4096)
    _int_slider("Repetition##orpheus", cfg.orpheus_repetition, 1, 200)


def _draw_spark_settings():
    from imgui_bundle import imgui
    imgui.text("Spark Settings")
    imgui.separator()
    _scaled_slider("Temperature##spark", cfg.spark_temperature, 0, 200)
    _scaled_slider("Top P##spark", cfg.spark_top_p, 0, 100)
    _int_slider("Top K##spark", cfg.spark_top_k, 1, 100)
    _int_slider("Max New Tokens##spark", cfg.spark_max_new_tokens, 100, 8000)


def _draw_csm_settings():
    from imgui_bundle import imgui
    imgui.text("CSM Settings")
    imgui.separator()
    _scaled_slider("Temperature##csm", cfg.csm_temperature, 0, 200)
    _int_slider("Top K##csm", cfg.csm_top_k, 1, 100)


def _draw_higgs_settings():
    from imgui_bundle import imgui
    imgui.text("Higgs Settings")
    imgui.separator()
    _scaled_slider("Temperature##higgs", cfg.higgs_temperature, 0, 200)
    _scaled_slider("Top P##higgs", cfg.higgs_top_p, 0, 100)
    _scaled_slider("Top K##higgs", cfg.higgs_top_k, 0, 100)
    _scaled_slider("Max New Tokens##higgs", cfg.higgs_max_new_tokens, 100, 8000, scale=1.0)
    _scaled_slider("RAS Window Length##higgs", cfg.higgs_ras_win_len, 1, 50, scale=1.0)
    _scaled_slider("RAS Max Repeats##higgs", cfg.higgs_ras_win_max_num_repeat, 1, 20, scale=1.0)


def _draw_chatterbox_settings():
    from imgui_bundle import imgui
    imgui.text("Chatterbox Settings")
    imgui.separator()
    _scaled_slider("Temperature##cb", cfg.chatterbox_temperature, 0, 200)
    _scaled_slider("Top P##cb", cfg.chatterbox_top_p, 0, 100)
    _scaled_slider("Min P##cb", cfg.chatterbox_min_p, 0, 100, scale=1000.0)
    _scaled_slider("Exaggeration##cb", cfg.chatterbox_exaggeration, 0, 100)
    _scaled_slider("Repetition Penalty##cb", cfg.chatterbox_repetition_penalty, 0, 200, scale=10.0)
    _scaled_slider("CFG Weight##cb", cfg.chatterbox_cfg_weight, 0, 100)


def _draw_dmo_settings():
    from imgui_bundle import imgui
    imgui.text("DMO Speech 2 Settings")
    imgui.separator()
    _scaled_slider("Temperature##dmo", cfg.dmo_temperature, 0, 200)
    _int_slider("Teacher Steps##dmo", cfg.dmo_teacher_steps, 1, 50)
    _scaled_slider("Teacher Stopping Time##dmo", cfg.dmo_teacher_stopping_time, 1, 20, scale=100.0)
    _int_slider("Student Start Step##dmo", cfg.dmo_student_start_step, 1, 10)


def _draw_vibe_settings():
    from imgui_bundle import imgui
    imgui.text("Vibe Settings")
    imgui.separator()
    _checkbox("Use Sampling##vibe", cfg.vibe_do_sample)
    _scaled_slider("CFG Scale##vibe", cfg.vibe_cfg_scale, 0, 100)
    _scaled_slider("Temperature##vibe", cfg.vibe_temperature, 0, 200)
    _scaled_slider("Top P##vibe", cfg.vibe_top_p, 0, 100)
    _int_slider("Inference Steps##vibe", cfg.vibe_inference_steps, 1, 200)
    _combo("Model Size##vibe", cfg.vibe_mode, ["1.5B", "7B"])


def _draw_qwen_settings():
    from imgui_bundle import imgui
    imgui.text("Qwen3 TTS Settings")
    imgui.separator()
    _combo("Model Version##qwen", cfg.qwen_model_version, ["1.7B-Base", "0.6B-Base"])
    _combo("Language##qwen", cfg.qwen_language, ["Auto", "Chinese", "English", "Japanese", "Korean"])
    imgui.text("Instruct:")
    _text_input("##qwen_instruct", cfg.qwen_instruct)


def _draw_rvc_settings():
    from imgui_bundle import imgui
    imgui.text("RVC Settings")
    imgui.separator()
    _combo("Pitch Extraction##rvc", cfg.rvc_pitch_extraction,
           ["rmvpe", "crepe", "crepe-tiny", "fcpe", "hybrid[rmvpe+fcpe]"])
    _int_slider("Training Data Size##rvc", cfg.rvc_training_data_size, 0, 1000)
    _scaled_slider("Index Influence##rvc", cfg.rvc_index_influence, 0, 100)
    _scaled_slider("Hop Length##rvc", cfg.rvc_hop_length, 1, 512, scale=1.0)
    _int_slider("Pitch Adjustment##rvc", cfg.rvc_pitch, -24, 24)
    _scaled_slider("Volume Envelope##rvc", cfg.rvc_volume_envelope, 0, 100)
    _scaled_slider("Protect##rvc", cfg.rvc_protect, 0, 50)
    _int_slider("Filter Radius##rvc", cfg.rvc_filter_radius, 0, 10)
    _checkbox("Autotune##rvc", cfg.rvc_autotune)
    _checkbox("Split Audio##rvc", cfg.rvc_split_audio)
    imgui.separator()
    imgui.text("Eleven Labs")
    imgui.text("API Key:")
    _text_input("##rvc_eleven_key", cfg.rvc_eleven_labs_key)


def _draw_text_processing_settings():
    from imgui_bundle import imgui
    if imgui.collapsing_header("Text Processing"):
        _int_slider("Max Characters##tp", cfg.max_text_size, 10, 2000)
        _int_slider("Min Characters##tp", cfg.min_chunk_size, 1, 500)
        _checkbox("Pad Short Phrases##tp", cfg.pad_short_phrases)
        _checkbox("Lowercase Conversion##tp", cfg.lowercase_conversion)
        _checkbox("Whitespace Normalization##tp", cfg.whitespace_normalization)
        _checkbox("Dot-Letter Fix##tp", cfg.dot_letter_fix)
        _checkbox("Inline Reference Removal##tp", cfg.inline_reference_removal)


def _draw_lip_fuz_settings():
    from imgui_bundle import imgui
    if imgui.collapsing_header("LIP/FUZ Settings"):
        _checkbox("Create FUZ##lip", cfg.xwm_enabled)
        _checkbox("Keep Only FUZ##lip", cfg.keep_only_fuz)
