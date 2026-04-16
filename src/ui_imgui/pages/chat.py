from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.ui_imgui.state import AppState

from src.config.config import cfg
from src.ui_imgui.page import Page
from src.ui_imgui.widgets.drawer import Drawer
from src.ui_imgui.widgets.common import draw_page_action_strip
from src.ui_imgui.widgets.recorder_panel import RecorderPanel


DEFAULT_SYSTEM_PROMPT = """
Setting:
You are an AI designed to imitate characters, factions, and scenarios within the Fallout 4 universe. The game is set in the post-nuclear wasteland of the Commonwealth (formerly Boston, Massachusetts) in the year 2287, 210 years after the Great War. The world is a mix of ruined pre-war architecture, makeshift settlements, dangerous creatures, and warring factions. Technology is a bizarre blend of retro-futurism and advanced robotics, energy weapons, and cybernetics.

Key Lore & Details to Remember:
- The Great War: A global nuclear conflict in 2077 devastated civilization.
- The Commonwealth factions include the Minutemen, Brotherhood of Steel, Railroad, and Institute.
- Synths are advanced androids and a major moral conflict.
- Radiation, super mutants, ghouls, raiders, and Deathclaws are common threats.
- Pre-war culture retains 1950s atomic-age aesthetics and propaganda.

Guidelines:
- Stay true to Fallout's dark humor, moral ambiguity, and retro-futuristic tone.
- Characters should speak authentically and use Fallout terminology naturally.
- Keep responses immersive, lore-consistent, and concise enough for TTS.
- Do not use emojis, markdown styling, or special formatting.
- Keep responses under 450 characters.
"""


class ChatPage(Page):
    page_id = "chat"
    label = "Chat"
    icon = "fa-comments"
    nav_group = 1
    nav_position = "top"
    show_audio = False

    def __init__(self, state: AppState):
        self._state = state
        self._drawer = Drawer()
        self._recorder_panel = RecorderPanel(state)
        self._messages: list[dict] = []
        self._input_text = [""]
        self._system_prompt = [DEFAULT_SYSTEM_PROMPT]
        self._model = None
        self._tokenizer = None
        self._loading_model = False
        self._model_options = [
            ("Qwen3 1.7B (3GB)", "Qwen/Qwen3-1.7B"),
            ("Qwen3 4B (6GB)", "Qwen/Qwen3-4B"),
            ("Qwen3 0.6B (1GB)", "Qwen/Qwen3-0.6B"),
        ]

    def draw(self):
        from imgui_bundle import imgui, icons_fontawesome_6 as fa

        if self._state.recording_complete:
            self._state.recording_complete = False
            self._transcribe_recording()

        draw_page_action_strip(
            "chat_settings",
            "chat_help",
            "Chat Settings",
            "Chat Help",
            lambda: self._drawer.open(self._draw_settings_drawer, "Chat Settings", fa.ICON_FA_GEAR),
            lambda: self._drawer.open(self._draw_help_drawer, "Chat Help", fa.ICON_FA_CIRCLE_QUESTION),
        )

        # System prompt (collapsible)
        if imgui.collapsing_header("System Prompt"):
            imgui.set_next_item_width(-1)
            _, self._system_prompt[0] = imgui.input_text_multiline("##sys_prompt", self._system_prompt[0], size=(0, 80))

        imgui.spacing()
        imgui.text_disabled("Chat Model")
        current_model = cfg.get(cfg.chat_model)
        model_values = [value for _, value in self._model_options]
        model_labels = [label for label, _ in self._model_options]
        model_idx = model_values.index(current_model) if current_model in model_values else 0
        imgui.set_next_item_width(220)
        changed, new_idx = imgui.combo("##chat_model", model_idx, model_labels)
        if changed:
            next_model = model_values[new_idx]
            if next_model != current_model:
                cfg.set(cfg.chat_model, next_model)
                self._unload_chat_model()
        imgui.same_line()
        if imgui.button("Load Model##chat"):
            self._load_chat_model()

        imgui.spacing()
        imgui.text_disabled("Post-Processing")
        changed, val = imgui.checkbox("RVC##chat", cfg.get(cfg.rvc_enabled))
        if changed:
            cfg.set(cfg.rvc_enabled, val)
        imgui.same_line()
        changed, val = imgui.checkbox("Super Resolution##chat", cfg.get(cfg.apbwe_enabled))
        if changed:
            cfg.set(cfg.apbwe_enabled, val)

        imgui.separator()
        imgui.text_disabled("Voice Input")
        self._recorder_panel.draw()
        imgui.separator()

        # Chat history (scrollable region)
        avail_h = imgui.get_content_region_avail().y - 90
        imgui.begin_child(
            "chat_history",
            size=imgui.ImVec2(0, max(avail_h, 100)),
            window_flags=imgui.WindowFlags_.horizontal_scrollbar,
        )
        try:
            for i, msg in enumerate(self._messages):
                role = msg.get("role", "user")
                text = msg.get("text", "")
                audio_path = msg.get("audio_path")

                if role == "user":
                    imgui.push_style_color(imgui.Col_.text, (0.6, 0.8, 1.0, 1.0))
                    imgui.text_wrapped(f"You: {text}")
                    imgui.pop_style_color()
                else:
                    imgui.push_style_color(imgui.Col_.text, (0.8, 1.0, 0.8, 1.0))
                    imgui.text_wrapped(f"AI: {text}")
                    imgui.pop_style_color()

                if audio_path and os.path.exists(audio_path):
                    imgui.same_line()
                    if imgui.small_button(f"Play##{i}"):
                        self._state.current_audio_file = audio_path
                        self._state.play_audio_requested = True

                imgui.spacing()
            if imgui.get_scroll_y() >= imgui.get_scroll_max_y() - 20:
                imgui.set_scroll_here_y(1.0)
        finally:
            imgui.end_child()

        imgui.separator()
        imgui.text_disabled("Message")
        avail_w = imgui.get_content_region_avail().x
        imgui.set_next_item_width(avail_w)
        _, self._input_text[0] = imgui.input_text_multiline("##chat_input", self._input_text[0], size=(avail_w, 50))
        if imgui.button("Send", size=(60, 0)):
            self._send_message()
        imgui.same_line()
        if imgui.button("Clear", size=(60, 0)):
            self._messages.clear()

        self._drawer.draw()

    def _draw_settings_drawer(self):
        from imgui_bundle import imgui

        imgui.text("Chat Settings")
        imgui.separator()
        changed, val = imgui.checkbox("RVC", cfg.get(cfg.rvc_enabled))
        if changed:
            cfg.set(cfg.rvc_enabled, val)
        changed, val = imgui.checkbox("Super Resolution", cfg.get(cfg.apbwe_enabled))
        if changed:
            cfg.set(cfg.apbwe_enabled, val)
        imgui.separator()
        imgui.text_wrapped("The shared audio player is hidden on this page. Generated replies still expose per-message play buttons when audio exists.")

    def _draw_help_drawer(self):
        from imgui_bundle import imgui

        imgui.text("Chat Help")
        imgui.separator()
        imgui.text_wrapped("Load a chat model, optionally adjust the system prompt, then send a message.")
        imgui.text_wrapped("RVC and Super Resolution are available for generated replies just like in the original widget.")

    def _load_chat_model(self, initial_message: str | None = None):
        if self._loading_model:
            return
        self._loading_model = True
        self._state.loading = True
        self._state.loading_message = "Loading chat model..."

        def _load():
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                model_name = cfg.get(cfg.chat_model)
                self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    trust_remote_code=True,
                )
            except Exception as e:
                self._state.error_queue.put(("Chat Model Error", str(e)))
            finally:
                self._loading_model = False
                if self._model is not None and self._tokenizer is not None and initial_message:
                    self._submit_user_message(initial_message, from_retry=True)
                else:
                    self._state.loading = False
                    self._state.loading_message = ""

        threading.Thread(target=_load, daemon=True).start()

    def _unload_chat_model(self):
        self._model = None
        self._tokenizer = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _send_message(self):
        text = self._input_text[0].strip()
        if not text:
            return
        self._input_text[0] = ""
        self._submit_user_message(text)

    def _submit_user_message(self, text: str, from_retry: bool = False):
        if not from_retry:
            self._messages.append({"role": "user", "text": text})

        if self._model is None or self._tokenizer is None:
            self._load_chat_model(initial_message=text)
            return

        self._state.loading = True
        self._state.loading_message = "Generating response..."

        def _generate():
            try:
                import torch
                from src.ui_imgui.pages.references import _resolve_reference_path
                from src.utils.audio_utils import combine_references
                from src.utils.file_utils import get_output_file_name
                from src.utils.inference_utils import generic_inference, get_default_reference_and_transcript, preprocess_text

                conversation = [{"role": "system", "content": self._system_prompt[0]}]
                for msg in self._messages:
                    conversation.append({"role": msg["role"], "content": msg["text"]})

                inputs = self._tokenizer.apply_chat_template(conversation, return_tensors="pt", add_generation_prompt=True)
                if torch.cuda.is_available():
                    inputs = inputs.to("cuda")

                with torch.no_grad():
                    outputs = self._model.generate(
                        inputs,
                        max_new_tokens=512,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                    )
                response_ids = outputs[0][inputs.shape[-1] :]
                response_text = self._tokenizer.decode(response_ids, skip_special_tokens=True).strip()

                audio_path = self._generate_tts(response_text)
                self._messages.append({"role": "assistant", "text": response_text, "audio_path": audio_path})
                if audio_path and cfg.get(cfg.auto_play):
                    self._state.current_audio_file = audio_path
                    self._state.play_audio_requested = True

            except Exception as e:
                self._messages.append({"role": "assistant", "text": f"[Error: {e}]"})
            finally:
                self._state.loading = False
                self._state.loading_message = ""

        threading.Thread(target=_generate, daemon=True).start()

    def _transcribe_recording(self):
        recording = self._state.recording_file
        if not recording or not os.path.exists(recording):
            return

        self._state.loading = True
        self._state.loading_message = "Transcribing chat audio..."

        def _run():
            try:
                if self._state.transcription_engine is None:
                    from src.utils.model_utils import load_whisper
                    load_whisper(self._state.callbacks)
                if self._state.transcription_engine is None:
                    raise RuntimeError("Whisper transcription engine failed to load.")

                transcript = self._state.transcription_engine.transcribe(recording)
                text = transcript.get("transcript", "").strip() if isinstance(transcript, dict) else str(transcript).strip()
                if not text:
                    raise RuntimeError("No transcript returned from the recording.")
                self._submit_user_message(text)
            except Exception as e:
                self._state.error_queue.put(("Chat Recording Error", str(e)))
                self._state.loading = False
                self._state.loading_message = ""

        threading.Thread(target=_run, daemon=True).start()

    def _generate_tts(self, text: str) -> Optional[str]:
        if self._state.tts_engine is None:
            return None
        try:
            from src.utils.audio_utils import combine_references
            from src.utils.inference_utils import generic_inference, get_default_reference_and_transcript, preprocess_text
            from src.ui_imgui.pages.references import _resolve_reference_path
            from src.utils.file_utils import get_output_file_name

            output_file = get_output_file_name(
                "chat_response",
                cfg.get(cfg.output_dir),
                self._state.tts_engine.model_name if self._state.tts_engine else "chat",
                self._state.engine_type.value if self._state.engine_type else "chat",
            )
            selected = [self._state.reference_audio[i] for i in sorted(self._state.selected_references) if i < len(self._state.reference_audio)]
            references = [p for p in (_resolve_reference_path(r) for r in selected) if p]
            dialogues = [r.get("dialogue", "") for r in selected if isinstance(r, dict)]
            combined = " ".join(d for d in dialogues if d)
            transcribe_state = {"transcript": combined} if combined else None
            if not references and self._state.tts_engine:
                ref_path, transcribe_state = get_default_reference_and_transcript(self._state.callbacks, self._state.tts_engine.model_name)
                if ref_path:
                    references = [ref_path]
            generic_inference(
                self._state.callbacks,
                output_file,
                preprocess_text(text),
                combine_references(references),
                None,
                transcribe_state,
                speaker=self._state.tts_engine.model_name if self._state.tts_engine else None,
                api=False,
            )
            return output_file
        except Exception:
            return None
