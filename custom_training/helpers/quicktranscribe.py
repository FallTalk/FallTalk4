import torch
from whisperx import load_model

device = "cpu"

if device == "cpu":
    compute_type = "int8"
else:
    compute_type = "float16"
model = load_model("distil-large-v3", device, language='en', compute_type=compute_type, asr_options={"suppress_numerals": True, "max_new_tokens": None, "clip_timestamps": None, "hallucination_silence_threshold": None})


audio = "D:\\datasets_40000\\fallout4.esm\\playervoicemale01\\000673df_1.wav"

resp = model.transcribe(audio)['segments']

transcript = " ".join([segment["text"] for segment in resp])

print(transcript)

print("done")