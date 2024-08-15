import os.path

from datasets import load_dataset
import datasets.config

from custom_generator import CustomAudioDataset

datasets.config.IN_MEMORY_MAX_SIZE=3.2e+10

print("Loading Dataset")

dataset_folder = os.path.join("C:\\", "AI", "voicecraft", "datasets", "robotmrhandy")
print(f"{dataset_folder}")
dataset = datasets.load_dataset("custom_generator.py", name="custom_audio_dataset", data_dir=dataset_folder, num_proc=8)

print(f"Dataset Loaded {dataset}")

print(f'{dataset["train"][0]}')
print(f'{dataset["train"][0]["audio"]}')

print("Validation Loaded")

print(f'{dataset["validation"][0]}')

