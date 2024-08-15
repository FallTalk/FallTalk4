import gc
import json
import os
import random
import shutil

import jsonlines
import librosa
import soundfile as sf
import torch
import whisperx
from pydub import AudioSegment
from whisperx import load_model

character = 'robotmrhandy'
# Create dataset folder if it doesn't exist

dataset_folder = os.path.join("C:\\", "AI", "VoiceCraft", "datasets", character)
data_folder = os.path.join(dataset_folder, 'data')
segment_folder = os.path.join(data_folder, 'train')
validation_folder = os.path.join(data_folder, 'validation')
json_folder = os.path.join(dataset_folder, 'json')
wavs_folder = os.path.join(os.path.dirname("E:\\"), "AI", "datasets", "fallout4.esm", character)

os.makedirs(dataset_folder, exist_ok=True)
os.makedirs(segment_folder, exist_ok=True)
os.makedirs(json_folder, exist_ok=True)
os.makedirs(validation_folder, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("distil-large-v3", device=device, language="en")


def clean_folder(folder):
    # Clean validation folder before use
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")


clean_folder(segment_folder)
clean_folder(validation_folder)
clean_folder(json_folder)


def format_array(array):
    return repr(array).replace('\n', '')


def extract_segments(audio_path, timestamps):
    audio = AudioSegment.from_file(audio_path)
    filename = os.path.splitext(os.path.basename(audio_path))[0]  # Extract filename without extension
    segments = []
    for i, timestamp in enumerate(timestamps):
        begin_time = timestamp['start']
        if i < len(timestamps) - 1:
            end_time = timestamps[i + 1]['start']
        else:
            end_time = timestamps[i]['end']

        segment_id = f'{filename}_{i + 1}'  # Include filename in segment ID
        segment_audio = audio[int(begin_time * 1000):int(end_time * 1000)]
        segment_audio_path = os.path.join(segment_folder, f'{segment_id}.wav')

        # Randomly decide whether to save in train or validation folder
        if random.random() < 0.3:  # 30% chance to save in validation folder
            segment_audio_path = os.path.join(validation_folder, f'{segment_id}.wav')

        # Downsample to 16000 Hz
        y, sr = librosa.load(segment_audio.export(format='wav'), sr=16000)
        sf.write(segment_audio_path, y, sr)

        formatted_array = y.tolist()  # Convert NumPy array to list

        relative_path = segment_audio_path.replace(dataset_folder, '').lstrip('/').lstrip('\\')

        segments.append({
            'segment_id': segment_id,
            'speaker': 'N/A',
            'text': timestamp['text'],
            'begin_time': begin_time,
            'end_time': end_time,
            'file_name': relative_path,
            'audio': {
                'path': segment_audio_path,
                'array': formatted_array,
                'sampling_rate': int(sr)
            }
        }
        )

    return segments


def process_wav_files(folder_path):
    wav_files = os.listdir(folder_path)
    metadata_list = []  # Initialize an empty list to accumulate metadata
    total = len(wav_files)
    count = 1
    for filename in wav_files[:10]:  # Process only first 10 files for testing
        if filename.endswith(".wav"):
            try:
                audio_path = os.path.join(folder_path, filename)
                audio_id = os.path.splitext(filename)[0]
                print(f"Processing {audio_id} Remaining: {total - count}")
                count = count + 1
                audio_metadata = {
                    'audio_id': audio_id,
                    'title': f'Title of {audio_id}',
                    'url': f'http://example.com/{audio_id}',
                    'source': 1,  # Example source (podcast)
                    'category': 0,  # Example category
                    'original_full_path': audio_path
                }

                # Transcribe using WhisperX
                audio = whisperx.load_audio(audio_path)
                transcription_result = model.transcribe(audio)
                timestamps = transcription_result['segments']

                del audio
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Extract audio segments based on timestamps
                segments_info = extract_segments(audio_path, timestamps)

                # Append metadata for each segment to the list
                for segment_info in segments_info:
                    segment_data = {
                        'file_name': segment_info['file_name'],  # Use the path stored in segment_info
                        'segment_id': segment_info['segment_id'],
                        'speaker': segment_info['speaker'],
                        'text': segment_info['text'],
                        'begin_time': segment_info['begin_time'],
                        'end_time': segment_info['end_time'],
                        'audio': segment_info['audio'],
                        'audio_id': audio_metadata['audio_id'],
                        'title': audio_metadata['title'],
                        'url': audio_metadata['url'],
                        'source': audio_metadata['source'],
                        'category': audio_metadata['category'],
                        'original_full_path': audio_metadata['original_full_path']
                    }

                    json_file_path = os.path.join(json_folder, f'{segment_info["segment_id"]}.json')
                    with open(json_file_path, 'w') as json_file:
                        json.dump(segment_data, json_file, indent=2)

                    metadata_list.append(json_file_path)  # Append JSON file path for combining later

                    #metadata_list.append(segment_data)

            except Exception as e:
                print(f"Unable to process {filename}: ", e)


    # metadata_file_path = os.path.join(dataset_folder, 'metadata.jsonl')
    # with jsonlines.open(metadata_file_path, mode='w') as writer:
    #     writer.write_all(metadata_list)

    metadata_file_path = os.path.join(dataset_folder, 'metadata.jsonl')
    with jsonlines.open(metadata_file_path, mode='w') as writer:
        for json_file in metadata_list:
            with open(json_file, 'r') as f:
                s_data = json.load(f)
                writer.write(s_data)


# Process .wav files from the 'wavs' folder
import datasets.config

datasets.config.IN_MEMORY_MAX_SIZE=3.2e+10

print(f"Starting... {wavs_folder} to {dataset_folder}")
process_wav_files(wavs_folder)
print(f"Complete")
