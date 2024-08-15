import os
import soundfile as sf
import numpy as np
import librosa
from concurrent.futures import ThreadPoolExecutor

target_sr = 22500
pad_silence = False

def downsample_wav(file_path):
    try:
        # Load the audio file using soundfile to determine the bit depth
        data, original_sr = sf.read(file_path)
        original_bit_depth = sf.info(file_path).subtype

        # Resample to the target sample rate if the original sample rate is different
        if original_sr != target_sr:
            data = librosa.resample(data, orig_sr=original_sr, target_sr=target_sr)

        if data.ndim == 2 and data.shape[1] == 2:
            # Convert to mono by averaging the two channels
            data = librosa.to_mono(data.T)

        # Convert to 16-bit PCM if the original bit depth is not 16-bit
        if original_bit_depth != 'PCM_16':
            if original_bit_depth == 'PCM_32':
                data = (data / 2 ** 16).astype('int16')
            elif original_bit_depth == 'FLOAT':
                data = (data * 32767).astype('int16')
            else:
                raise ValueError(f"Unsupported bit depth: {original_bit_depth}")

        if data.dtype != np.int16:
            data = (data * 32767).astype(np.int16)

        # Pad with 400 ms of silence if the flag is set
        if pad_silence:
            silence_samples = int(0.4 * target_sr)
            if data.ndim == 1:  # Mono audio
                silence = np.zeros(silence_samples, dtype='int16')
            else:  # Stereo audio
                silence = np.zeros((silence_samples, data.shape[1]), dtype='int16')
            data = np.concatenate((data, silence))

        # Save the converted audio back to the original file
        sf.write(file_path, data, target_sr, 'PCM_16')

        print(f"down sampled {file_path}")
    except Exception as e:
        print(f"Unable to convert {file_path}: {e}")

def downsample_directory(directory):
    # List to hold the tasks
    tasks = []

    # Iterate over all files in the directory and subdirectories
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".wav"):
                input_file = os.path.join(root, filename)
                tasks.append(input_file)

    with ThreadPoolExecutor(max_workers=12) as executor:
        for task in tasks:
            executor.submit(downsample_wav, task)

# Specify the root directory containing the WAV files
directory = 'C:\\AI\\Datasets\\dlccoast.esm'

# Downsample all WAV files in the directory and subdirectories
downsample_directory(directory)