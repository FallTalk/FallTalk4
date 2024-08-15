import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

def downsample_wav(file_path):
    try:
        # Load the audio file using soundfile to determine the bit depth
        data, original_sr = sf.read(file_path)
        original_bit_depth = sf.info(file_path).subtype

        # Pad with 400 ms of silence if the flag is set
        silence_samples = int(0.4 * original_sr)
        if data.ndim == 1:  # Mono audio
            silence = np.zeros(silence_samples, dtype='int16')
        else:  # Stereo audio
            silence = np.zeros((silence_samples, data.shape[1]), dtype='int16')
        data = np.concatenate((silence, data, silence))

        # Save the converted audio back to the original file
        sf.write(file_path, data, original_sr, original_bit_depth)

        print(f"padded {file_path}")
    except BaseException as e:
        print(f"Unable to convert {file_path}", e)


def downsample_directory(directory):
    # List to hold the tasks
    tasks = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            input_file = os.path.join(directory, filename)
            tasks.append(input_file)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for task in tasks:
            executor.submit(downsample_wav, task)


directory = 'C:\\AI\\text-generation-webui\\extensions\\alltalk_tts\\finetune\\playervoicemale01-coast\\wavs'

# Downsample all WAV files in the directory
downsample_directory(directory)
