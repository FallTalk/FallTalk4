import os
import soundfile as sf
import librosa
from concurrent.futures import ThreadPoolExecutor

def convert_to_mono(file_path):
    try:
        # Load the audio file using soundfile to determine the bit depth
        data, sample_rate = sf.read(file_path)

        # Check if the audio is stereo
        if data.ndim == 2 and data.shape[1] == 2:
            # Convert to mono by averaging the two channels
            data = librosa.to_mono(data.T)

            # Save the converted audio back to the original file
            sf.write(file_path, data, sample_rate, 'PCM_16')
            print(f"Converted {file_path} to mono")

    except Exception as e:
        print(f"Unable to convert {file_path} to mono: {e}")

def convert_directory_to_mono(directory):
    # List to hold the tasks
    tasks = []

    # Iterate over all files in the directory and subdirectories
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".wav"):
                input_file = os.path.join(root, filename)
                tasks.append(input_file)

    with ThreadPoolExecutor(max_workers=8) as executor:
        for task in tasks:
            executor.submit(convert_to_mono, task)

# Specify the root directory containing the WAV files
directory = 'C:\\AI\\Datasets\\dlccoast.esm'

# Convert all stereo WAV files in the directory and subdirectories to mono
convert_directory_to_mono(directory)