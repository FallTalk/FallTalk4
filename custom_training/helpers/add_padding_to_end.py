from pydub import AudioSegment
import os
import glob
import concurrent.futures


#directory = 'C:\\AI\\text-generation-webui\\extensions\\alltalk_tts\\finetune\\nate\\wavs'
directory = 'C:\\AI\\Datasets\\nate'

wav_files = glob.glob(os.path.join(directory, '*.wav'))

def process_wav_file(wav_file):
    audio = AudioSegment.from_wav(wav_file)
    # duration of silence in ms
    silence = AudioSegment.silent(duration=200)  # This is length of silence to pad beginning and end of segments. You can change this to whatever you want.
    new_audio = silence + audio + silence
    new_file_path = os.path.join(directory, os.path.basename(wav_file))
    print(f"Modified file {wav_file}")
    new_audio.export(new_file_path, format='wav')

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(process_wav_file, wav_files)

