import os
from pydub import AudioSegment


def convert_mp3_to_wav(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".mp3"):
            print(f"converting {filename}")
            mp3_file_path = os.path.join(directory, filename)
            wav_file_path = os.path.join(directory, filename.replace(".mp3", ".wav"))

            # Load the MP3 file
            audio = AudioSegment.from_mp3(mp3_file_path)

            # Export the audio as WAV
            audio.export(wav_file_path, format="wav")

            # Remove the original MP3 file
            os.remove(mp3_file_path)

            print(f"Converted {mp3_file_path} to {wav_file_path} and removed the original MP3 file.")


if __name__ == "__main__":
    directory = "C:\\AI\\FallTalk\\samples"
    convert_mp3_to_wav(directory)