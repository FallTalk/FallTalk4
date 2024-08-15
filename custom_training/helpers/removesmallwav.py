import csv
import os

import librosa


def get_wav_duration(file_path):
    try:
        with open(file_path, 'rb') as f:
            y, sr = librosa.load(f, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            return duration
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def process_csv_file(csv_path, wav_dir):
    rows_to_keep = []
    with open(csv_path, mode='r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter='|')
        header = next(reader)  # Skip the header row
        rows_to_keep.append(header)

        for row in reader:
            wav_file_path = os.path.join(wav_dir, row[0])
            if os.path.exists(wav_file_path):
                duration = get_wav_duration(wav_file_path)
                if duration >= 0.7:
                    rows_to_keep.append(row)
            else:
                print(f"Warning: WAV file not found - {wav_file_path}")

    # Write the updated CSV file
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter='|')
        writer.writerows(rows_to_keep)


def process_directory(csv_dir, wav_dir):
    # for csv_file_name in os.listdir(csv_dir):
    #     if csv_file_name.endswith('.csv'):
    #         csv_file_path = os.path.join(csv_dir, csv_file_name)
    #         process_csv_file(csv_file_path, wav_dir)

    # Delete WAV files that are less than 0.7 seconds
    count = 0;
    for wav_file_name in os.listdir(wav_dir):
        if wav_file_name.endswith('.wav'):
            wav_file_path = os.path.join(wav_dir, wav_file_name)
            duration = get_wav_duration(wav_file_path)
            if duration < 0.7:
                print(f"removing {wav_file_path}")
                count += 1
                os.remove(wav_file_path)
            else:
                pass
                #print(f"keeping {wav_file_path} dur={duration}")
    print(f"total checked {count}")

#
csv_dir = 'C:\\AI\\text-generation-webui\\extensions\\alltalk_tts\\finetune\\nate'
# wav_dir = 'C:\\AI\\text-generation-webui\\extensions\\alltalk_tts\\finetune\\nate\\wavs'
wav_dir = 'C:\\AI\\datasets\\nate'

process_directory(csv_dir, wav_dir)