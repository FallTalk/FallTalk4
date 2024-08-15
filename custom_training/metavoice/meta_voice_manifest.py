import os
import shutil
import csv

# Define input and output directories
input_directory = 'C:\\AI\\text-generation-webui\\extensions\\alltalk_tts\\finetune\\nate'
output_directory = 'C:\\AI\\metavoice-src\\datasets'
wavs_folder = 'C:\\AI\\text-generation-webui\\extensions\\alltalk_tts\\finetune\\nate\\wavs'
data_folder = 'C:\\AI\\metavoice-src\\data'

# Ensure the data and output directories exist
os.makedirs(data_folder, exist_ok=True)
os.makedirs(output_directory, exist_ok=True)

# Process each CSV file in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.csv'):
        input_csv_path = os.path.join(input_directory, filename)
        output_csv_path = os.path.join(output_directory, filename)

        # Read the input CSV and prepare the output data
        output_data = []
        with open(input_csv_path, mode='r', newline='', encoding='utf-8') as input_file:
            reader = csv.reader(input_file, delimiter='|')
            next(reader)  # Skip the header row
            for row in reader:
                audio_file, text, _ = row

                # Create new paths for audio and text files
                new_audio_file = os.path.join(data_folder, os.path.basename(audio_file))
                new_text_file = os.path.join(data_folder, os.path.splitext(os.path.basename(audio_file))[0] + '.txt')

                # Copy the audio file
                shutil.copy(os.path.join(wavs_folder, os.path.basename(audio_file)), new_audio_file)

                # Create the text file
                with open(new_text_file, mode='w', encoding='utf-8') as text_file:
                    text_file.write(text)

                # Append the new paths to the output data
                output_data.append([new_audio_file, new_text_file])

        # Write the output CSV
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as output_file:
            writer = csv.writer(output_file, delimiter='|')
            writer.writerow(['audio_files', 'captions'])
            writer.writerows(output_data)

        print(f"Processed {input_csv_path} and wrote output to {output_csv_path}")

print("Processing complete.")