import csv
import os


def process_csv(input_csv, output_csv, captions_dir):
    # Ensure the captions directory exists
    os.makedirs(captions_dir, exist_ok=True)

    with open(input_csv, mode='r', encoding='utf-8') as infile, \
            open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile, delimiter='|')
        fieldnames = ['audio_files', 'captions']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='|')

        writer.writeheader()

        for row in reader:
            audio_file = row['audio_file']
            text = row['text']

            # Create the caption file name
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            caption_file = os.path.join(captions_dir, f"{base_name}.txt")

            # Write the text to the caption file
            with open(caption_file, 'w', encoding='utf-8') as caption_f:
                caption_f.write(text)

            # Write the new CSV row
            writer.writerow({
                'audio_files': audio_file,
                'captions': caption_file
            })


if __name__ == "__main__":
    input_csv_path = 'C:\\AI\\metavoice-src\\metadata_eval.csv'  # Change to your input CSV file path
    output_csv_path = 'C:\\AI\\metavoice-src\\dataset_val.csv'  # Change to your desired output CSV file path
    captions_directory = 'C:\\AI\\metavoice-src\\captions'  # Change to your desired captions directory

    process_csv(input_csv_path, output_csv_path, captions_directory)