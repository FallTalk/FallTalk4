import csv
import random

# Function to read a CSV file
def read_csv(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        data = list(reader)
    return header, data

# Function to write data to a CSV file
def write_to_csv(filename, header, data):
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(data)

# Read the existing train and eval CSV files
header_train, train_data = read_csv('C:\\AI\\text-generation-webui\\extensions\\alltalk_tts\\finetune\\nate\\metadata_train.csv')
header_eval, eval_data = read_csv('C:\\AI\\text-generation-webui\\extensions\\alltalk_tts\\finetune\\nate\\metadata_eval.csv')

# Combine the data
combined_data = train_data + eval_data

# Shuffle the combined data
random.seed(65)
random.shuffle(combined_data)

# Calculate the new split index
split_index = int(len(combined_data) * 0.80)

# Split the data
new_train_data = combined_data[:split_index]
new_eval_data = combined_data[split_index:]

# Write the new train and eval data to separate CSV files
write_to_csv('C:\\AI\\text-generation-webui\\extensions\\alltalk_tts\\finetune\\nate\\metadata_train.csv', header_train, new_train_data)
write_to_csv('C:\\AI\\text-generation-webui\\extensions\\alltalk_tts\\finetune\\nate\\metadata_eval.csv', header_train, new_eval_data)
write_to_csv('C:\\AI\\text-generation-webui\\extensions\\alltalk_tts\\finetune\\nate\\output.txt', header_train, combined_data)
write_to_csv('C:\\AI\\text-generation-webui\\extensions\\alltalk_tts\\finetune\\nate\\metadata.csv', header_train, combined_data)


print("Files have been split and saved as 'metadata_train.csv' and 'metadata_eval.csv'.")