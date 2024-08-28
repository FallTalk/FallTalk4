import json
import re

# Hardcoded reference dictionary for arcname
arcname_dict = {
    "dlcworkshop03.esm": "DLCworkshop03 - Voices_en.ba2",
    "dlcoast.esm": "DLCCoast - Voices_en.ba2",
    "fallout4.esm": "Fallout4 - Voices.ba2",
    "dlcrobot.esm": "DLCRobot - Voices_en.ba2",
    "dlcnukaworld.esm": "DLCNukaWorld - Voices_en.ba2"
}

# Read dialog.txt
dialog_data = {}
with open('dialog.txt', 'r') as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) == 7:
            form_id = parts[0].split(': ')[1]
            dialogue = parts[6]
            index = int(parts[5]) + 1
            dialog_data[form_id] = (dialogue, index)

# Function to read voice lines from a file
def read_voice_lines(filename):
    voice_lines = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split('\\')
            plugin = parts[3]
            name = parts[4]
            form_id = parts[5].split('_')[0]
            voice_lines.append((plugin, name, form_id))
    return voice_lines

# Read voicelines.txt and voicelines2.txt
voice_lines = read_voice_lines('MenuVoiceFiles.txt') + read_voice_lines('NonMenuVoiceFiles.txt')

# Create the JSON structure
json_data = []
for plugin, name, form_id in voice_lines:
    if form_id in dialog_data:
        dialogue, index = dialog_data[form_id]
        filename = f"{form_id}_{index}.fuz"
        arcname = arcname_dict.get(plugin.lower(), "Unknown")
        entry = {
            "name": name,
            "voicefiles": [
                {
                    "filename": filename,
                    "dialogue": dialogue,
                    "arcname": arcname.lower(),
                    "plugin": plugin
                }
            ]
        }
        # Check if the name already exists in json_data
        found = False
        for item in json_data:
            if item["name"] == name:
                item["voicefiles"].append(entry["voicefiles"][0])
                found = True
                break
        if not found:
            json_data.append(entry)

# Write the JSON output to a file
with open('output.json', 'w') as file:
    json.dump(json_data, file, indent=4)

print("JSON file has been created as output.json")