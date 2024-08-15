import json
import json_repair

def main():
    input_file = 'fallout4_squashed.json'
    output_file = 'fallout4_squashed2.json'

    fd = open(input_file, 'r')

    decoded_object = json.load(fd)

    print(f"object {decoded_object}")

    combined_list = []
    names_seen = set()

    for plugin in decoded_object['plugins']:
        for voice in plugin['voicetypes']:
            for voicefile in voice['voicefiles']:
                voicefile['arcname'] = plugin['arcname']
                voicefile['plugin'] = plugin['name']
            if voice['name'] in names_seen:
                for combined_voice_type in combined_list:
                    if combined_voice_type['name'] == voice['name']:
                        combined_voice_type['voicefiles'].extend(voice['voicefiles'])
                        break
            else:
                names_seen.add(voice['name'])
                combined_list.append(voice)

    json_string = json.dumps(combined_list)

    # Write the fixed JSON text to a new file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(json_string)

    json_string = json.dumps(combined_list[0])

    with open('fallout4_sample.json', 'w', encoding='utf-8') as file:
        file.write(json_string)

    print(f"Fixed JSON has been written to {output_file}")

if __name__ == "__main__":
    main()