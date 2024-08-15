import json
import json_repair

def main():
    input_file = 'fallout4.json'
    output_file = 'fallout4_fixed.json'

    fd = open(input_file, 'r')

    decoded_object = json_repair.repair_json(json_fd=fd, skip_json_loads=True, logging=True, return_objects=True)

    json_string = json.dumps(decoded_object)

    # Write the fixed JSON text to a new file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(json_string)

    print(f"Fixed JSON has been written to {output_file}")

if __name__ == "__main__":
    main()