import struct

class StringDataObject:
    def __init__(self, string_id, address, relative_offset, absolute_offset, null_point, length, string_array):
        self.id = string_id
        self.address = address
        self.relative_offset = relative_offset
        self.absolute_offset = absolute_offset
        self.null_point = null_point
        self.length = length
        self.string_array = string_array

    def __repr__(self):
        return f"StringDataObject(id={self.id}, address={self.address}, relative_offset={self.relative_offset}, absolute_offset={self.absolute_offset}, null_point={self.null_point}, length={self.length}, string_array={self.string_array})"

def read_strings_file(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()

    # Parse the header
    num_entries, data_size = struct.unpack('<II', data[:8])
    header_size = 8 + num_entries * 8

    # Parse the directory entries
    entries = []
    for i in range(num_entries):
        offset = 8 + i * 8
        string_id, string_offset = struct.unpack('<II', data[offset:offset + 8])
        entries.append((string_id, string_offset))

    # Parse the string data
    string_data_objects = []
    for string_id, string_offset in entries:
        if file_path.endswith('.strings'):
            # Null-terminated C-style string
            end_offset = data.find(b'\x00', string_offset + header_size)
            string_data = data[string_offset + header_size:end_offset]
            length = end_offset - (string_offset + header_size)
            null_point = end_offset
        else:
            # .dlstrings, .ilstrings
            length = struct.unpack('<I', data[string_offset + header_size:string_offset + header_size + 4])[0]
            string_data = data[string_offset + header_size + 4:string_offset + header_size + 4 + length - 1]
            null_point = string_offset + header_size + 4 + length - 1

        try:
            string_data = string_data.decode('utf-8')
        except UnicodeDecodeError:
            string_data = string_data.decode('windows-1252')

        string_array = [string_data]
        absolute_offset = string_offset + header_size

        string_data_object = StringDataObject(
            string_id=string_id,
            address=string_offset,
            relative_offset=string_offset,
            absolute_offset=absolute_offset,
            null_point=null_point,
            length=length,
            string_array=string_array
        )
        string_data_objects.append(string_data_object)

    return string_data_objects

# Example usage
strings_objects = read_strings_file("M:\FO4\Strings\Fallout4_en.ilstrings")
# dlstrings_objects = read_strings_file('path/to/your/file.dlstrings')
# ilstrings_objects = read_strings_file('path/to/your/file.ilstrings')

hex_value = '0009FFDF'
decimal_value = int(hex_value, 16)

for obj in strings_objects:
    print(obj.id)





# Example usage
# read_strings_file("M:\FO4\Strings\Fallout4_en.STRINGS")
# read_strings_file("M:\FO4\Strings\Fallout4_en.dlstrings")
