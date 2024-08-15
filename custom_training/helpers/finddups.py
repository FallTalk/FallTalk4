def read_second_column(file_path):
    """Read the second column from a file."""
    second_column_values = set()
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) > 1:
                second_column_values.add(parts[1])
    return second_column_values


def find_duplicates(file1_path, file2_path):
    """Find duplicates in the second column between two files."""
    set1 = read_second_column(file1_path)
    set2 = read_second_column(file2_path)

    duplicates = set1.intersection(set2)
    return duplicates


def main():
    file1_path = "C:/AI/VoiceCraft/manifest/test_train.txt"
    file2_path = "C:/AI/VoiceCraft/train.txt"

    duplicates = find_duplicates(file1_path, file2_path)

    if duplicates:
        print("Duplicates found in the second column:")
        for duplicate in duplicates:
            print(duplicate)
    else:
        print("No duplicates found in the second column.")


if __name__ == "__main__":
    main()