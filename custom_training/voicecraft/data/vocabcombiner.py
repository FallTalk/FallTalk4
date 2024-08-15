character = "robotmrhandy"

def load_vocab(file_path, encoding='utf-8'):
    vocab_set = set()
    with open(file_path, 'r', encoding=encoding) as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                vocab_set.add(parts[1])
    return vocab_set

def save_vocab(vocab_set, file_path, encoding='utf-8'):
    with open(file_path, 'w', encoding=encoding) as file:
        for i, vocab in enumerate(sorted(vocab_set)):
            file.write(f"{i} {vocab}\n")

def combine_vocab(file1, file2, output_file, encoding='utf-8'):
    vocab1 = load_vocab(file1, encoding)
    vocab2 = load_vocab(file2, encoding)

    # Combine vocab sets
    combined_vocab = vocab1.union(vocab2)

    # Save the combined vocab to the output file
    save_vocab(combined_vocab, output_file, encoding)

def print_unique_vocab(vocab_set1, vocab_set2):
    print(f"1 {vocab_set1}")
    print(f"2 {vocab_set2}")

    unique_vocab = vocab_set2 - vocab_set1

    print(f"2 {unique_vocab}")

    for idx, vocab in enumerate(sorted(unique_vocab)):
        print(f"{idx} {vocab}")


def find_unique_vocab(file1, file2):
    vocab_set1 = load_vocab(file1)
    vocab_set2 = load_vocab(file2)

    print_unique_vocab(vocab_set1, vocab_set2)


# Paths to the input and output files
file2_path = f'C:/AI/VoiceCraft/datasets/{character}_phn_enc_manifest_debug/xs/vocab.txt'
file1_path = 'C:/AI/VoiceCraft/vocab.txt'
output_file_path = f'C:/AI/VoiceCraft/datasets/{character}_phn_enc_manifest_debug/xs/combined_vocab.txt'

# Combine vocab files
combine_vocab(file1_path, file2_path, output_file_path)
find_unique_vocab(file1_path, file2_path)