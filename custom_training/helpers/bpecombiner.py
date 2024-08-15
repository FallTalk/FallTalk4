import json
import os
from pathlib import Path


def combine_vocab_files(base_path, out_path):
    # Load the vocab file
    vocab_file_path = os.path.join(str(Path("C:\\AI\\text-generation-webui\\extensions\\alltalk_tts\\models\\xtts") / "xttsv2_2.0.3" / "vocab.json"))
    vocab_file = json.load(open(vocab_file_path, 'r', encoding='utf-8'))

    # Load the trained vocab file
    trained_file_path = os.path.join(str(out_path / "trained-vocab.json"))
    trained_file = json.load(open(trained_file_path, 'r', encoding='utf-8'))

    # Extract vocab arrays
    vocab_array = vocab_file.get('model', {}).get('vocab', {})
    vocab_array_trained = trained_file.get('model', {}).get('vocab', {})

    # Combine vocab arrays
    combined_vocab_array = {}
    for key, value in vocab_array.items():
        combined_vocab_array[key] = value

    next_index = int(len(vocab_array))

    # Add new entries from trained vocab if they do not already exist
    for key, value in vocab_array_trained.items():
        if key not in combined_vocab_array:
            combined_vocab_array[key] = next_index
            next_index += 1

    # Update the original vocab file with the combined vocab array
    vocab_file['model']['vocab'] = combined_vocab_array

    # Save the updated vocab file
    with open(str(out_path / "bpe_tokenizer-vocab.json"), 'w', encoding='utf-8') as file:
        json.dump(vocab_file, file, indent=4, ensure_ascii=False)


# Example usage
this_dir = Path("C:\\AI\\text-generation-webui\\extensions\\alltalk_tts\\finetune")
base_path = this_dir / "nate"
out_path = this_dir / "nate"

combine_vocab_files(base_path, out_path)