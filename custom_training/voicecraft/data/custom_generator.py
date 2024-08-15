import json
import os

import datasets
from datasets import GeneratorBasedBuilder, Audio

import logging

formatter = (
    "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
)
logging.basicConfig(format=formatter, level=logging.INFO)

class CustomAudioDataset(GeneratorBasedBuilder):
    logging.info(f"CustomAudioDataset")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="custom_audio_dataset",
            version=datasets.Version("1.0.0"),
            description="Custom audio dataset without audio field in metadata",
        ),
    ]

    def _info(self):
        logging.info(f"_info")
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "file_name": datasets.Value("string"),
                    "segment_id": datasets.Value("string"),
                    "speaker": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "begin_time": datasets.Value("float"),
                    "end_time": datasets.Value("float"),
                    "audio_id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "url": datasets.Value("string"),
                    "source": datasets.Value("int64"),
                    "category": datasets.Value("int64"),
                    "original_full_path": datasets.Value("string"),
                    "audio": Audio(sampling_rate=16_000),  # Adjust sampling rate as needed
                }
            ),
            homepage="http://example.com",
        )

    def _split_generators(self, dl_manager):
        logging.info(f"_split_generators {dl_manager}")

        train_folder = os.path.join(self.config.data_dir, "data", "train")
        validation_folder = os.path.join(self.config.data_dir, "data", "validation")

        logging.info(f"validation_folder {train_folder}")
        logging.info(f"train_folder {validation_folder}")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"folder": train_folder},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"folder": validation_folder},
            ),
        ]

    def _generate_examples(self, folder):
        logging.info(f"_generate_examples {folder}")
        metadata_file = os.path.join(self.config.data_dir, "metadata.jsonl")
        logging.info(f"metadata_file {metadata_file}")
        logging.info(f"self.config.data_dir {self.config.data_dir}")


        # Load metadata and filter examples based on folder prefix
        with open(metadata_file, "r") as f:
            for line in f:
                example = json.loads(line.strip())
                file_name = example["file_name"]

                if folder == os.path.join(self.config.data_dir, "data", "train"):
                    if "train" in file_name:
                        yield example["segment_id"], example
                elif folder == os.path.join(self.config.data_dir, "data", "validation"):
                    if "validation" in file_name:
                        yield example["segment_id"], example
                else:
                    raise ValueError(f"Invalid folder: {folder}")
