import os
import subprocess
import logging
from pathlib import Path
import pickle
import argparse
import torch
import torch.distributed as dist
from config import MyParser
from steps import trainer

def setup_logging():
    formatter = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

def main():
    setup_logging()

    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['WORLD_SIZE'] = '1'

    # Define variables
    dataset = 'gigaspeech'
    character = 'nate'
    logs_dir = Path(f'./logs/{dataset}')
    logs_dir.mkdir(parents=True, exist_ok=True)

    exp_root = "C:\\AI\\VoiceCraft\\training"
    exp_name = "830M_TTSEnhanced"
    dataset_dir = "C:\\AI\\VoiceCraft\\datasets\\nate_phn_enc_manifest_debug\\xs"
    encodec_codes_folder_name = "encodec_16khz_4codebooks"
    load_model_from = "C:\\AI\\VoiceCraft\\830M_TTSEnhanced.pth"

    # Construct the command for torchrun
    torchrun_command = [
        'torchrun',
        '--nnodes=1',
        '--rdzv-backend=c10d',
        '--rdzv-endpoint=localhost:41977',
        '--nproc_per_node=1',
        'main.py',
        f'--load_model_from={load_model_from}',
        '--reduced_eog=1',
        '--drop_long=1',
        '--eos=2051',
        '--n_special=4',
        '--pad_x=0',
        '--codebook_weight=[3,1,1,1]',
        '--encodec_sr=50',
        '--num_steps=23970',
        '--lr=0.0001',
        '--warmup_fraction=0.1',
        '--optimizer_name=AdamW',
        '--d_model=2048',
        '--audio_embedding_dim=2048',
        '--nhead=16',
        '--num_decoder_layers=16',
        '--max_num_tokens=20000',
        '--gradient_accumulation_steps=32',
        '--val_max_num_tokens=6000',
        '--num_buckets=6',
        '--audio_max_length=20',
        '--audio_min_length=0',
        '--text_max_length=400',
        '--text_min_length=10',
        '--mask_len_min=1',
        '--mask_len_max=600',
        '--tb_write_every_n_steps=100',
        '--print_every_n_steps=100',
        '--val_every_n_steps=2397',
        '--text_vocab_size=100',
        '--text_pad_token=100',
        '--phn_folder_name=phonemes',
        '--manifest_name=manifest',
        f'--encodec_folder_name={encodec_codes_folder_name}',
        '--audio_vocab_size=2048',
        '--empty_token=2048',
        '--eog=2049',
        '--audio_pad_token=2050',
        '--n_codebooks=4',
        '--max_n_spans=3',
        '--shuffle_mask_embedding=0',
        '--mask_sample_dist=poisson1',
        '--max_mask_portion=0.9',
        '--min_gap=5',
        '--num_workers=8',
        '--dynamic_batching=0',
        '--batch_size=4',
        f'--dataset={dataset}',
        f'--exp_dir={exp_root}/{character}/{exp_name}',
        f'--dataset_dir={dataset_dir}'
    ]

    # Execute the command
    subprocess.run(torchrun_command)

if __name__ == "__main__":
    main()