import torch
import os
import time
import h5py
import argparse
from torch.utils.data import DataLoader

from clm import GenPromptEmb
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--device", type=str, default="cuda")
    # parser.add_argument("--data_path", type=str, default="ETTh1")
    # parser.add_argument("--num_nodes", type=int, default=7)
    # parser.add_argument("--input_len", type=int, default=96)
    # parser.add_argument("--output_len", type=int, default=96)
    # parser.add_argument("--batch_size", type=int, default=1)
    # parser.add_argument("--d_model", type=int, default=768)
    # parser.add_argument("--l_layers", type=int, default=12)
    # parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--divide", type=str, default="train")
    # parser.add_argument("--num_workers", type=int, default=10)
    return parser.parse_args()

def save_embeddings(args):
    print(f'test_amex: {args.divide}')
    return

if __name__ == "__main__":
    args = parse_args()
    t1 = time.time()
    save_embeddings(args)
    t2 = time.time()
    print(f"Total time spent: {(t2 - t1)/60:.4f} minutes")


