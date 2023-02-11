import argparse
from pathlib import Path
from Bio import SeqIO
from sklearn.model_selection import train_test_split
import numpy as np

parser = argparse.ArgumentParser(prog="Prompt Tuning")
parser.add_argument("--dataset-path", dest="dataset_path", help="path to the Fasta file", required=True)
args = parser.parse_args()

random_seed = 123456

all_seqs = list(SeqIO.parse(args.dataset_path, "fasta"))
train_and_val, test = train_test_split(all_seqs, test_size=0.1, shuffle=True, random_state=random_seed)
train, val = train_test_split(train_and_val, test_size=1/9, shuffle=True, random_state=random_seed)

dataset_path = Path(args.dataset_path)
train_dataset_filename = dataset_path.stem + '_train' + dataset_path.suffix
val_dataset_filename = dataset_path.stem + '_val' + dataset_path.suffix
test_dataset_filename = dataset_path.stem + '_test' + dataset_path.suffix

with open(dataset_path.parent / train_dataset_filename, 'w') as f:
    SeqIO.write(train, f, "fasta")

with open(dataset_path.parent / val_dataset_filename, 'w') as f:
    SeqIO.write(val, f, "fasta")

with open(dataset_path.parent / test_dataset_filename, 'w') as f:
    SeqIO.write(test, f, "fasta")
