"""
Script for checking how many sequences from a smaller fasta file are present in another larger fasta file. The smaller
fasta file has to fit in memory.
"""

import argparse
from pathlib import Path

import numpy as np
from Bio import SeqIO

parser = argparse.ArgumentParser(prog="Check sequence occurence in (expected) superset")
parser.add_argument("-i", "--input-file", dest="input_file", help="path to the Fasta file that contains the smaller dataset", required=True)
parser.add_argument("-l", "--larger-file", dest="larger_file", help="path to the Fasta file that contains the larger dataset", required=True)
args = parser.parse_args()

subset_seq_ids = np.array([seq.id.split('|')[0] for seq in SeqIO.parse(args.input_file, "fasta")])
subset_seq_ids_mask = np.ones(len(subset_seq_ids), dtype=bool)

with open(args.larger_file) as f:
    for seq in SeqIO.parse(f, 'fasta'):
        seq_id = seq.id.split('_')[-1]
        subset_seq_ids_mask[subset_seq_ids == seq_id] = False

print(subset_seq_ids_mask.sum())
print(subset_seq_ids[subset_seq_ids_mask])
with open(Path(args.input_file).parent / "unmatched_sequences.txt", 'w') as f:
    f.write(subset_seq_ids[subset_seq_ids_mask])
