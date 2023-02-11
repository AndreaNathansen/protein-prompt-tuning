from Bio import SeqIO
import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(prog="Check sequence occurence in (expected) superset")
parser.add_argument("-i", "--input-file", dest="input_file", help="path to the Fasta file that contains the subset dataset", required=True)
parser.add_argument("-l", "--larger-file", dest="larger_file", help="path to the Fasta file that contains the superset dataset", required=True)
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
