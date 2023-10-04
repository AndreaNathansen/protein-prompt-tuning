"""
Auxilary script used for radomly sampling 10 sequences of each of the families 
PF18369, PF04680, PF17988, PF12325, PF03938, PF17724, PF10696, PF11968, PF04153, PF06173, 
PF12378, PF04420, PF10841, PF06917, PF03492, PF06905, PF15340, PF17055 and PF05318.
The resulting dataset was used as a negative test set to detect false positives in our evaluation methods.
"""

import argparse
import random
from Bio import SeqIO

random.seed(42)

def _parse_args():
    parser = argparse.ArgumentParser(prog="Create a dataset consisting of 10 randomly sampled sequences out of each input fasta file")
    parser.add_argument("-i", "--input-files", dest="input_file", help="paths of the input fasta files", nargs='+')
    parser.add_argument("-o", "--output-file", dest="output_file", help="path and filename of the output fasta file", default="merged.fasta")
    return parser.parse_args()

def read_datasets(files):
    output = []
    for file in files:
        output.append(list(SeqIO.parse(file, "fasta")))
    return output

def sample_datasets(datasets):
    output = []
    for dataset in datasets:
        output.extend(random.sample(dataset, 10))
    random.shuffle(output)
    return output

if __name__ == "__main__":
    args = _parse_args()
    datasets = read_datasets(args.input_file)
    sampled = sample_datasets(datasets)
    SeqIO.write(sampled, args.output_file, "fasta")
