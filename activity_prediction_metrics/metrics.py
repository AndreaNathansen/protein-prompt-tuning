# Script used to generate activity predictions for the generated protein sequences
# See metrics.ipynb for an interactive version

import argparse
import os
from Bio import SeqIO
from quality_checks import main as quality_checks
from esm_likelihood import get_esm_likelihood_scores

def main(input_file, testset_file):
    dataset = list(SeqIO.parse(input_file, 'fasta'))
    testset = list(SeqIO.parse(testset_file, 'fasta'))

    # Perform the 'Quality Checks'
    dataset = quality_checks(dataset)

    os.makedirs('intermediate', exist_ok=True)  
    path = os.path.join("intermediate", f"{os.path.basename(input_file)}_qc.fasta")
    with open(path, 'w') as f:
        SeqIO.write(dataset, f, "fasta")

    # filter sequences with more than 1024 amino acids because ESM can't handle longer sequences
    testset = [seq for seq in testset if len(seq) <= 1024]
    
    # Calculate cutoff (10th percentile of ESM likelihood scores for the test set)
    testset_df = get_esm_likelihood_scores(testset)
    cutoff = testset_df['score'].quantile(q=0.9)

    testset_df.to_csv(os.path.join("intermediate", f"{os.path.basename(input_file)}_testset_scores.csv"))

    # Filter sequences with scores below cutoff
    df = get_esm_likelihood_scores(dataset)
    df.to_csv(os.path.join("intermediate", f"{os.path.basename(input_file)}_scores.csv"))
    filtered = df[df['score'] > cutoff]
    ids = list(filtered['id'])

    with open(os.path.join("intermediate", f"{os.path.basename(input_file)}_ids.txt"), 'w') as fp:
        fp.write(str(ids))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Protein Activity Metrics for Protein Sequences")
    parser.add_argument("--dataset", dest="dataset", help="path to the FASTA file of generated protein sequences to be evaluated", required=True)
    parser.add_argument("--testset", dest="testset", help="path to the FASTA file of the testset")
    args = parser.parse_args()
    results = main(args.dataset, args.testset)
