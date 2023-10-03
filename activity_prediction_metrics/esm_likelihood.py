from protein_gibbs_sampler.src.pgen.likelihood_esm import main as likelihood_esm
import os
import tempfile
import pandas as pd
from Bio import SeqIO


def get_esm_likelihood_scores(sequences):

    with tempfile.TemporaryDirectory() as tmpdirname:

        dataset_file = os.path.join(tmpdirname, 'data.fasta')
        output_file = os.path.join(tmpdirname, 'output.fasta')

        with open(dataset_file, '+tx') as input_handle:
            SeqIO.write(list(sequences), input_handle, 'fasta')

        with (open(dataset_file, 'r') as input_handle, 
              open(output_file, 'w') as output_handle):
            likelihood_esm(input_handle, output_handle, True, 'cpu', 'esm1v', 1, float('inf'), True, 'score', None)

        df = pd.read_csv(output_file)
    return df