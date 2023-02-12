import numpy as np
from Bio.Seq import Seq

replacement_B = ['D', 'N']
replacement_Z = ['E', 'Q']
replacement_J = ['I', 'L']

def remove_X(dataset):
    """
    Data preparation as in Hesslow et al. RITA: a Study on Scaling Up Generative Protein Sequence Models.
    Remove all sequences with an X.
    """
    return [record for record in dataset if not 'X' in record.seq]
    


def replace_amino_acids(random_seed, dataset):
    """
    Data preparation as in Hesslow et al. RITA: a Study on Scaling Up Generative Protein Sequence Models.
    Randomly map amino acids B to (D,N), Z to (E, Q), J to (I, L).
    However, we do not add each sequence's reverse to the dataset as done by Hesslow et al.
    """
    prepared_sequences = []
    rng = np.random.default_rng(seed=random_seed)
    prepared_sequences = []
    for record in dataset:
        record_seq_list = np.array(list(record.seq))

        idcs_B = np.where(record_seq_list == 'B')[0]
        replacements_B = rng.choice(replacement_B, size=len(idcs_B))
        record_seq_list[idcs_B] = replacements_B

        idcs_Z = np.where(record_seq_list == 'Z')[0]
        replacements_Z = rng.choice(replacement_Z, size=len(idcs_Z))
        record_seq_list[idcs_Z] = replacements_Z

        idcs_J = np.where(record_seq_list == 'J')[0]
        replacements_J = rng.choice(replacement_J, size=len(idcs_J))
        record_seq_list[idcs_J] = replacements_J

        record_seq = "".join(record_seq_list)
        
        record.seq = Seq(record_seq)
        prepared_sequences.append(record)
    return prepared_sequences
