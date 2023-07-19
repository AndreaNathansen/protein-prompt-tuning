"""
Script that we used to measure how many generated sequences are classified by ProtCNN
as belonging to the target Pfam family.
Taken (and adapted) from https://github.com/google-research/google-research/blob/master/using_dl_to_annotate_protein_universe/Using_Deep_Learning_to_Annotate_the_Protein_Universe.ipynb
(the official implementation of Bileschi et al. Using deep learning to annotate the protein universe.)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from Bio import SeqIO
# Suppress noisy log messages.
from tensorflow.python.util import deprecation
from tqdm import tqdm

deprecation._PRINT_DEPRECATION_WARNINGS = False

parser = argparse.ArgumentParser(prog="Prompt Tuning")
parser.add_argument("--dataset", dest="dataset", help="path to the JSON config file", required=True)
parser.add_argument("--window-size", dest="window_size", type=int, help="Size of the sliding window for domain calling. If omitted, every possible window size (1 to the sequence's length) is used")
parser.add_argument("--window-stride", dest="window_stride", type=int, help="Stride of the sliding window for domain calling", required=True)
parser.add_argument("--family", dest="family", help="Pfam name of the target family", required=True)
parser.add_argument("--probability-threshold", dest="probability_threshold", type=float, help="Minimum probability for a called domain to be considered", required=False)
args = parser.parse_args()

AMINO_ACID_VOCABULARY = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y'
]
def residues_to_one_hot(amino_acid_residues):
  """Given a sequence of amino acids, return one hot array.

  Supports ambiguous amino acid characters B, Z, and X by distributing evenly
  over possible values, e.g. an 'X' gets mapped to [.05, .05, ... , .05].

  Supports rare amino acids by appropriately substituting. See
  normalize_sequence_to_blosum_characters for more information.

  Supports gaps and pads with the '.' and '-' characters; which are mapped to
  the zero vector.

  Args:
    amino_acid_residues: string. consisting of characters from
      AMINO_ACID_VOCABULARY

  Returns:
    A numpy array of shape (len(amino_acid_residues),
     len(AMINO_ACID_VOCABULARY)).

  Raises:
    ValueError: if sparse_amino_acid has a character not in the vocabulary + X.
  """
  to_return = []
  normalized_residues = amino_acid_residues.replace('U', 'C').replace('O', 'X')
  for char in normalized_residues:
    if char in AMINO_ACID_VOCABULARY:
      to_append = np.zeros(len(AMINO_ACID_VOCABULARY))
      to_append[AMINO_ACID_VOCABULARY.index(char)] = 1.
      to_return.append(to_append)
    elif char == 'B':  # Asparagine or aspartic acid.
      to_append = np.zeros(len(AMINO_ACID_VOCABULARY))
      to_append[AMINO_ACID_VOCABULARY.index('D')] = .5
      to_append[AMINO_ACID_VOCABULARY.index('N')] = .5
      to_return.append(to_append)
    elif char == 'Z':  # Glutamine or glutamic acid.
      to_append = np.zeros(len(AMINO_ACID_VOCABULARY))
      to_append[AMINO_ACID_VOCABULARY.index('E')] = .5
      to_append[AMINO_ACID_VOCABULARY.index('Q')] = .5
      to_return.append(to_append)
    elif char == 'X':
      to_return.append(
          np.full(len(AMINO_ACID_VOCABULARY), 1. / len(AMINO_ACID_VOCABULARY)))
    elif char == _PFAM_GAP_CHARACTER:
      to_return.append(np.zeros(len(AMINO_ACID_VOCABULARY)))
    else:
      raise ValueError('Could not one-hot code character {}'.format(char))
  return np.array(to_return)

def _test_residues_to_one_hot():
  expected = np.zeros((3, 20))
  expected[0, 0] = 1.   # Amino acid A
  expected[1, 1] = 1.   # Amino acid C
  expected[2, :] = .05  # Amino acid X

  actual = residues_to_one_hot('ACX')
  np.testing.assert_allclose(actual, expected)
_test_residues_to_one_hot()

def pad_one_hot_sequence(sequence: np.ndarray,
                         target_length: int) -> np.ndarray:
  """Pads one hot sequence [seq_len, num_aas] in the seq_len dimension."""
  sequence_length = sequence.shape[0]
  pad_length = target_length - sequence_length
  if pad_length < 0:
    raise ValueError(
        'Cannot set a negative amount of padding. Sequence length was {}, target_length was {}.'
        .format(sequence_length, target_length))
  pad_values = [[0, pad_length], [0, 0]]
  return np.pad(sequence, pad_values, mode='constant')

def _test_pad_one_hot():
  input_one_hot = residues_to_one_hot('ACX')
  expected = np.array(input_one_hot.tolist() + np.zeros((4, 20)).tolist())
  actual = pad_one_hot_sequence(input_one_hot, 7)

  np.testing.assert_allclose(expected, actual)

assert len(tf.config.list_physical_devices('GPU')) > 0, "No GPU detected"
print("GPU:")
print(tf.config.list_physical_devices('GPU'))

_test_pad_one_hot()

if args.probability_threshold is not None:
  assert 0 <= args.probability_threshold <= 1, \
  f"Probabilities are in range between 0 and 1, but you set a probability threshold of {args.probability_threshold}"

sess = tf.Session()
graph = tf.Graph()

with graph.as_default():
  saved_models = [tf.saved_model.load(sess, ['serve'], 'trn-_cnn_random__random_sp_gpu-cnn_for_random_pfam-5356760')]
                  #tf.saved_model.load(sess, ['serve'], 'trn-_cnn_random__random_sp_gpu-cnn_for_random_pfam-5356766'),]
                  #tf.saved_model.load(sess, ['serve'], 'trn-_cnn_random__random_sp_gpu-cnn_for_random_pfam-5365208')]
assert not (len(saved_models) > 1 and args.probability_threshold is not None), \
"Currently not supporting setting a probability threshold for an ensemble, because then the shapes for aggregation would not match anymore"


class_confidence_signatures = [saved_model.signature_def['confidences'] for saved_model in saved_models]
class_confidence_signature_tensor_names = [class_confidence_signature.outputs['output'].name for class_confidence_signature in class_confidence_signatures]

sequence_input_tensor_names = [saved_model.signature_def['confidences'].inputs['sequence'].name for saved_model in saved_models]
sequence_lengths_input_tensor_names = [saved_model.signature_def['confidences'].inputs['sequence_length'].name for saved_model in saved_models]

def split_sequence_into_windows(seq, window_size = args.window_size):
  subseqs = []
  if window_size is None:
    # TODO: look up again, but I think in the ProtCNN paper they start at 50 minimum window size
    for i in range(min(50, len(seq) - 1),len(seq)):
      subseqs.extend(split_sequence_into_windows(seq, i))
  else:
    if len(seq) > window_size:
      for i in range(0, len(seq) - (window_size - args.window_stride), args.window_stride):
        # TODO: optimize this for flexible window sizes (otherwise some smaller windows towards
        # the end of the sequence get called multiple times)
        if i + window_size > len(seq):
          subseq  = seq[len(seq) - window_size : len(seq)]
        else:
          subseq = seq[i:i+window_size]
        subseqs.append(subseq)
    else:
      subseqs.append(seq)
  return subseqs

def predict_families_for_fasta_file(filename):
    dataset = list(SeqIO.parse(filename, "fasta"))
    results_df = pd.DataFrame(columns=["id", "is_family"], index=range(len(dataset)))
    bar = tqdm(total=len(dataset))

    # Load vocab
    with open('trained_model_pfam_32.0_vocab.json') as f:
        vocab = np.array(json.loads(f.read()))
    
    for j, record in enumerate(dataset):
        seq = str(record.seq)

        # handle edge case where the sequence has only one amino acid
        if len(seq) <= 1:
          results_df.iloc[j] = [record.id, False]
          continue

        subseqs = split_sequence_into_windows(seq)
        is_family = False
        batch_size = 64
        batch_bar = tqdm(total=np.ceil(len(subseqs) / batch_size))
        with graph.as_default():
            for i in range(0, len(subseqs), batch_size):
              batch_sequences = subseqs[i:i+batch_size]
              max_subseq_length = max([len(s) for s in batch_sequences])
              padded_one_hot_sequences = [pad_one_hot_sequence(residues_to_one_hot(seq), max_subseq_length) for seq in batch_sequences]
              predicted_families_all_models = []
              for i in range(len(saved_models)):
                confidences_by_class = sess.run(
                    class_confidence_signature_tensor_names[i],
                    {
                        # Note that this function accepts a batch of sequences which
                        # can speed up inference when running on many sequences.
                        sequence_input_tensor_names[i]: padded_one_hot_sequences,
                        sequence_lengths_input_tensor_names[i]: [len(seq) for seq in padded_one_hot_sequences],
                    }
                )
                
                protein_family_idcs = np.argmax(confidences_by_class, axis=1)

                if args.probability_threshold is not None:
                  max_confidences = np.max(confidences_by_class, axis=1)
                  protein_family_idcs = protein_family_idcs[max_confidences >= args.probability_threshold]

                predicted_families_for_model = vocab[protein_family_idcs]
                predicted_families_all_models.append(predicted_families_for_model)
              predicted_families_all_models = np.array(predicted_families_all_models)

              # check if all models agree on the family in at least one element of the batch
              agreements_on_desired_family = (predicted_families_all_models == args.family).sum(axis=0)
              if len(saved_models) in agreements_on_desired_family:
                  is_family = True
        batch_bar.update(1)
        results_df.iloc[j] = [record.id, is_family]
        bar.update(1)
    return results_df

results_df = predict_families_for_fasta_file(args.dataset)
if args.window_size is not None:
  results_df.to_csv(args.dataset + f"_protcnn_results_windowsize{args.window_size:04d}_stride{args.window_stride}.csv", index=False)
else:
  results_df.to_csv(args.dataset + f"_protcnn_results_flexible_windowsize_stride{args.window_stride}_threshold{args.probability_threshold}.csv", index=False)
