from phobius import scrape_phobius_for_transmembrane_domains
import numpy as np

def find_longest_repeat(seq, k):
    longest = [1] * len(seq)
    pattern = [None] * len(seq)
    
    seq_len = len(seq)
    for i in range(seq_len):
        if i + k <= seq_len:
            pattern[i] = seq[i:i+k]
        if i - k >= 0:
            if pattern[i-k] == pattern[i]:
                longest[i] = longest[i-k] + 1
    return max(longest)

def main(dataset):
    # remove seqences that don't start with methionine
    dataset = [seq for seq in dataset if str(seq.seq).startswith('M')]

    # remove sequences that contain ambiguous amino acids
    dataset = [seq for seq in dataset if 'Z' not in str(seq.seq) and 'X' not in str(seq.seq) and 'J' not in str(seq.seq) and 'B' not in str(seq.seq)]

    # remove sequences with long repeats
    dataset = [s for s in dataset if find_longest_repeat(str(s.seq), 1) <= 3 and find_longest_repeat(str(s.seq), 2) <= 4]

    # remove sequences with predicted transmembrane domains
    # hint - scraping might take several seconds to a few minutes, depending on the dataset size
    transmembrane_mask = scrape_phobius_for_transmembrane_domains(dataset)

    # transmembrane_mask is an array of boolean predictions, so we use its negation it to only retain sequences without predicted transmembrane domains
    return np.array(dataset, dtype=object)[~transmembrane_mask]
