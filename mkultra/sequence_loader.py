import torch
from Bio import SeqIO
from torch.utils.data import Dataset


class FastaDataset(Dataset):
    """
    Loads sequences from a fasta file and prepares them as tokenized and padded blocks for training/evaluation.
    Can be passed into a torch.utils.data.DataLoader as dataset.
    Allows exclusion of blocks that are filled with more padding tokens than specified by max_pad_token_fraction.
    """
    def __init__(self, filename, tokenizer, block_size, pad_token, max_pad_token_fraction=1.0):
        records = SeqIO.parse(filename, "fasta")
        sequences_encoded = [tokenizer.encode(str(record.seq)) for record in records]
        blocks = [[sequence[i:i+block_size] for i in range(0, len(sequence), block_size)] for sequence in sequences_encoded]
        blocks_flat = [block for sublist in blocks for block in sublist]
        blocks_flat = [block + [pad_token]*(block_size - len(block)) for block in blocks_flat]
        self.blocks = torch.tensor(blocks_flat)

        fraction_pad_tokens = (self.blocks == pad_token).sum(axis=1) / block_size
        self.blocks = self.blocks[fraction_pad_tokens <= max_pad_token_fraction]

        self.attention_masks = ~(self.blocks == pad_token) * 1
        self.labels = self.blocks.clone()
        # exclude pad token from loss
        self.labels[self.labels == pad_token] = -100

    def __getitem__(self, index):
        return self.blocks[index], self.attention_masks[index], self.labels[index]

    def __len__(self):
        return len(self.blocks)
