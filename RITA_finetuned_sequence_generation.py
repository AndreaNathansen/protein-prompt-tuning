import argparse
import csv
import json
import os
from pathlib import Path

"""
Script for generating a number of sequences with a finetuned RITA model. For a setup consistent to that of the prompt-tuned model,
the configuration parameters are to be loaded by specifying the config (see folder training_configs/) that was used for training the respective prompt.
The finetuned model is expected to be stored in ../transformers/{model_name}_finetune_test/, where model_name is the same as specified in
the prompt tuning config (e.g. RITA_s).
Currently, sequences are generated from scratch without any starting amino acid.
The generation sampling parameters are taken from https://github.com/lightonai/RITA
"""

import numpy as np
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from mkultra.checkpoint_loader import CheckpointLoader
from mkultra.tuning import RITAPromptTuningLM
from utils.train_utils import seed_everything

from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PretrainedConfig, pipeline)

parser = argparse.ArgumentParser(prog="Prompt Tuning")
parser.add_argument("--config", dest="config", help="path to the JSON config file", required=True)
parser.add_argument("--num-sequences", dest="num_sequences", type=int, help="number of sequences to generate", required=True)
parser.add_argument("--batch-size", dest="batch_size", type=int, help="amount of sequences to generate at once", required=True)
args = parser.parse_args()

with open(args.config) as config_file:
    config = json.load(config_file)

seed = config["seed"]
seed_everything(seed)
model_name = Path(config['model']).stem

block_size = config["block_size"]-config["n_tokens"]
num_sequences = args.num_sequences
batch_size = args.batch_size
num_batches = int(np.ceil(num_sequences / batch_size))
last_batch_size = num_sequences % batch_size

torch.cuda.empty_cache()
tokenizer = AutoTokenizer.from_pretrained(config["model"])

# Checkpoint loading
model = AutoModelForCausalLM.from_config(PretrainedConfig.from_json_file(f"../transformers/{model_name}_finetune_test/config.json"), trust_remote_code=True).half().to("cuda")
model.load_state_dict(torch.load(f"../transformers/{model_name}_finetune_test/pytorch_model.bin"))
model.eval()

for n in range(num_batches):
    if n == num_batches - 1:
        num_seq = last_batch_size
    else:
        num_seq = batch_size
    # leave out the EOS token that the RITA tokenizer always appends
    input_ids = tokenizer("<EOS>", return_tensors="pt").input_ids[:, :-1].to("cuda")
    output = model.generate(input_ids=input_ids, max_length=block_size, do_sample=True, top_k=950, repetition_penalty=1.2, 
                        num_return_sequences=num_seq, eos_token_id=2)
    sequences = [tokenizer.decode(output_ids) for output_ids in output]                   
    sequences_records = [SeqRecord(seq=Seq(sequence.replace('<EOS>','').replace(' ', '')), id=str(n*batch_size+j), name=f'generated_prot_{j}') for j, sequence in enumerate(sequences)]
    with open(os.path.join("generated_sequences", f"finetuned-{model_name}-generated.fasta"), 'a') as f:
        SeqIO.write(sequences_records, f, "fasta")

    del input_ids
    del output
    del sequences
    del sequences_records
    torch.cuda.empty_cache()
del model
torch.cuda.empty_cache()