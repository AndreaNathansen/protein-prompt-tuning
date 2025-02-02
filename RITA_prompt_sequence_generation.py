import argparse
import csv
import json
import os
from pathlib import Path

"""
Script for generating a number of sequences with a prompt-tuned RITA model and the respective
base model (= without prompt). The prompt-tuned model along with its base model can be
loaded by specifying the config (see folder training_configs/) that was used for training the prompt.
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
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from utils.train_utils import seed_everything

parser = argparse.ArgumentParser(prog="Prompt Tuning")
parser.add_argument("--config", dest="config", help="path to the JSON config file", required=True)
parser.add_argument("--num-sequences", dest="num_sequences", type=int, help="number of sequences to generate", required=True)
parser.add_argument("--batch-size", dest="batch_size", type=int, help="amount of sequences to generate at once", required=True)
args = parser.parse_args()

with open(args.config) as config_file:
    config = json.load(config_file)

seed = config["seed"]
seed_everything(seed)

# Name of the soft prompt project to evaluate.
model_name = Path(config['model']).stem
sp_name = f"{config['project_name']}-{model_name}-fromvocab-{config['init_from_vocab']}"

# Specify the project directory bases.
project_dir_root = f"soft_prompts/{sp_name}/"
if not os.path.exists(project_dir_root):
    os.makedirs(project_dir_root)

block_size = config["block_size"]-config["n_tokens"]
num_sequences = args.num_sequences
batch_size = args.batch_size
num_batches = int(np.ceil(num_sequences / batch_size))
last_batch_size = num_sequences % batch_size

torch.cuda.empty_cache()
tokenizer = AutoTokenizer.from_pretrained(config["model"])

# Generate sequences with the base model
seed_everything(seed)
base_model = AutoModelForCausalLM.from_pretrained(config["model"], trust_remote_code=True).half().to("cuda")

for n in range(num_batches):
    if n == num_batches - 1:
        num_seq = last_batch_size
    else:
        num_seq = batch_size
    # leave out the EOS token that the RITA tokenizer always appends
    base_input_ids = tokenizer("<EOS>", return_tensors="pt").input_ids[:, :-1].to("cuda")
    base_output = base_model.generate(input_ids=base_input_ids, max_length=block_size, do_sample=True, top_k=950, repetition_penalty=1.2, 
                        num_return_sequences=num_seq, eos_token_id=2)
    base_sequences = [tokenizer.decode(output_ids) for output_ids in base_output]                   
    base_sequences_records = [SeqRecord(seq=Seq(sequence.replace('<EOS>','').replace(' ', '')), id=str(n*batch_size+j), name=f'generated_prot_{j}') for j, sequence in enumerate(base_sequences)]
    with open(os.path.join("generated_sequences", f"basemodel-{model_name}-generated.fasta"), 'a') as f:
        SeqIO.write(base_sequences_records, f, "fasta")

    del base_input_ids
    del base_output
    del base_sequences
    del base_sequences_records
    torch.cuda.empty_cache()
del base_model
torch.cuda.empty_cache()


# Generate sequences for each seed for the chosen model
for i in range(config["num_iterations"]):
    seed_everything(seed)
    torch.cuda.empty_cache()

    current_init_seed = seed + i
    project_dir = os.path.join(project_dir_root, f"{sp_name}-seed-{current_init_seed}")

    model = RITAPromptTuningLM.from_pretrained(config["model"]).half().to("cuda")
    checkpoint_loader = CheckpointLoader(project_dir)
    loaded_sp = checkpoint_loader.load_best_checkpoint()
    model.set_soft_prompt(loaded_sp)

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
        sequences_records = [SeqRecord(seq=Seq(sequence.replace('<EOS>','').replace(' ', '')), id=str(n*batch_size+j)) for j, sequence in enumerate(sequences)]
        with open(os.path.join("generated_sequences", f"{sp_name}-seed-{i}-generated.fasta"), 'a') as f:
            SeqIO.write(sequences_records, f, "fasta")

        del input_ids
        del output
        del sequences
        del sequences_records
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
