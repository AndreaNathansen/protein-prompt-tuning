"""
Perplexity evaluation for a prompt-tuned RITA model on the training and validation set for a training run
and on the test set. The prompt-tuned model along with the datasets can be
loaded by specifying the config (see folder training_configs/) that was used for training the prompt.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import mkultra.sequence_loader as sequence_loader
import numpy as np
import torch
from mkultra.evaluator import Evaluator
from mkultra.tuning import RITAPromptTuningLM
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.train_utils import seed_everything

parser = argparse.ArgumentParser(prog="Prompt Tuning")
parser.add_argument("--config", dest="config", help="path to the JSON config file", required=True)
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

csv_path = os.path.join("experiment_results", f"{sp_name}-trainvaltest_comparison.csv") 
header = ["train", "val", "test"]

block_size = config["block_size"]-config["n_tokens"]


torch.cuda.empty_cache()
tokenizer = AutoTokenizer.from_pretrained(config["model"])

dataset_train = sequence_loader.FastaDataset(config["dataset_file_train"], tokenizer, block_size, tokenizer.vocab['<PAD>'], tokenizer.vocab['<EOS>'], tokenizer.vocab['<EOS>'])
dataloader_train = DataLoader(dataset_train, batch_size=config["batch_size"], shuffle=False)
dataset_val = sequence_loader.FastaDataset(config["dataset_file_validation"], tokenizer, block_size, tokenizer.vocab['<PAD>'], tokenizer.vocab['<EOS>'], tokenizer.vocab['<EOS>'])
dataloader_val = DataLoader(dataset_val, batch_size=config["batch_size"], shuffle=False)
dataset_test = sequence_loader.FastaDataset(config["dataset_file_test"], tokenizer, block_size, tokenizer.vocab['<PAD>'], tokenizer.vocab['<EOS>'], tokenizer.vocab['<EOS>'])
dataloader_test = DataLoader(dataset_test, batch_size=config["batch_size"], shuffle=False)

# One entry {train: ..., val: ..., test: ...} per initialization seed
perplexities_seeds = [{} for _ in range(config["num_iterations"])]
for i in range(config["num_iterations"]):
    seed_everything(seed)
    torch.cuda.empty_cache()

    current_init_seed = seed + i
    project_dir = os.path.join(project_dir_root, f"{sp_name}-seed-{current_init_seed}")

    model = RITAPromptTuningLM.from_pretrained(config["model"]).half().to("cuda")
    
    perplexities_seeds[i]["train"] = Evaluator(
        model=model,
        is_prompt_tuned=True,
        data_loader_test=dataloader_train,
        project_dir=project_dir).evaluate_perplexity()
    
    perplexities_seeds[i]["val"] = Evaluator(
        model=model,
        is_prompt_tuned=True,
        data_loader_test=dataloader_val,
        project_dir=project_dir).evaluate_perplexity()

    perplexities_seeds[i]["test"] = Evaluator(
        model=model,
        is_prompt_tuned=True,
        data_loader_test=dataloader_test,
        project_dir=project_dir).evaluate_perplexity()

torch.cuda.empty_cache()

with open(csv_path, 'w') as f:
    csvwriter = csv.DictWriter(f, fieldnames=header)
    csvwriter.writeheader()
    for i in range(config["num_iterations"]):
        csvwriter.writerow(perplexities_seeds[i])
