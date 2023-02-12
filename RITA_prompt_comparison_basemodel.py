"""
Perplexity evaluation for a prompt-tuned RITA model and comparison to the respective
base model (= without prompt). The prompt-tuned model along with its base model can be
loaded by specifying the config (see folder training_configs/) that was used for training the prompt.
Evaluation is done on the test set that is specified in the configs.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import mkultra.sequence_loader as sequence_loader
from mkultra.evaluator import Evaluator
from mkultra.tuning import RITAPromptTuningLM
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

csv_path = os.path.join("experiment_results", f"{sp_name}-test_perplexity_comparison.csv") 
header = ['prompt_tuned', 'base']

block_size = config["block_size"]-config["n_tokens"]


torch.cuda.empty_cache()
tokenizer = AutoTokenizer.from_pretrained(config["model"])
dataset_test = sequence_loader.FastaDataset(config["dataset_file_test"], tokenizer, block_size, tokenizer.vocab['<PAD>'])
dataloader_test = DataLoader(dataset_test, batch_size=config["batch_size"], shuffle=False)


# Compare test perplexity for the base model
seed_everything(seed)
base_model = AutoModelForCausalLM.from_pretrained(config["model"], trust_remote_code=True).half().to("cuda")

base_evaluator = Evaluator(
        model=base_model,
        is_prompt_tuned=False,
        data_loader_test=dataloader_test)
    
base_perplexity = base_evaluator.evaluate_perplexity()
torch.cuda.empty_cache()

# One entry {prompt_tuned: ..., base: ...} per initialization seed.
# Base model has only one run, but this format makes postprocessing easier
perplexities_seeds = [{} for _ in range(config["num_iterations"])]
# Compare test perplexity for each seed for the chosen model
for i in range(config["num_iterations"]):
    seed_everything(seed)
    torch.cuda.empty_cache()

    current_init_seed = seed + i
    project_dir = os.path.join(project_dir_root, f"{sp_name}-seed-{current_init_seed}")

    model = RITAPromptTuningLM.from_pretrained(config["model"]).half().to("cuda")

    evaluator = Evaluator(
        model=model,
        is_prompt_tuned=True,
        data_loader_test=dataloader_test,
        project_dir=project_dir)
    
    perplexities_seeds[i][f"prompt_tuned"] = evaluator.evaluate_perplexity()
    perplexities_seeds[i][f"base"] = base_perplexity

with open(csv_path, 'w') as f:
    csvwriter = csv.DictWriter(f, fieldnames=header)
    csvwriter.writeheader()
    for i in range(config["num_iterations"]):
        csvwriter.writerow(perplexities_seeds[i])
