"""
Perplexity evaluation for a finetuned RITA model on the training and validation set for a training run
and on the test set. For a setup consistent to that of the prompt-tuned model, the configuration parameters are to be loaded
by specifying the config (see folder training_configs/) that was used for training the respective prompt.
The finetuned model is expected to be stored in ../transformers/{model_name}_finetune_test/, where model_name is the same as specified in
the prompt tuning config (e.g. RITA_s).
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
from utils.train_utils import seed_everything

from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

parser = argparse.ArgumentParser(prog="Prompt Tuning")
parser.add_argument("--config", dest="config", help="path to the JSON config file", required=True)
args = parser.parse_args()

with open(args.config) as config_file:
    config = json.load(config_file)

seed = config["seed"]
seed_everything(seed)
model_name = Path(config['model']).stem

# Checkpoint loading
model = AutoModelForCausalLM.from_config(PretrainedConfig.from_json_file(f"../transformers/{model_name}_finetune_test/config.json"), trust_remote_code=True).half().to("cuda")
model.load_state_dict(torch.load(f"../transformers/{model_name}_finetune_test/pytorch_model.bin"))
model.eval()

csv_path = os.path.join("experiment_results", f"finetuned-{model_name}-trainvaltest_comparison.csv") 
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

# One entry {train: ..., val: ..., test: ...}
seed_everything(seed)
torch.cuda.empty_cache()

perplexities = {}

perplexities["train"] = Evaluator(
    model=model,
    is_prompt_tuned=False,
    data_loader_test=dataloader_train).evaluate_perplexity()

perplexities["val"] = Evaluator(
    model=model,
    is_prompt_tuned=False,
    data_loader_test=dataloader_val).evaluate_perplexity()

perplexities["test"] = Evaluator(
    model=model,
    is_prompt_tuned=False,
    data_loader_test=dataloader_test).evaluate_perplexity()

torch.cuda.empty_cache()

with open(csv_path, 'w') as f:
    csvwriter = csv.DictWriter(f, fieldnames=header)
    csvwriter.writeheader()
    csvwriter.writerow(perplexities)
