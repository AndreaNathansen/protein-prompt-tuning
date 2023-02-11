import argparse
import json
import os
from pathlib import Path

import mkultra.sequence_loader as sequence_loader
import torch
from mkultra.trainers import SoftPromptTrainer
from mkultra.tuning import RITAPromptTuningLM
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer
from utils.train_utils import seed_everything

parser = argparse.ArgumentParser(prog="Prompt Tuning")
parser.add_argument("--config", dest="config", help="path to the JSON config file", required=True)
args = parser.parse_args()

with open(args.config) as config_file:
    config = json.load(config_file)

seed = config["seed"]
seed_everything(seed)

model_name = Path(config['model']).stem
sp_name = f"{config['project_name']}-{model_name}-fromvocab-{config['init_from_vocab']}"

# Specify the project directory bases.
project_dir_root = f"soft_prompts/{sp_name}/"
if not os.path.exists(project_dir_root):
    os.makedirs(project_dir_root)

block_size = config["block_size"]-config["n_tokens"]

for i in range(config["num_iterations"]):
    seed_everything(seed)
    torch.cuda.empty_cache()

    current_init_seed = seed + i
    project_dir = os.path.join(project_dir_root, f"{sp_name}-seed-{current_init_seed}")

    model = RITAPromptTuningLM.from_pretrained(config["model"]).half().to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(config["model"])

    dataset = sequence_loader.FastaDataset(config["dataset_file_train"], tokenizer, block_size, tokenizer.vocab['<PAD>'])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    dataset_val = sequence_loader.FastaDataset(config["dataset_file_validation"], tokenizer, block_size, tokenizer.vocab['<PAD>'])
    dataloader_val = DataLoader(dataset_val, batch_size=config["batch_size"], shuffle=False)

    optimizer_params = {"lr": config["learning_rate"]}
    
    trainer = SoftPromptTrainer(
        model=model,
        optimizer_class=AdamW,
        optimizer_params=optimizer_params,
        project_dir=project_dir,
        data_loader_train=dataloader,
        data_loader_eval=dataloader_val,
        checkpoint_interval=config["checkpoint_interval"],
        patience=config["patience"],
        n_tokens=config["n_tokens"],
        shuffle_seed=seed,
        init_from_vocab=config["init_from_vocab"],
        prompt_init_seed=current_init_seed)

    trainer.train(num_epochs=config["num_epochs"])     

