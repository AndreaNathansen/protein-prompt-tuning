import argparse
import os
import json
from utils.train_utils import seed_everything
from datetime import datetime
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from mkultra.tuning import GPT2PromptTuningLM
import mkultra.sequence_loader as sequence_loader
from mkultra.trainers import SoftPromptTrainer
from transformers import Adafactor, AdamW

parser = argparse.ArgumentParser(prog="Prompt Tuning")
parser.add_argument("--config", dest="config", help="path to the JSON config file", required=True)
args = parser.parse_args()

with open(args.config) as config_file:
    config = json.load(config_file)

seed_everything(config["shuffle_seed"])

# Name your soft prompt project.
sp_name = f"prompt-tuning-try-training-RITA-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Specify the project directory bases.
project_dir_root = f"soft_prompts/{sp_name}/"
if not os.path.exists(project_dir_root):
    os.makedirs(project_dir_root)

model = GPT2PromptTuningLM.from_pretrained(config["model"]).half().to("cuda")
tokenizer = AutoTokenizer.from_pretrained(config["model"])

project_dir = os.path.join(project_dir_root, sp_name)

# TODO: load proper train set
block_size = config["block_size"]-config["n_tokens"]
dataset = sequence_loader.FastaDataset(config["dataset_file_train"], tokenizer, block_size, 0)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True) # TODO use shuffle_seed for shuffling?

dataset_val = sequence_loader.FastaDataset(config["dataset_file_validation"], tokenizer, block_size, 0)
dataloader_val = DataLoader(dataset_val, batch_size=config["batch_size"], shuffle=False)

trainer = SoftPromptTrainer(
            model=model,
            tokenizer=tokenizer,
            optimizer_class=AdamW,
            optimizer_params=config["optimizer_params"],
            project_dir=project_dir,
            data_loader_train=dataloader,
            data_loader_eval=dataloader_val,
            checkpoint_interval=1,
            patience=20,
            n_tokens=config["n_tokens"],
            shuffle_seed=config["shuffle_seed"],
            init_from_vocab=config["init_from_vocab"],
            prompt_init_seed=config["shuffle_seed"])
trainer.train(num_epochs=300)


#eval_loss = trainer.evaluate(eval_percentage=eval_percentage)
#dataset_test = sequence_loader.FastaDataset(config["dataset_file_test"], tokenizer, block_size, tokenizer.vocab['<PAD>'])
#dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

# trainer.evaluate_perplexity(dataloader_test)
