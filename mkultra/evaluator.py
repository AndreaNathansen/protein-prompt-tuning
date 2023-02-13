import math
import os

import torch
from tqdm import tqdm

from mkultra.checkpoint_loader import CheckpointLoader


class Evaluator:
    """
    Evaluates a model with a trained soft prompt. Currently supports perplexity evaluation.
    """
    def __init__(self,
                model,
                is_prompt_tuned,
                data_loader_test,
                project_dir=None):
        torch.cuda.empty_cache()

        self.model=model
        self.data_loader_test = data_loader_test
        

        if is_prompt_tuned:
            checkpoint_loader = CheckpointLoader(project_dir)
            self.loaded_sp = checkpoint_loader.load_best_checkpoint()
            self.model.set_soft_prompt(self.loaded_sp)
    
    def evaluate_perplexity(self):
        """
        Perplexity calculation taken from https://huggingface.co/docs/transformers/perplexity, just without sliding window.
        Currently calculates perplexity per token. TODO: support perplexity per amino acid, to allow comparison between
        models with different vocabularies.
        """
        self.model.eval()
        eval_steps = len(self.data_loader_test)
        bar = tqdm(total=eval_steps)

        nlls = []
        num_amino_acids = 0

        eval_loss=0
        for i, (batch, attention_mask, labels) in enumerate(self.data_loader_test):
            # use cuda when on GPU
            input_ids = batch.cuda().detach()
            input_attention_mask = attention_mask.cuda().detach()
            input_labels = labels.cuda().detach()
            # input_ids = batch.detach()
            # input_attention_mask = attention_mask.detach()
            # input_labels = labels.detach()

            len_loss_included_tokens = (input_labels != -100).sum().to(dtype=torch.float32)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, labels=input_labels, attention_mask=input_attention_mask)
                # loss is calculated using CrossEntropyLoss which averages over input tokens.
                # Multiply it with len_loss_included_tokensto get the summation instead of average.
                # We will take average over all the tokens to get the true average
                # in the last step of this example.
                neg_log_likelihood = outputs.loss.to(dtype=torch.float32) * len_loss_included_tokens
                eval_loss += outputs.loss.item()
            nlls.append(neg_log_likelihood)
            num_amino_acids += len_loss_included_tokens

            instant_loss = outputs.loss.item()
            if math.isnan(instant_loss):
                    raise ValueError(f"NaN loss at step {i}")
            bar.update(1)
            bar.set_postfix({
                "Loss"   : instant_loss,
            })
        print(f"Loss: {eval_loss / eval_steps}")
        perplexity = torch.exp(torch.stack(nlls).sum() / num_amino_acids)
        return perplexity.item()
