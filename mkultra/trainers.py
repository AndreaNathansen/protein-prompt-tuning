import math
import os
import random

import torch
from tqdm import tqdm

from mkultra.checkpoint_loader import CheckpointLoader
from mkultra.soft_prompt import SoftPrompt


class SoftPromptTrainer:
    """
    A trainer for soft prompts. Stores only the respective soft prompt for checkpoint
    (along with the optimizer state to allow continuing training after interruption)
    and not the whole model.
    """
    def __init__(self,
                 model,
                 optimizer_class,
                 optimizer_params,
                 project_dir,
                 data_loader_train,
                 data_loader_eval,
                 patience=None,
                 n_tokens=20,
                 ema_alpha=0.1,
                 checkpoint_interval=1,
                 init_from_vocab=True,
                 prompt_init_seed=None,
                 shuffle_seed=None,
                 logging_interval=100):
        """
        Params:
            model: A model that supports setting soft prompts, see tuning.py.
            optimizer_class: The class of the optimizer to use. The optimizer has to be instantiated
                after initializing/loading the soft prompt and adding it to the model as trainable parameters,
                in order to avoid bugs/behaviour where the soft prompt does not get updated during training.
                That is why this method takes the class and hyperparameters for the optimizer seperately.
            optimizer_params: Hyperparameters for the optimizer, which have to match the optimizer class.
            project_dir: The project directory.
            data_loader_train/_val: Data loader for training and validation set.
            patience: Number of epochs for early stopping.
            n_tokens: Soft prompt length.
            ema_alpha: Parameter for EMAl loss calculation (for logging training loss)
            checkpoint_interval: The interval after which a checkpoint is saved (in epochs). An evaluation run on the 
                validation set is done once per epoch.
            init_from_vocab: If true, initialize a prompt from the model's vocabulary embeddings. If False, initialize
                from a uniform distribution of range [-0.5, 0.5] (as in Lester et al. The power of scale for parameter-efficient
                prompt tuning.)
            prompt_init_seed: Random seed for prompt initialization.
            shuffle_seed: Seed for dataset shuffling during training.
            logging_interval: Interval for logging (in steps (1 step = 1 batch processed), not epochs)

        """
        torch.cuda.empty_cache()

        self.model=model
        self.project_dir=project_dir
        self.data_loader_train=data_loader_train
        self.data_loader_eval=data_loader_eval
        self.patience=patience
        self.n_tokens=n_tokens
        self.ema_alpha=ema_alpha
        self.checkpoint_interval=checkpoint_interval
        self.shuffle_seed = shuffle_seed
        self.logging_interval = logging_interval

        self._maybe_create_project_directory()
        self.checkpoint_loader = CheckpointLoader(self.project_dir)
        highest_epoch, self.loaded_sp = self.checkpoint_loader.load_latest_checkpoint()

        # Initialize soft prompt in model
        if self.loaded_sp is None:
            self.model.initialize_soft_prompt(n_tokens=n_tokens, init_from_vocab=init_from_vocab, prompt_init_seed=prompt_init_seed)
            self.sp_epoch = 0
            self.ema_loss = None
            self.eval_loss = None
            self.min_eval_loss = math.inf
            self.min_eval_loss_epoch = 0
        else:
            self.model.set_soft_prompt(self.loaded_sp)
            # the saved epoch is the previous one
            self.sp_epoch = self.loaded_sp._metadata['epoch'] + 1
            self.ema_loss = self.loaded_sp._metadata['loss']
            self.eval_loss = self.loaded_sp._metadata['eval_loss']
            self.min_eval_loss = self.loaded_sp._metadata['min_eval_loss']
            self.min_eval_loss_epoch = self.loaded_sp._metadata['min_eval_loss_epoch']
        optimizer_params['params'] = [self.model.get_soft_params()]
        self.optimizer = optimizer_class(**optimizer_params)
        self._load_optimizer_state_dict(highest_epoch)

    def _maybe_create_project_directory(self):
        # Look for existing project directory
        try:
            os.makedirs(self.project_dir)
            print(f"Created project directory at {self.project_dir}")
        except FileExistsError:
            print(f"Found project directory at {self.project_dir}")


    def _load_optimizer_state_dict(self, highest_epoch):
        if highest_epoch is not None:
            state = self.checkpoint_loader.load_optimizer_state_dict(highest_epoch)
            self.optimizer.load_state_dict(state)

    def _save_checkpoint(self):
        sp = SoftPrompt.from_tuning_model(self.model,
                    {"name"     : f"{self.checkpoint_loader.project_name()} Epoch {self.sp_epoch}",
                    "epoch"     : self.sp_epoch,
                    "loss"      : self.ema_loss,
                    "min_eval_loss": self.min_eval_loss,
                    "min_eval_loss_epoch": self.min_eval_loss_epoch,
                    "eval_loss": self.eval_loss})
        sp.to_file( os.path.join( self.project_dir,self.checkpoint_loader.json_filename_for_checkpoint(self.sp_epoch) ))
        torch.save(self.optimizer.state_dict(), os.path.join(self.project_dir, self.checkpoint_loader.optimizer_filename_for_checkpoint(self.sp_epoch)))

    def train(self, num_epochs=1):
        self.model.train()
        torch.cuda.empty_cache()
        loss_log_path_train = os.path.join(self.project_dir,"loss_log_train.csv")
        loss_log_path_eval = os.path.join(self.project_dir,"loss_log_eval.csv")
        steps_per_epoch = len(self.data_loader_train)
        bar = tqdm(total=steps_per_epoch * num_epochs)

        while self.sp_epoch < num_epochs:
            self.model.train()
            current_seed = self.shuffle_seed + self.sp_epoch
            torch.manual_seed(current_seed)
            torch.cuda.manual_seed(current_seed)
            for i, (batch, attention_mask, labels) in enumerate(self.data_loader_train):
                # use cuda when on GPU
                input_ids = batch.cuda().detach()
                input_attention_mask = attention_mask.cuda().detach()
                input_labels = labels.cuda().detach()
                # input_ids = batch.detach()
                # input_attention_mask = attention_mask.detach()
                # input_labels = labels.detach()

                # Forward pass and optimize
                outputs = self.model(input_ids=input_ids, labels=input_labels, attention_mask=input_attention_mask)
                loss = outputs.loss
                loss.backward()
                instant_loss = loss.item()

                self.optimizer.step()
                self.optimizer.zero_grad()

                lr = self.optimizer.param_groups[0]["lr"]

                # Discard tensor that was moved to GPU
                del input_ids
                del input_labels
                del input_attention_mask
                torch.cuda.empty_cache()

                if math.isnan(instant_loss):
                    raise ValueError(f"NaN loss at step {i} in epoch {self.sp_epoch}")

                # Calculate EMA loss
                self.ema_loss = self.ema_alpha*instant_loss + (1-self.ema_alpha)*self.ema_loss if self.ema_loss is not None else instant_loss
            
                if i % self.logging_interval == 0:
                    bar.set_postfix({
                        "Epoch" : self.sp_epoch,
                        "Step" : i,
                        "EMA Loss" : self.ema_loss,
                        "lr": lr
                    })
                    bar.update(self.logging_interval)

                total_step = self.sp_epoch * steps_per_epoch + i

                with open(loss_log_path_train, 'a', encoding='utf-8') as file:
                    file.write(f"{total_step},{self.ema_loss}\n")

            # Save checkpoint every so often and in the last epoch
            if (self.sp_epoch%self.checkpoint_interval == 0) or (self.sp_epoch == num_epochs - 1):
                # evaluate once per checkpoint
                self.eval_loss = self.evaluate()
                with open(loss_log_path_eval, 'a', encoding='utf-8') as file:
                    file.write(f"{self.sp_epoch},{self.eval_loss}\n")
                if self.eval_loss < self.min_eval_loss:
                    self.min_eval_loss = self.eval_loss
                    self.min_eval_loss_epoch = self.sp_epoch

                self._save_checkpoint()

                # Early stopping if patience is set and eval loss hasn't increased for >=patience epochs
                if self.patience is not None:
                    if self.sp_epoch - self.min_eval_loss_epoch >= self.patience:
                        print(f"Eval loss hasn't increased for {self.sp_epoch - self.min_eval_loss_epoch} epochs, \
                            stopping training after epoch {self.sp_epoch}.")
                        break

            self.sp_epoch += 1

    def evaluate(self):
        self.model.eval()
        eval_steps = len(self.data_loader_eval)
        bar = tqdm(total=eval_steps)

        eval_loss = 0
        for i, (batch, attention_mask, labels) in enumerate(self.data_loader_eval):
            # use cuda when on GPU
            input_ids = batch.cuda().detach()
            input_attention_mask = attention_mask.cuda().detach()
            input_labels = labels.cuda().detach()
            #input_ids = batch.detach()
            #input_attention_mask = attention_mask.detach()
            #input_labels = labels.detach()

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, labels=input_labels, attention_mask=input_attention_mask)
                loss = outputs.loss.item()
            eval_loss += loss

            # Discard tensor that was moved to GPU
            del input_ids
            torch.cuda.empty_cache()
            
            if i % self.logging_interval == 0:
                bar.set_postfix({
                    "Loss"   : loss,
                })
                bar.update(self.logging_interval)

        eval_loss /= eval_steps
        return eval_loss
