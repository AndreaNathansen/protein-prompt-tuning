import os

import torch

from mkultra.soft_prompt import SoftPrompt


class CheckpointLoader:
    """
    Loads checkpoints for trained soft prompts which can then be added to a model from tuning.py.
    """
    def __init__(self, project_dir):
        self.project_dir = project_dir

    def _filename_for_checkpoint(self, epoch):
        return f"{self.project_name()}-epoch-{epoch}"
    
    def json_filename_for_checkpoint(self, epoch):
        return self._filename_for_checkpoint(epoch) + ".json"

    def optimizer_filename_for_checkpoint(self, epoch):
        return self._filename_for_checkpoint(epoch) + "-optimizer.pt"

    def project_name(self):
        return os.path.basename(os.path.normpath(self.project_dir))

    def load_latest_checkpoint(self):
        """ Returns: highest epoch and soft prompt for latest checkpoint."""
        project_files = os.listdir(self.project_dir)
        if project_files is not None:
            checkpoint_files = [check_file for check_file in project_files if ('-epoch-' in check_file and not 'optimizer' in check_file) ]
            if len(checkpoint_files) > 0:
                highest_epoch = max([ int(check_file[check_file.rfind('-epoch-')+7:-5]) for check_file in checkpoint_files ])
                print(f"Loading latest checkpoint: {highest_epoch}")
                return highest_epoch, SoftPrompt.from_file( os.path.join(self.project_dir, self.json_filename_for_checkpoint(highest_epoch)) )
            else:
                print("No checkpoints found")

        return None, None
    
    
    def load_best_checkpoint(self):
        """
        Returns: soft prompt for checkpoint that had the minimum validation loss.
        """
        _, latest_sp = self.load_latest_checkpoint()
        min_eval_loss_epoch = latest_sp._metadata['min_eval_loss_epoch']
        print(f"Loading best checkpoint: {min_eval_loss_epoch}")
    
        return SoftPrompt.from_file( os.path.join(self.project_dir, self.json_filename_for_checkpoint(min_eval_loss_epoch)) )
    

    def load_optimizer_state_dict(self, highest_epoch):
        """
        Returns: loptimizer state dict for an epoch (usually the highest epoch where training was interrupted) to continue training.
        """
        if highest_epoch is not None:
            state = torch.load(os.path.join(self.project_dir, self.optimizer_filename_for_checkpoint(highest_epoch)))
            return state
