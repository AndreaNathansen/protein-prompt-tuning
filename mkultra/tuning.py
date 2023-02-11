from transformers import GPT2LMHeadModel, GPTNeoForCausalLM, GPTJForCausalLM, TextGenerationPipeline, AutoModelForCausalLM, AutoTokenizer
# Workaround so that RITA import works
AutoModelForCausalLM.from_pretrained("lightonai/RITA_s", trust_remote_code=True)
AutoTokenizer.from_pretrained("lightonai/RITA_s")
from transformers_modules.lightonai.RITA_s.fced662eadd2b7099a3b92a88365dfc3c98eb3da.rita_modeling import RITAModelForCausalLM
from mkultra.soft_prompt import SoftPrompt
import torch
import torch.nn as nn
import numpy as np

class PromptTuningMixin:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        for param in model.parameters():
            param.requires_grad = False

        model.learned_embedding = None

        return model

    def initialize_soft_prompt(self, n_tokens = 20, init_from_vocab=True, prompt_init_seed=None):
        init_rng = np.random.default_rng(prompt_init_seed)
        if init_from_vocab:
            vocab_size = self.transformer.get_input_embeddings().weight.shape[0]
            init_vocab_idcs = init_rng.choice(vocab_size, size=n_tokens, replace=True)
            self.learned_embedding = nn.parameter.Parameter(
                self.transformer.get_input_embeddings().weight[init_vocab_idcs].clone().detach())
        else:
            input_dim = self.transformer.get_input_embeddings().weight.shape[1]
            init_values = init_rng.uniform(low=-0.5, high=0.5, size=(n_tokens, input_dim))
            self.learned_embedding = nn.parameter.Parameter(torch.Tensor(init_values).to(self.transformer.get_input_embeddings().weight.dtype).to(self.device).detach())

    def set_soft_prompt_embeds(self, soft_prompt_embeds):
        self.learned_embedding = nn.parameter.Parameter(soft_prompt_embeds.clone().detach())

    def set_soft_prompt(self, sp: SoftPrompt):
        self.learned_embedding = nn.parameter.Parameter(sp.get_inputs_embeds().clone().detach().squeeze(0))

    def get_soft_params(self):
        return self.learned_embedding

    def _cat_learned_embedding_to_input(self, input_ids):
        inputs_embeds = self.transformer.get_input_embeddings()(input_ids)

        if len(list(inputs_embeds.shape)) == 2:
            ie = inputs_embeds.unsqueeze(0)
        else:
            ie = inputs_embeds

        # GPT2 has dropout on embeddings
        learned_embedding = self._drop_embedding_if_supported()

        inputs_embeds = torch.cat([learned_embedding.repeat(ie.size(0), 1, 1),
                                   ie],
                                   dim=1)

        return inputs_embeds
    
    def _drop_embedding_if_supported(self):
        raise NotImplementedError

    def _extend_labels(self, labels):
        n_tokens = self.learned_embedding.shape[-2]

        if len(list(labels.shape)) == 1:
            lb = labels.unsqueeze(0)
        else:
            lb = labels

        # Add '-100's (prevent loss calculation where the learned embed would be)
        n_batches = lb.shape[0]
        return torch.cat([torch.full((n_batches,n_tokens), -100).to(self.device), lb], dim=1)

    def _extend_attention_mask(self, attention_mask):
        n_tokens = self.learned_embedding.shape[-2]

        if len(list(attention_mask.shape)) == 1:
            am = attention_mask.unsqueeze(0)
        else:
            am = attention_mask

        n_batches = am.shape[0]
        return torch.cat([torch.full((n_batches,n_tokens), 1).to(self.device), am], dim=1)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        # This fixes CUDA for some reason
        kwargs['input_ids'] = kwargs['input_ids'].to(self.device)

        return super().generate(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if input_ids is not None:
            inputs_embeds = self._cat_learned_embedding_to_input(input_ids)

        input_ids = self._get_input_ids_if_required(input_ids)

        if labels is not None:
            labels = self._extend_labels(labels)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask)

        # Drop most of the args for now
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict,
        )
    
    def _get_input_ids_if_required(self, input_ids):
        raise NotImplementedError

class RITAPromptTuningLM(PromptTuningMixin, RITAModelForCausalLM):
    def __init__(self, config):
        super().__init__(config)
    
    def prepare_inputs_for_generation(self, input_ids, past=None, *args, **kwargs):
        input_ids = input_ids.to(self.device)
        return super().prepare_inputs_for_generation(input_ids, *args, **kwargs)
    
    def _drop_embedding_if_supported(self):
        return self.learned_embedding
    
    def _get_input_ids_if_required(self, input_ids):
        if input_ids is not None:
            input_ids = self._extend_labels(input_ids)
        return input_ids


class GPT2PromptTuningLM(PromptTuningMixin, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

    def prepare_inputs_for_generation(self, input_ids, past=None, *args, **kwargs):
        input_ids = input_ids.to(self.device)
        # Drop 'past' to make things easier for us later
        return super().prepare_inputs_for_generation(input_ids, None, *args, **kwargs)
    
    def _drop_embedding_if_supported(self):
        return self.transformer.drop(self.learned_embedding)

    def _get_input_ids_if_required(self, input_ids):
        return None

class GPTNeoPromptTuningLM(PromptTuningMixin, GPTNeoForCausalLM):
    def __init__(self, config):
        super().__init__(config)

class GPTJPromptTuningLM(PromptTuningMixin, GPTJForCausalLM):
    def __init__(self, config):
        super().__init__(config)