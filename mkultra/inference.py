from transformers import GPT2LMHeadModel, GPTNeoForCausalLM, GPTJForCausalLM, TextGenerationPipeline, AutoModelForCausalLM, AutoTokenizer
# Workaround so that RITA import works
AutoModelForCausalLM.from_pretrained("lightonai/RITA_s", trust_remote_code=True)
AutoTokenizer.from_pretrained("lightonai/RITA_s")
from transformers_modules.lightonai.RITA_s.fced662eadd2b7099a3b92a88365dfc3c98eb3da.rita_modeling import RITAModelForCausalLM
from mkultra.soft_prompt import SoftPrompt
import torch

EXTRA_ALLOWED_MODELS = [
    "GPT2SoftPromptLM",
    "GPTNeoSoftPromptLM",
    "GPTJSoftPromptLM"
    ]

#for model in EXTRA_ALLOWED_MODELS:
#    if model not in TextGenerationPipeline.ALLOWED_MODELS:
#        TextGenerationPipeline.ALLOWED_MODELS.append(model)

class SoftPromptMixin:
    def replace_special_tokens(self, input_ids):
        # Embed everything normally first
        inputs_embeds = self.transformer.get_input_embeddings()(input_ids.to(self.device))

        n_batches = input_ids.shape[0]
        n_tokens = input_ids.shape[-1]

        for b in range(n_batches):
            # Replace special tokens with soft prompts
            for t in range(n_tokens):
                # Check each id for a special token
                input_id = input_ids[b,t].item()
                sp = SoftPrompt.from_input_id(input_id)
                # If we find one, replace special token and padding with soft prompt
                if sp:
                    replacement = sp.get_inputs_embeds().to(self.device).clone().unsqueeze(0)
                    inputs_embeds[b,t:t+len(sp),:] = replacement[0,:,:]

        return inputs_embeds

    def forward(self, *args, **kwargs):
        # Alow setting input_ids as positional arg 0
        if kwargs.get('input_ids') is None:
            kwargs['input_ids'] = args[0]

        input_ids = kwargs.get('input_ids').to(self.device)

        if input_ids is None:
            # User is using inputs_embeds, nothing more we can do
            return super().forward(*args, **kwargs)

        kwargs = self.update_kwargs_input_ids(kwargs)
        kwargs['inputs_embeds'] = self.replace_special_tokens(input_ids)

        args = ()

        return super().forward(*args, **kwargs)

    def update_kwargs_input_ids(kwargs):
        raise NotImplementedError
        

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        # Alow setting input_ids as positional arg 0
        if kwargs.get('input_ids') is None:
            kwargs['input_ids'] = args[0]

        # This fixes CUDA for some reason
        kwargs['input_ids'] = kwargs['input_ids'].to(self.device)

        # Ban all special logits from output
        bad_words_ids = kwargs.get('bad_words_ids', list())

        for id in SoftPrompt.get_special_token_ids():
            bad_words_ids.append([id])
        kwargs['bad_words_ids'] = bad_words_ids

        args = ()

        return super().generate(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        SoftPrompt._register_model(model)
        return model


class GPTSoftPromptMixin(SoftPromptMixin):
    def update_kwargs_input_ids(self, kwargs_dict):
        kwargs_dict['input_ids'] = None
        return kwargs_dict

    def prepare_inputs_for_generation(self, input_ids, past=None, *args, **kwargs):
        input_ids = input_ids.to(self.device)
        return super().prepare_inputs_for_generation(input_ids, past, *args, **kwargs)


class RITASoftPromptMixin(SoftPromptMixin):
    def update_kwargs_input_ids(self, kwargs_dict):
        return kwargs_dict

    def prepare_inputs_for_generation(self, input_ids, *args, **kwargs):
        input_ids = input_ids.to(self.device)
        return super().prepare_inputs_for_generation(input_ids, *args, **kwargs)


class GPT2SoftPromptLM(GPTSoftPromptMixin, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

class GPTNeoSoftPromptLM(GPTSoftPromptMixin, GPTNeoForCausalLM):
    def __init__(self, config):
        super().__init__(config)

class GPTJSoftPromptLM(GPTSoftPromptMixin, GPTJForCausalLM):
    def __init__(self, config):
        super().__init__(config)

class RITAModelForCausalSoftPromptLM(RITASoftPromptMixin, RITAModelForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        