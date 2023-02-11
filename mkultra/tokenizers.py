from transformers import GPT2Tokenizer, GPT2TokenizerFast, PreTrainedTokenizerFast
from mkultra.soft_prompt import SoftPrompt

class GPT2SPTokenizerFast(GPT2TokenizerFast):
    def __init__(
        self,
        vocab_file,
        merges_file,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        add_prefix_space=False,
        **kwargs
    ):
        print(vocab_file)
        print(merges_file)
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
        SoftPrompt._register_tokenizer(self)

class GPT2SPTokenizer(GPT2Tokenizer):
    def __init__(
        self,
        vocab_file,
        merges_file,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        add_prefix_space=False,
        **kwargs
    ):
        print(vocab_file)
        print(merges_file)
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
        SoftPrompt._register_tokenizer(self)

class RITASPTokenizerFast(PreTrainedTokenizerFast):
    def __init__(
        self,
        #vocab_file,
        #merges_file,
        tokenizer_file=None,
        unk_token="<unk>",
        # TODO: check: is this padding?
        bos_token="<EOS>",
        eos_token="<EOS>",
        add_prefix_space=False,
        **kwargs
    ):
        super().__init__(
            #vocab_file,
            #merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
        SoftPrompt._register_tokenizer(self)
