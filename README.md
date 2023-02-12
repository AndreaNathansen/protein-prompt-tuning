# Prompt tuning for protein sequence generation, implementation for paper XXX (TODO: add)
Adapted (forked and modified/enhanced) from https://github.com/corolla-johnson/mkultra, which is a prompt
tuning implementation for NLP, supporting GPT-2.
The prompt tuning method was developed by Lester et al., see Lester et al. The power of scale for parameter-efficient prompt tuning.
For setup, you can reconstruct our conda environment by the `environment.yaml` file.

## Models
This implementation currently support ProtGPT2 (Ferruz et al. ProtGPT2 is a deep unsupervised language model for protein design.)
and RITA (Hesslow et al. RITA: a Study on Scaling Up Generative Protein Sequence Models.). 

## Training
Training scripts for RITA and ProtGPT2 are `RITA_prompt_tuning.py` and `ProtGPT2_prompt_tuning.py`, respectively.
They can be configured with parameters specified in a JSON config, see `training_configs/` folder and the
Trainer documentation in `mkultra/trainers.py`.
The training configurations in `training_configs/` are the ones that we used for our paper.

## Datasets
You can train the model with Fasta datasets if you use the `FastaDataset` class (`mkultra.sequence_loader.py`)
as dataset input for a PyTorch `DataLoader`.
Dataset preprocessing as in RITA can be found in the `utils` folder (TODO: add exact scripts).

We also provide the datasets that we used for our experiments in the `datasets/` folder.
They contain sequences from the Pfam family PF03272.
The dataset (`datasets/InterProUniprotPF03272.fasta`) was downloaded from [InterPro](https://www.ebi.ac.uk/interpro/entry/pfam/PF03272/protein/UniProt/)
on January 5, 2023 (Paysan-Lafosse et al. InterPro in 2022.). Then, we removed all sequences containing an X, which created the dataset `InterProUniprotPF03272_Xremoved.fasta`.
We clustered our data with [MMseqs2](https://github.com/soedinglab/MMseqs2)(Steinegger et al. MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. And: Steinegger et al. Clustering huge protein sequence sets in linear time.). The datasets clustered with a sequence similarity threshold of 100%,
which were our main datasets, are `InterProUniprotPF03272prepared_<train|validation|test>.fasta`. Datasets clustered with different
thresholds can be found as `InterProUniprotPF03272_<threshold>_<train|validation|test>.fasta`

## Evaluation
This implementation currently supports perplexity evaluation as perplexity per token.
See `RITA_prompt_comparison_basemodel.py` and `RITA_prompt_comparison_trainvaltest.py`, which are the evaluation scripts that
we used for our experiments. If you want to write an evaluation script yourself, have a look at `mkultra/evaluator.py`.

We further evaluated the amount of generated sequences that were classified by ProtCNN (Bileschi et al. Using deep learning to annotate the protein universe.) as belonging to our target family PF03272.
The script we used for that is `protcnn.py`, taken and adapted from the [official ProtCNN notebook](https://github.com/google-research/google-research/blob/master/using_dl_to_annotate_protein_universe/Using_Deep_Learning_to_Annotate_the_Protein_Universe.ipynb). As ProtCNN uses Tensorflow and we wanted to keep it separate from our other scripts, we created an extra conda environment for it, specified in `environment-protcnn.yml`.

## Sequence generation
For generating sequences, instantiate a base model and then load and add a trained prompt, as for example in `RITA_prompt_sequence_generation.py`.

TODO: add ipynb example notebooks


## Limitations (for now)
- mkultra syncs a set of special tokens between its tokenizers the scenes. Adding your own tokens may result in unexpected behaviour.
