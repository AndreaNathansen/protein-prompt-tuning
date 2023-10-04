# Implementation for the paper Evaluating Prompt Tuning for Conditional Protein Sequence Generation.
Adapted (forked and modified/enhanced) from https://github.com/corolla-johnson/mkultra, which is a prompt
tuning implementation for NLP, supporting GPT-2.
The prompt tuning method was developed by Lester et al., see [Lester et al. The power of scale for parameter-efficient prompt tuning.](https://aclanthology.org/2021.emnlp-main.243/)
For setup, you can reconstruct our conda environment for prompt tuning by the `environment.yaml` file.

## Models
This implementation currently supports ProtGPT2 ([Ferruz et al. ProtGPT2 is a deep unsupervised language model for protein design.](https://doi.org/10.1038/s41467-022-32007-7))
and RITA ([Hesslow et al. RITA: a Study on Scaling Up Generative Protein Sequence Models.](https://arxiv.org/abs/2205.05789)).  See `mkultra/tuning.py` for implementation details.

## Training
Training scripts for RITA and ProtGPT2 are `RITA_prompt_tuning.py` and `ProtGPT2_prompt_tuning.py`, respectively.
They can be configured with parameters specified in a JSON config, see `training_configs/` folder and the
Trainer documentation in `mkultra/trainers.py`. You can enable memory tracking during training, but this makes the training slower.
The training configurations in `training_configs/` are the ones that we used for our paper. This includes the configs for training and evaluation, as well as those for runtime measuring and memory tracking.

An example notebook for training a prompt for RITA is `RITA_prompt_tuning_example.ipynb`, open in Colab [here](https://colab.research.google.com/github/AndreaNathansen/protein-prompt-tuning/blob/main/RITA_prompt_tuning_example.ipynb).

## Datasets
You can train the model with Fasta datasets if you use the `FastaDataset` class (`mkultra.sequence_loader.py`)
as dataset input for a PyTorch `DataLoader`.
Dataset preprocessing as in RITA can be done with the `prepare_dataset.ipynb` notebook in the `utils` folder.
Also, have a look into the script `utils/clustering.sh` to see the configuration we used to cluster our datasets using [MMseqs2](https://github.com/soedinglab/MMseqs2). The current setup of the dataset notebook and clustering script are for clustering with 100% sequence similarity threshold, but you can adjust that.

We also provide the datasets that we used for our experiments in the `datasets/` folder.
They contain sequences from the Pfam family PF03272.
The dataset (`datasets/InterProUniprotPF03272.fasta`) was downloaded from [InterPro](https://www.ebi.ac.uk/interpro/entry/pfam/PF03272/protein/UniProt/)
on January 5, 2023 ([Paysan-Lafosse et al. InterPro in 2022.](https://doi.org/10.1093/nar/gkac993)). Then, we removed all sequences containing an X, which created the dataset `InterProUniprotPF03272_Xremoved.fasta`.
We clustered our data with [MMseqs2](https://github.com/soedinglab/MMseqs2)([Steinegger et al. MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets.](https://doi.org/10.1038/nbt.3988) And: [Steinegger et al. Clustering huge protein sequence sets in linear time.](https://doi.org/10.1038/s41467-018-04964-5)). The datasets clustered with a sequence similarity threshold of 100%,
which were our main datasets, are `InterProUniprotPF03272prepared_<train|validation|test>.fasta`. Datasets clustered with different
thresholds can be found as `InterProUniprotPF03272_<threshold>_<train|validation|test>.fasta`

## Evaluation

### Perplexity
This implementation currently supports perplexity evaluation as perplexity per token.
See `RITA_prompt_comparison_basemodel.py` and `RITA_prompt_comparison_trainvaltest.py`, which are the evaluation scripts that
we used for our experiments. If you want to write an evaluation script yourself, have a look at `mkultra/evaluator.py`.

### Protein family prediction for generated sequences
We further evaluated the amount of generated sequences that were classified by ProtCNN ([Bileschi et al. Using deep learning to annotate the protein universe.](https://doi.org/10.1038/s41587-021-01179-w)) as belonging to our target family PF03272.
The script we used for that is `protcnn.py`, taken and adapted from the [official ProtCNN notebook](https://github.com/google-research/google-research/blob/master/using_dl_to_annotate_protein_universe/Using_Deep_Learning_to_Annotate_the_Protein_Universe.ipynb). As ProtCNN uses Tensorflow and we wanted to keep it separate from our other scripts, we created an extra conda environment for it, specified in `environment-protcnn.yml`. For using the script, you have to
download the ProtCNN model and vocabulary as described in the official notebook:
```
wget -qN https://storage.googleapis.com/brain-genomics-public/research/proteins/pfam/models/single_domain_per_sequence_zipped_models/seed_random_32.0/5356760.tar.gz
tar xzf 5356760.tar.gz
wget https://storage.googleapis.com/brain-genomics-public/research/proteins/pfam/models/single_domain_per_sequence_zipped_models/trained_model_pfam_32.0_vocab.json
```
You can set a fixed sliding window size and/or stride, or let the prediction be run on all possible windows of minimum size 50 (or the sequence length if shorter) for a sequence. Our code also supports running an ensemble of multiple ProtCNN models (as described in the ProtCNN paper), for this you have to modify the list of saved models in the script and download the respective additional models. Further, in single-model mode, you can set a probability threshold to discard predictions with a lower probability. In our final experiments, we used no fixed window size or stride, a single model, and a probability threshold of 0.5.

In addition to ProtCNN, we also evaluated protein family prediction with [HMMER](http://hmmer.org/), which you can run as follows:
```
hmmsearch --cut_ga --tblout <path/to/sequences/file>
```
The HMMER runs for all our sets of generated sequences are bundled in the script `hmmer_search.sh`.

### Activity filters for generated sequences
To reproduce our evaluations of protein activity (see [Johnson et al. Computational Scoring and Experimental Evaluation of Enzymes Generated by Neural Networks.](https://doi.org/10.1101/2023.03.04.531015)), run `activity_prediction_metrics/metrics.py` for all generated datasets. See also `activity_prediction_metrics/metrics.ipynb` for an interactive version.
Because these evaluations need specific dependencies, use an extra conda environment that is created with the same packages as specifed in the `environment.yml` file. Furthermore, the ESM likelihood computation requires the Github repo https://github.com/seanrjohnson/protein_gibbs_sampler. We advise to run the notebook `activity_prediction_metrics/metrics.ipynb` once initially, because it installs all required dependencies additional to the `environment.yml`.

Then, use `activity_prediction_metrics/aggregate.ipynb` to aggregate the results. This counts the activity predictions for each generated dataset and writes everything into `activity_prediction_metrics/activity.csv` for further usage, e.g. plotting.

## Sequence generation
For generating sequences, instantiate a prompt tuning model (see `mkultra/tuning.py`) and then load and add a prompt (`see mkultra/checkpoint_loader.py`) that was trained for that type of model, as for example in `RITA_prompt_sequence_generation.py`. In our experiments, we generated 193 sequences (size of our test set) in batches of 10.

## A note about finetuning
In our paper, we compare the performance of prompt-tuned models to that of finetuned models. For finetuning, we use the [run_clm.py script by Huggingface](https://github.com/huggingface/transformers/tree/v4.20.1/examples/pytorch/language-modeling) with the same batch sizes as for prompt tuning. You have to add `trust_remote_code=True` to the model loading call in line 376.
To generate the datasets as txt, you can use the `prepare_dataset.ipynb` notebook in the `utils` folder. 
