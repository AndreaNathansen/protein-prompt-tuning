#!/bin/sh

if [ ! -d "../datasets/clustered" ]
then
    mkdir ../datasets/clustered
fi

mmseqs easy-cluster ../datasets/InterProUniprotPF03272_Xremoved.fasta ../datasets/clustered/InterProUniprotPF03272 ./tmp --min-seq-id 1

# Use these commands to additionally create the datasets clustered with 35%, 65% and 95% sequence identity
#mmseqs easy-cluster ../datasets/InterProUniprotPF03272_Xremoved.fasta ../datasets/clustered/InterProUniprotPF03272_95 ./tmp --min-seq-id 0.95
#mmseqs easy-cluster ../datasets/InterProUniprotPF03272_Xremoved.fasta ../datasets/clustered/InterProUniprotPF03272_65 ./tmp --min-seq-id 0.65
#mmseqs easy-cluster ../datasets/InterProUniprotPF03272_Xremoved.fasta ../datasets/clustered/InterProUniprotPF03272_35 ./tmp --min-seq-id 0.35
