if [ ! -d "../datasets/clustered" ]
then
    mkdir ../datasets/clustered
fi

mmseqs easy-cluster ../datasets/InterProUniprotPF03272_Xremoved.fasta ../datasets/clustered/InterProUniprotPF03272 ./tmp --min-seq-id 1
