for i in $(seq 0 1 2);
do
    hmmsearch --cut_ga --tblout experiment_results/hmmer_results/hmmer_s_prompt-tuned_$i.out PF03272.hmm generated_sequences/prompt-tuning-clustered-100-RITA_s-fromvocab-True-seed-$i-generated.fasta
    hmmsearch --cut_ga --tblout experiment_results/hmmer_results/hmmer_m_prompt-tuned_$i.out PF03272.hmm generated_sequences/prompt-tuning-clustered-100-RITA_m-fromvocab-True-seed-$i-generated.fasta
    hmmsearch --cut_ga --tblout experiment_results/hmmer_results/hmmer_l_prompt-tuned_$i.out PF03272.hmm generated_sequences/prompt-tuning-clustered-100-RITA_l-fromvocab-True-seed-$i-generated.fasta
    hmmsearch --cut_ga --tblout experiment_results/hmmer_results/hmmer_xl_prompt-tuned_$i.out PF03272.hmm generated_sequences/prompt-tuning-clustered-100-RITA_xl-fromvocab-True-seed-$i-generated.fasta
done

hmmsearch --cut_ga --tblout experiment_results/hmmer_results/hmmer_s_basemodel.out PF03272.hmm generated_sequences/basemodel-RITA_s-generated.fasta
hmmsearch --cut_ga --tblout experiment_results/hmmer_results/hmmer_m_basemodel.out PF03272.hmm generated_sequences/basemodel-RITA_m-generated.fasta
hmmsearch --cut_ga --tblout experiment_results/hmmer_results/hmmer_l_basemodel.out PF03272.hmm generated_sequences/basemodel-RITA_l-generated.fasta
hmmsearch --cut_ga --tblout experiment_results/hmmer_results/hmmer_xl_basemodel.out PF03272.hmm generated_sequences/basemodel-RITA_xl-generated.fasta

hmmsearch --cut_ga --tblout experiment_results/hmmer_results/hmmer_s_finetuned.out PF03272.hmm generated_sequences/finetuned-RITA_s-generated.fasta
hmmsearch --cut_ga --tblout experiment_results/hmmer_results/hmmer_m_finetuned.out PF03272.hmm generated_sequences/finetuned-RITA_m-generated.fasta
hmmsearch --cut_ga --tblout experiment_results/hmmer_results/hmmer_l_finetuned.out PF03272.hmm generated_sequences/finetuned-RITA_l-generated.fasta
hmmsearch --cut_ga --tblout experiment_results/hmmer_results/hmmer_xl_finetuned.out PF03272.hmm generated_sequences/finetuned-RITA_xl-generated.fasta
hmmsearch --cut_ga --tblout experiment_results/hmmer_results/testset.out PF03272.hmm datasets/InterProUniprotPF03272prepared_test.fasta