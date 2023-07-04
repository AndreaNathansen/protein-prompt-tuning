#!/bin/sh

# This script was used to create a randomly shuffled dataset as a 
# negative test set to detect false positives in our evaluation methods.
# It requires one parameter; the original fasta file with the sequences to be shuffled.
# We used our main test dataset.

esl-shuffle -o esl-shuffled.fasta -d --seed 42 $1