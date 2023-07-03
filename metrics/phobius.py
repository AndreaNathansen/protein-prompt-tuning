import requests
import pandas as pd
from io import StringIO
from Bio import SeqIO
import re

def scrape_phobius_for_transmembrane_domains(sequences):

    files = {
        'protfile': ('input.fasta', format_sequences_as_fasta_string(sequences)),
        'format': ('', 'short')
    }

    response = requests.post('https://phobius.sbc.su.se/cgi-bin/predict.pl', files=files)
    response_body = response.text

    data = re.findall(r'<pre>([\s\S]*)</pre>', response_body)[0]
    data = re.sub(' +', ' ', data) # remove multiple spaces
    data = data.replace('SEQENCE ID', 'SEQENCE_ID')

    df = pd.read_table(StringIO(data), delimiter=" ")

    return df['TM'].to_numpy() > 0

def format_sequences_as_fasta_string(sequences):
    out_handle = StringIO()
    SeqIO.write(sequences, out_handle, 'fasta')
    return out_handle.getvalue()

if __name__ == '__main__':
    print(scrape_phobius_for_transmembrane_domains())