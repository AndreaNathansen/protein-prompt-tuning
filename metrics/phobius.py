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
    
    if (response.status_code != 200):
        raise RuntimeError(f"Invalid response: status code {response.status_code}")

    matches = re.findall(r'<pre>([\s\S]*)</pre>', response_body)

    if (len(matches) == 0):
        raise RuntimeError("Response contains no results")

    data = matches[0]
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