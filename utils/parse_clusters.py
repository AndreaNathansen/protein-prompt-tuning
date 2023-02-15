from Bio import SeqIO
import os

class Cluster:
    def __init__(self, representative_id: str):
        self.representative_id = representative_id
        self.members = []
    
    def __len__(self):
        return len(self.members)

    def add_member(self, member: str):
        self.members.append(member)

def parse_clusters(filename):
    """
    Parses the *_all_seqs.fasta file output of mmseqs2 for further use.
    The complete filename is required as parameter.
    The representative member of a cluster is always the first member.
    """
    clusters = []
    cluster = None

    if os.path.isfile(filename):
        for record in SeqIO.parse(filename, "fasta"):
            id = record.id.split("|")[0]
            is_cluster = len(record) == 0
            if (is_cluster and cluster is not None):
                clusters.append(cluster)
            if (is_cluster):
                cluster = Cluster(id)
            else:
                cluster.add_member(record)
        clusters.append(cluster)
        
    return clusters
