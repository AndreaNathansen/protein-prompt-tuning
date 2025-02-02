"""
The script used to plot number of clusters and size of largest cluster for different sequence identity thresholds.
See ./clustering.sh for the parameters we used for clustering with mmseqs2.
"""

import matplotlib.pyplot as plt
from parse_clusters import parse_clusters

number_of_clusters=[]
largest_cluster=[]

for i in range(101):
    clusters = parse_clusters(f"../datasets/clustered/InterProUniprotPF03272_{i}_all_seqs.fasta")
    number_of_clusters.append(len(clusters))
    cluster_sizes = [len(c) for c in clusters]
    cluster_sizes.sort()
    largest_cluster.append(cluster_sizes[-1])

fig, ax = plt.subplots()
ax.plot(range(101), number_of_clusters, label="Number of clusters")
ax.plot(range(101), largest_cluster, label="Size of largest cluster")

ax.set(xticks=range(0, 101, 5), xlabel="Sequence identity threshold")
ax.grid(visible=True)
ax.legend()
plt.savefig("cluster_sizes.png")
#plt.show()
