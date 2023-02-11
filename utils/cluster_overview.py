from parse_clusters import parse_clusters

for i in range(101):
    clusters = parse_clusters(f"../datasets/clustered/InterProUniprotPF03272_{i}_all_seqs.fasta")
    if len(clusters) > 0:
        print(f"Minimum coverage:   {i}%")
        print(f"Number of Clusters: {len(clusters)}")
        lens = [len(cluster) for cluster in clusters]
        lens.sort()
        print(f"Cluster-sizes:      {lens}")
        print("------------------------------------")
