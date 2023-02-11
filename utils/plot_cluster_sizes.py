import matplotlib.pyplot as plt
from parse_clusters import parse_clusters

bins = range(0, 1000, 50)

fig, axs = plt.subplots(5, 2, sharex=True, sharey=True, figsize=(5,15))

datasets = []

for i in range(10):
    datasets.append(parse_clusters(f"../datasets/clustered/UniprotPF03272_{(i)*10}__all_seqs.fasta"))     

for i in range(10):
    axs[int(i/2)][i%2].hist([len(clusters) for clusters in datasets[i]], bins=bins)
plt.savefig("hists.png")

for i in range(10):
    axs[int(i/2)][i%2].hist([len(clusters) for clusters in datasets[i]], bins=bins)
    axs[int(i/2)][i%2].set_yscale("log")
plt.savefig("hists_log.png")


fig, axs = plt.subplots()

data = [[len(clusters) for clusters in dataset] for dataset in datasets]
axs.boxplot(data)
plt.savefig("boxplots.png")

