import os

import numpy as np

import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.LHCb2)


def plot_weights(pos_weight, neg_weights, labels, version):
    true_weights = np.ones_like(pos_weight) / len(pos_weight)
    fake_weights = np.ones_like(neg_weights) / len(neg_weights)

    f, ax = plt.subplots(figsize=(9, 6))
    ax.hist(pos_weight, bins=100, range=[0, 1], alpha=.7, label=labels[1], color='#B22222',
            weights=true_weights)
    ax.hist(neg_weights, bins=100, range=[0, 1], alpha=.8, label=labels[2], color='#4169E1',
            weights=fake_weights)

    outdir = f"lightning_logs/version_{version}/plots"
    os.makedirs(outdir, exist_ok=True)

    ax.set_xlabel("NN weights [a.u.]")
    ax.set_ylabel("Normalized entries [a.u.]")
    ax.legend()
    ax.set_yscale("log")
    plt.savefig(f"{outdir}/{labels[0]}.pdf")
    plt.savefig(f"{outdir}/{labels[0]}.png")
    plt.close()
