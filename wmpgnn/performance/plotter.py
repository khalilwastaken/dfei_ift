import os

import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.LHCb2)


def plot_weights(pos_weight, neg_weights, labels, version, model="DFEI", channel="inclusive", log_dir='lightning_logs',
                 suffix=None):
    true_weights = np.ones(pos_weight.shape[0]) / len(pos_weight)
    fake_weights = np.ones(neg_weights.shape[0]) / len(neg_weights)

    f, ax = plt.subplots(figsize=(9, 6))
    ax.hist(pos_weight, bins=100, range=[0, 1], alpha=.7, label=labels[1], color='#B22222',
            weights=true_weights
            )
    ax.hist(neg_weights, bins=100, range=[0, 1], alpha=.8, label=labels[2], color='#4169E1',
            weights=fake_weights
            )

    if suffix is None:
        suffix = "/"
        if "nodes" in labels[0]:
            suffix = "nodes"
        elif "edges" in labels[0]:
            suffix = "edges"
        elif "pv" in labels[0]:
            suffix = "pv"

    outdir = f"{log_dir}/{model}/version_{version}/plots_{channel}/{suffix}"
    os.makedirs(outdir, exist_ok=True)

    ax.set_xlabel("NN weights [a.u.]")
    ax.set_ylabel("Normalized entries [a.u.]")
    ax.legend()
    ax.set_yscale("log")
    plt.savefig(f"{outdir}/{labels[0]}.pdf")
    plt.savefig(f"{outdir}/{labels[0]}.png")
    plt.close()


def plot_roc_curve(sig, bkg, plt_label, version, model="DFEI", channel="inclusive", log_dir='lightning_logs'):
    pred = np.concatenate([sig, bkg])
    labels = np.concatenate([
        np.ones(len(sig)),
        np.zeros(len(bkg))
    ])
    fpr, tpr, th = roc_curve(labels, pred)
    auc_score = roc_auc_score(labels, pred)
    rnd_class = np.linspace(0, 1, 100)

    suffix = "/"
    if "nodes" in plt_label[0]:
        suffix = "nodes"
    elif "edges" in plt_label[0]:
        suffix = "edges"
    elif "pv" in plt_label[0]:
        suffix = "pv"

    outdir = f"{log_dir}/{model}/version_{version}/plots_{channel}/{suffix}"
    os.makedirs(outdir, exist_ok=True)

    f, ax = plt.subplots(figsize=(9, 6))
    ax.plot(fpr, tpr, label=f'AUC = {auc_score:.5f}', color="black", alpha=1)
    ax.plot(rnd_class, rnd_class, '--', label='Rnd classifier', color="grey", alpha=.8)
    ax.set_xlabel(r'$\epsilon_{bkg}$ - FPR')
    ax.set_ylabel(r'$\epsilon_{s}$ - TPR')
    ax.legend(fontsize='medium')

    plt.savefig(f"{outdir}/{plt_label[0]}.pdf")
    plt.savefig(f"{outdir}/{plt_label[0]}.png")
    plt.close()


def plot_LCA_acc(df, version, log_dir='lightning_logs'):
    trn_LCA_acc0 = np.array(df["train_LCA_class0_pred_class0"])
    trn_LCA_acc1 = np.array(df["train_LCA_class1_pred_class1"])
    trn_LCA_acc2 = np.array(df["train_LCA_class2_pred_class2"])
    trn_LCA_acc3 = np.array(df["train_LCA_class3_pred_class3"])

    val_LCA_acc0 = np.array(df["val_LCA_class0_pred_class0"])
    val_LCA_acc1 = np.array(df["val_LCA_class1_pred_class1"])
    val_LCA_acc2 = np.array(df["val_LCA_class2_pred_class2"])
    val_LCA_acc3 = np.array(df["val_LCA_class3_pred_class3"])

    epochs = np.arange(len(trn_LCA_acc0))

    # Plot dir
    outdir = f"{log_dir}/DFEI/version_{version}/plots"
    os.makedirs(outdir, exist_ok=True)

    # Plot LCA acc
    f, ax = plt.subplots(figsize=(9, 6))
    ax.plot(epochs, trn_LCA_acc0, color="black", label="LCA=0")
    ax.plot(epochs, val_LCA_acc0, color="black", linestyle='dashed')

    ax.plot(epochs, trn_LCA_acc1, color="blue", label="LCA=1")
    ax.plot(epochs, val_LCA_acc1, color="blue", linestyle='dashed')

    ax.plot(epochs, trn_LCA_acc2, color="red", label="LCA=2")
    ax.plot(epochs, val_LCA_acc2, color="red", linestyle='dashed')

    ax.plot(epochs, trn_LCA_acc3, color="green", label="LCA=3")
    ax.plot(epochs, val_LCA_acc3, color="green", linestyle='dashed')

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy [%]")
    ax.legend()
    plt.savefig(f"{outdir}/LCA_acc.pdf")
    plt.savefig(f"{outdir}/LCA_acc.png")
    plt.close()


def plot_loss(df, version, loss, mode="DFEI", log_dir='lightning_logs'):
    trn_LCA_loss = np.array(df[f"train_{loss}_loss"])
    val_LCA_loss = np.array(df[f"val_{loss}_loss"])
    epochs = np.arange(len(trn_LCA_loss))

    # Plot dir
    outdir = f"{log_dir}/{mode}/version_{version}/plots"
    os.makedirs(outdir, exist_ok=True)

    # Plot combined loss
    f, ax = plt.subplots(figsize=(9, 6))
    ax.plot(epochs, trn_LCA_loss, color="#4169E1", label="trn loss")
    ax.plot(epochs, val_LCA_loss, color="#B22222", label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend()
    plt.savefig(f"{outdir}/{loss}_loss.pdf")
    plt.savefig(f"{outdir}/{loss}_loss.png")
    plt.close()


def plot_pv_missasso(pv_asso_ml, pv_asso_ip, pv_asso_ntracks, log_npvs,
                     version, channel, selbool=None, log_dir='lightning_logs'):
    npvs = np.array([])
    nPV_bins = []
    ml_mean, ml_err = [], []
    ip_mean, ip_err = [], []

    for key in pv_asso_ml.keys():
        nPV_bins.append(key)
        ml_mean.append(100 - pv_asso_ml[key] / pv_asso_ntracks[key] * 100)
        ml_err.append(
            pv_asso_ml[key] / pv_asso_ntracks[key] * np.sqrt(1 / pv_asso_ml[key] + 1 / pv_asso_ntracks[key]) * 100)
        if pv_asso_ip is not None:
            ip_mean.append(100 - pv_asso_ip[key] / pv_asso_ntracks[key] * 100)
            ip_err.append(
                pv_asso_ip[key] / pv_asso_ntracks[key] * np.sqrt(1 / pv_asso_ip[key] + 1 / pv_asso_ntracks[key]) * 100)

        npvs = np.concatenate([npvs, np.ones(log_npvs[key]) * key])

    f, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(nPV_bins, np.array(ml_mean), yerr=ml_err, fmt='.', color='red', label="DFEI")
    if pv_asso_ip is not None:
        ax.errorbar(nPV_bins, np.array(ip_mean), yerr=ip_err, fmt='.', color='black', label="minIP")
    ax.hist(npvs, bins=15, range=(0.5, 15.5), alpha=.3, color='grey', weights=np.ones_like(npvs) / len(npvs) * 50)
    ax.set_ylabel("PV miss-association rate [%]", fontsize=28)
    ax.set_xlabel("# PVs [a.u.]", fontsize=28)
    ax.set_ylim([0, 30])
    ax.set_xlim([0, 16])
    ax.legend()

    outdir = f"{log_dir}/DFEI/version_{version}/plots_{channel}/pv"
    os.makedirs(outdir, exist_ok=True)

    if selbool is None:
        info_string = "all_tracks"
    else:
        info_string = selbool
    plt.savefig(f"{outdir}/{info_string}_pv_asso.pdf")
    plt.savefig(f"{outdir}/{info_string}_pv_asso.png")
    plt.close()

    # Getting the absolute numbers
    total_tracks = sum(pv_asso_ntracks.values())
    total_pred = sum(pv_asso_ml.values())
    total_ip = None
    if pv_asso_ip is not None:
        total_ip = sum(pv_asso_ip.values())
    return total_tracks, total_pred, total_ip
