import os, re

import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.LHCb2)


def process_ft(df, sig_df, version, signal, log_dir="lightning_logs"):
    pattern = re.compile(r"bbar_ft_score_(\d+)")
    ft_layers = [int(match.group(1)) for k in df for match in [pattern.match(k)] if match]

    # Plot the node level output
    for i in ft_layers:
        bbar_score = 1 - df[f"bbar_ft_score_{i}"]  # optimal 0
        b_score = df[f"b_ft_score_{i}"]  # optimal 1
        plot_weights(b_score, bbar_score, [f"ft_decision_{i}", "b", "bbar"], version,
                     model="IFT", channel=signal, log_dir=log_dir)

    # Plot the B particle decision
    selbool = sig_df["AllParticles"] == 1
    has_signal = np.sum(sig_df["SigMatch"]) != 0
    if has_signal:
        sig_selbool = sig_df["SigMatch"] == 1
        sig_ch_df = sig_df[selbool * sig_selbool]
        rem_B_df = sig_df[selbool * ~sig_selbool]
        # Plotting signal B results
        bbar_selbool = np.sign(sig_ch_df["B_id"]) == 1
        b_selbool = np.sign(sig_ch_df["B_id"]) == -1
        b_dec = sig_ch_df["ft_b_score"][b_selbool]
        bbar_dec = 1 - sig_ch_df["ft_bbar_score"][bbar_selbool]
        plot_weights(b_dec, bbar_dec, [f"signal_b_id_decision", "b", "bbar"], version,
                     model="IFT", channel=signal, log_dir=log_dir)

        # Plot the weights of the final state particles
        b_dec_final = np.array(
            [float(x) for item in sig_ch_df["final_b_score"][b_selbool].values for x in item.split(',')])
        bbar_dec_final = 1 - np.array(
            [float(x) for item in sig_ch_df["final_bbar_score"][bbar_selbool].values for x in item.split(',')])
        plot_weights(b_dec_final, bbar_dec_final, [f"signal_b_decision_final", "b", "bbar"], version,
                     model="IFT", channel=signal, log_dir=log_dir)
    else:
        rem_B_df = sig_df[selbool]

    b_hadrons = [511, 521, 531]
    for b in b_hadrons:
        bbar_selbool = rem_B_df["B_id"] == b
        b_selbool = rem_B_df["B_id"] == -b
        b_dec = rem_B_df["ft_b_score"][b_selbool]
        bbar_dec = 1 - rem_B_df["ft_bbar_score"][bbar_selbool]
        plot_weights(b_dec, bbar_dec, [f"OS{b}_id_decision", "b", "bbar"], version,
                     model="IFT", channel=signal, log_dir=log_dir)

        # Plot the weights of the final state particles
        b_dec_final = np.array(
            [float(x) for item in rem_B_df["final_b_score"][b_selbool].values for x in item.split(',')])
        bbar_dec_final = 1 - np.array(
            [float(x) for item in rem_B_df["final_bbar_score"][bbar_selbool].values for x in item.split(',')])
        plot_weights(b_dec_final, bbar_dec_final, [f"OS{b}_id_decision_final", "b", "bbar"], version,
                     model="IFT", channel=signal, log_dir=log_dir)


def plot_weights(pos_weight, neg_weights, labels, version, model="DFEI", channel="inclusive", log_dir='lightning_logs'):
    true_weights = np.ones(pos_weight.shape[0]) / len(pos_weight)
    fake_weights = np.ones(neg_weights.shape[0]) / len(neg_weights)

    f, ax = plt.subplots(figsize=(9, 6))
    ax.hist(pos_weight, bins=100, range=[0, 1], alpha=.7, label=labels[1], color='#B22222',
            weights=true_weights
            )
    ax.hist(neg_weights, bins=100, range=[0, 1], alpha=.8, label=labels[2], color='#4169E1',
            weights=fake_weights
            )

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


def metrics_eval(metrics, configs, version, mode="DFEI", log_dir='lightning_logs'):
    if configs.get("LCA", False) and mode == "DFEI":
        plot_LCA_acc(metrics, version, log_dir=log_dir)

    loss_val = [
        match.group(1)
        for key in metrics.keys()
        if (match := re.fullmatch(r"train_(.+?)_loss", key))
    ]

    for loss in loss_val:
        plot_loss(metrics, version, loss, mode=mode, log_dir=log_dir)


def plot_pv_missasso(log, version, channel, selbool=None, log_dir='lightning_logs'):
    pv_asso_ml, pv_asso_ip, pv_asso_ntracks = log["pv_corr_ml"], log["pv_corr_ip"], log["pv_total"]
    npvs = np.array([])
    nPV_bins = []
    ml_mean, ml_err = [], []
    ip_mean, ip_err = [], []

    for key in pv_asso_ml.keys():
        nPV_bins.append(key)

        ml_mean.append(np.sum(pv_asso_ml[key]) / np.sum(pv_asso_ntracks[key]) * 100)
        ip_mean.append(np.sum(pv_asso_ip[key]) / np.sum(pv_asso_ntracks[key]) * 100)

        ml_err.append(ml_mean[-1] * np.sqrt(1 / np.sum(pv_asso_ml[key]) + 1 / np.sum(pv_asso_ntracks[key])))
        ip_err.append(ip_mean[-1] * np.sqrt(1 / np.sum(pv_asso_ip[key]) + 1 / np.sum(pv_asso_ntracks[key])))

        npvs = np.concatenate([npvs, np.ones(len(pv_asso_ml[key])) * key])
    f, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(nPV_bins, 100 - np.array(ml_mean), yerr=ml_err, fmt='.', color='red', label="HGNN")
    ax.errorbar(nPV_bins, 100 - np.array(ip_mean), yerr=ip_err, fmt='.', color='black', label="minIP")
    ax.hist(npvs, bins=15, range=(0.5, 15.5), alpha=.3, color='grey', weights=np.ones_like(npvs) / len(npvs) * 50)
    ax.set_ylabel("PV miss-association rate [%]", fontsize=28)
    ax.set_xlabel("# PVs [a.u.]", fontsize=28)
    ax.set_ylim([0, 15])
    ax.set_xlim([0, 16])
    ax.legend()

    outdir = f"{log_dir}/DFEI/version_{version}/plots_{channel}/pv"
    os.makedirs(outdir, exist_ok=True)

    if selbool is None:
        info_string = "all_tracks"
    else:
        info_string = "signal_" + selbool
    plt.savefig(f"{outdir}/{info_string}_pv_asso.pdf")
    plt.savefig(f"{outdir}/{info_string}_pv_asso.png")
    plt.close()


def plot_sig_pv_missasso(df, version, signal, log_dir="lightning_logs"):
    if "inclusive" not in signal:
        sig_df = df[df["SigMatch"] == 1]
    else:
        sig_df = df
    sig_df = sig_df[sig_df["NotFound"] != 1]

    def pv_asso(_df, _version, _signal, selbool=None):
        if selbool is not None:
            _df = _df[_df[selbool] == 1]
        true_pv, pred_pv, min_ip_pv = _df["true_pv"].values, _df["pred_pv"].values, _df["minIP_pv"].values
        npvs = _df["npvs"].values

        pv_log = {"pv_corr_ml": {}, "pv_corr_ip": {}, "pv_total": {}}

        for i in range(len(true_pv)):
            if npvs[i] not in pv_log["pv_total"].keys():
                pv_log["pv_corr_ml"][npvs[i]], pv_log["pv_corr_ip"][npvs[i]], pv_log["pv_total"][npvs[i]] = [], [], []

            evt_true_pv = np.array(true_pv[i].split("_"), dtype=int)
            evt_pred_pv = np.array(pred_pv[i].split("_"), dtype=int)
            evt_minIP_pv = np.array(min_ip_pv[i].split("_"), dtype=int)
            pv_log["pv_corr_ml"][npvs[i]].append(np.sum(evt_true_pv == evt_pred_pv))
            pv_log["pv_corr_ip"][npvs[i]].append(np.sum(evt_true_pv == evt_minIP_pv))
            pv_log["pv_total"][npvs[i]].append(evt_true_pv.shape[0])
        plot_pv_missasso(pv_log, _version, _signal, selbool if selbool is not None else "no_selection", log_dir=log_dir)

    pv_asso(sig_df, version, signal, "PerfectReco")
    pv_asso(sig_df, version, signal, "AllParticles")
    pv_asso(sig_df, version, signal, "NoneIso")
    pv_asso(sig_df, version, signal, "PartReco")
    pv_asso(sig_df, version, signal)





