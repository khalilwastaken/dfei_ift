import os, re

import numpy as np

import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.LHCb2)


def process_ft(df, sig_df, version, signal):
    pattern = re.compile(r"bbar_ft_score_(\d+)")
    ft_layers = [int(match.group(1)) for k in df for match in [pattern.match(k)] if match]

    # Plot the node level output
    for i in ft_layers:
        bbar_score = 1 - df[f"bbar_ft_score_{i}"]  # optimal 0
        b_score = df[f"b_ft_score_{i}"]  # optimal 1
        plot_weights(b_score, bbar_score, [f"ft_decision_{i}", "b", "bbar"], version, channel=signal)

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
        plot_weights(b_dec, bbar_dec, [f"signal_b_id_decision", "b", "bbar"], version, channel=signal)

        # Plot the weights of the final state particles
        b_dec_final = np.array(
            [float(x) for item in sig_ch_df["final_b_score"][b_selbool].values for x in item.split(',')])
        bbar_dec_final = 1 - np.array(
            [float(x) for item in sig_ch_df["final_bbar_score"][bbar_selbool].values for x in item.split(',')])
        plot_weights(b_dec_final, bbar_dec_final, [f"signal_b_decision_final", "b", "bbar"], version, channel=signal)
    else:
        rem_B_df = sig_df[selbool]

    b_hadrons = [511, 521, 531]
    for b in b_hadrons:
        bbar_selbool = rem_B_df["B_id"] == b
        b_selbool = rem_B_df["B_id"] == -b
        b_dec = rem_B_df["ft_b_score"][b_selbool]
        bbar_dec = 1 - rem_B_df["ft_bbar_score"][bbar_selbool]
        plot_weights(b_dec, bbar_dec, [f"{b}_id_decision", "b", "bbar"], version, channel=signal)

        # Plot the weights of the final state particles
        b_dec_final = np.array(
            [float(x) for item in rem_B_df["final_b_score"][b_selbool].values for x in item.split(',')])
        bbar_dec_final = 1 - np.array(
            [float(x) for item in rem_B_df["final_bbar_score"][bbar_selbool].values for x in item.split(',')])
        plot_weights(b_dec_final, bbar_dec_final, [f"{b}_id_decision_final", "b", "bbar"], version, channel=signal)


def plot_weights(pos_weight, neg_weights, labels, version, model="DFEI", channel="inclusive"):
    true_weights = np.ones_like(pos_weight) / len(pos_weight)
    fake_weights = np.ones_like(neg_weights) / len(neg_weights)

    f, ax = plt.subplots(figsize=(9, 6))
    ax.hist(pos_weight, bins=100, range=[0, 1], alpha=.7, label=labels[1], color='#B22222',
            weights=true_weights)
    ax.hist(neg_weights, bins=100, range=[0, 1], alpha=.8, label=labels[2], color='#4169E1',
            weights=fake_weights)

    outdir = f"lightning_logs/{model}/version_{version}/plots_{channel}"
    os.makedirs(outdir, exist_ok=True)

    ax.set_xlabel("NN weights [a.u.]")
    ax.set_ylabel("Normalized entries [a.u.]")
    ax.legend()
    ax.set_yscale("log")
    plt.savefig(f"{outdir}/{labels[0]}.pdf")
    plt.savefig(f"{outdir}/{labels[0]}.png")
    plt.close()


def plot_LCA_acc(df, version, channel="inclusive"):
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
    outdir = f"lightning_logs/version_{version}/plots_{channel}"
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


def plot_loss(df, version, loss):
    trn_LCA_loss = np.array(df[f"train_{loss}_loss"])
    val_LCA_loss = np.array(df[f"val_{loss}_loss"])
    epochs = np.arange(len(trn_LCA_loss))

    # Plot dir
    outdir = f"lightning_logs/version_{version}/plots"
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


def obtain_reco_accuracy(df, version, signal):
    sig_df = df[df["SigMatch"] == 1]

    nevents = len(sig_df)
    all_particles = np.sum(sig_df["AllParticles"])
    all_particles_ratio = all_particles / nevents * 100
    all_particles_ratio_err = np.nan if all_particles == 0 else all_particles_ratio * np.sqrt(
        1 / all_particles + 1 / nevents)
    perfect_reco = np.sum(sig_df["PerfectReco"])
    perfect_reco_ratio = perfect_reco / nevents * 100
    perfect_reco_ratio_err = np.nan if perfect_reco == 0 else perfect_reco_ratio * np.sqrt(
        1 / perfect_reco + 1 / nevents)
    none_iso = np.sum(sig_df["NoneIso"])
    none_iso_ratio = none_iso / nevents * 100
    none_iso_ratio_err = np.nan if none_iso == 0 else none_iso_ratio * np.sqrt(1 / none_iso + 1 / nevents)
    part_reco = np.sum(sig_df["PartReco"])
    part_reco_ratio = part_reco / nevents * 100
    part_reco_ratio_err = np.nan if part_reco == 0 else part_reco_ratio * np.sqrt(1 / part_reco + 1 / nevents)

    file = f"lightning_logs/DFEI/version_{version}/info_{signal}_FT.txt"
    cond = "a" if os.path.exists(file) else "w"
    with open(file, cond) as f:
        f.write("=" * 30 + "\n")
        f.write("reconstruction efficiency: \n")
        f.write(f"all_particles: {all_particles_ratio} +/- {all_particles_ratio_err} \n")
        f.write(f"perfect_reco: {perfect_reco_ratio} +/- {perfect_reco_ratio_err} \n")
        f.write(f"none_iso: {none_iso_ratio} +/- {none_iso_ratio_err} \n")
        f.write(f"part_reco: {part_reco_ratio} +/- {part_reco_ratio_err} \n")
