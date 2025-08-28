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


def plot_weights(pos_weight, neg_weights, labels, version, channel="inclusive"):
    true_weights = np.ones_like(pos_weight) / len(pos_weight)
    fake_weights = np.ones_like(neg_weights) / len(neg_weights)

    f, ax = plt.subplots(figsize=(9, 6))
    ax.hist(pos_weight, bins=100, range=[0, 1], alpha=.7, label=labels[1], color='#B22222',
            weights=true_weights)
    ax.hist(neg_weights, bins=100, range=[0, 1], alpha=.8, label=labels[2], color='#4169E1',
            weights=fake_weights)

    outdir = f"lightning_logs/version_{version}/plots_{channel}"
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


def obtain_tagging_power(df, version, signal):
    # Three cases
    # 1. Full on signal B
    # 2. Requirement of OS B exists
    # 3. additional requirement of OS B being B+/-
    def calculate_tagging_power(num_right, num_wrong, num_unclassified):
        efficiency = (num_right + num_wrong) / (num_right + num_wrong + num_unclassified)
        wrong_fra = num_wrong / (num_right + num_wrong)
        power = efficiency * (1 - 2 * wrong_fra) ** 2
        return wrong_fra, power

    def tagging_power_per_eta(df, eta_centers, eta_bins):
        num_right, num_wrong, num_unclassified = [], [], []

        for i in range(len(eta_centers)):
            in_bin = (df["eta"] >= eta_bins[i]) & (df["eta"] < eta_bins[i + 1])
            bin_df = df[in_bin]

            unclassified = len(bin_df) - np.sum((bin_df["ft_b_score"] > 0.5) | (bin_df["ft_bbar_score"] > 0.5))
            true_tag = np.sign(bin_df["B_id"])
            predicted_tag = np.argmax(bin_df[["ft_b_score", "ft_bbar_score"]], axis=1) * 2 - 1

            num_right.append(np.sum(true_tag == predicted_tag))
            num_wrong.append(np.sum(true_tag != predicted_tag))
            num_unclassified.append(unclassified)

        wrong_fra, power = calculate_tagging_power(
            np.array(num_right), np.array(num_wrong), np.array(num_unclassified)
        )
        combined_wrong_fra, combined_power = calculate_tagging_power(
            np.sum(num_right), np.sum(num_wrong), np.sum(num_unclassified)
        )

        return wrong_fra, power, (combined_wrong_fra, combined_power)

    def plot_tagging_power(eta_centers, eta_bins, power, wrong_fra, eta_dist, version, channel, label):
        xerr_lower = [center - eta_bins[i] for i, center in enumerate(eta_centers)]
        xerr_upper = [eta_bins[i + 1] - center for i, center in enumerate(eta_centers)]
        xerr = [xerr_lower, xerr_upper]

        outdir = f"lightning_logs/version_{version}/plots_{channel}"
        os.makedirs(outdir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(9, 6))
        weights = np.ones_like(eta_dist) / len(eta_dist)
        ax.hist(eta_dist, bins=eta_bins, label=r"Underlying $\eta$ distribution", color="grey", weights=weights / 2)
        ax.errorbar(eta_centers, power, xerr=xerr, fmt='o', color="red", label="Tagging Power")
        ax.errorbar(eta_centers, wrong_fra, xerr=xerr, fmt='o', color="blue", label="Wrong Fraction")
        ax.set_xlim(0, 0.5)
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel("[%]")
        ax.legend()
        plt.savefig(f"{outdir}/{label}.pdf")
        plt.savefig(f"{outdir}/{label}.png")
        plt.close()

    eta_centers = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55]
    eta_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1]

    df = df.copy()
    df["eta"] = 1 - np.max(df[["ft_b_score", "ft_bbar_score"]], axis=1)

    # Full tagging power
    full_df = df[df["SigMatch"] == 1]
    wrong_frac_full, power_full, combined_full = tagging_power_per_eta(full_df, eta_centers, eta_bins)
    plot_tagging_power(eta_centers, eta_bins, power_full, wrong_frac_full, full_df["eta"], version, signal,
                       "full_tagging_power")

    # Tagging power for only signal B in event and the case where OS exists
    event_ids, event_counts = np.unique(df["EventNumber"], return_counts=True)

    # Tagging power for only signal B in event
    single_b_df = df[df["EventNumber"].isin(event_ids[event_counts == 1])]
    signal_b_df = single_b_df[single_b_df["SigMatch"] == 1]

    wrong_frac_one_b, power_one_b, combined_one_b = tagging_power_per_eta(signal_b_df, eta_centers, eta_bins)
    plot_tagging_power(eta_centers, eta_bins, power_one_b, wrong_frac_one_b, signal_b_df["eta"], version, signal,
                       "only_signal_B_tagging_power")

    # Tagging power where there is an OS B
    two_b_df = df[df["EventNumber"].isin(event_ids[event_counts == 2])]
    signal_b_df = two_b_df[two_b_df["SigMatch"] == 1]

    wrong_frac_two_b, power_two_b, combined_two_b = tagging_power_per_eta(signal_b_df, eta_centers, eta_bins)
    plot_tagging_power(eta_centers, eta_bins, power_two_b, wrong_frac_two_b, signal_b_df["eta"], version, signal,
                       "os_b_exists_tagging_power")


    # Third case:
    is_bpm = np.abs(two_b_df["B_id"]) == 521
    two_b_bpm_df = two_b_df[two_b_df["EventNumber"].isin(two_b_df["EventNumber"][is_bpm])]

    os_bpm_df_sig = two_b_bpm_df[two_b_bpm_df["SigMatch"] == 1]
    wrong_frac_bpm, power_bpm, combined_bpm = tagging_power_per_eta(os_bpm_df_sig, eta_centers, eta_bins)
    plot_tagging_power(eta_centers, eta_bins, power_bpm, wrong_frac_bpm, os_bpm_df_sig["eta"], version, signal,
                       "os_bpm_exists_tagging_power")

    # For opposite-side Bpm
    os_bpm_df_non_sig = two_b_bpm_df[two_b_bpm_df["SigMatch"] == 0]
    wrong_frac_bpm_non, power_bpm_non, combined_bpm_non = tagging_power_per_eta(os_bpm_df_non_sig, eta_centers,
                                                                                eta_bins)
    plot_tagging_power(eta_centers, eta_bins, power_bpm_non, wrong_frac_bpm_non, os_bpm_df_non_sig["eta"], version,
                       signal, "os_bpm_tagging_power")
    # Sanity check: determine flavour from charge predictions
    pid = np.array(os_bpm_df_non_sig["final_pid"])
    B_pid = np.sign(np.array(os_bpm_df_non_sig["B_id"]))
    res = []
    for i in range(len(B_pid)):
        charge = np.sign(list(map(int, re.findall(r'-?\d+', pid[i]))))
        res.append(B_pid[i] == np.sum(charge))
    # Last sanity check: tagging power for the OS B+/- which are fully reco based on sum of charge
    charge_reco_bpm = os_bpm_df_non_sig[res]
    w_frac_bpm_cor_q, eff_bpm_cor_q, combined_bpm_cor_q = tagging_power_per_eta(charge_reco_bpm, eta_centers, eta_bins)
    plot_tagging_power(eta_centers, eta_bins, eff_bpm_cor_q, w_frac_bpm_cor_q, charge_reco_bpm["eta"], version,
                       signal, "os_bpm_tagging_power_true_charge")

    # also save some numbers
    with open(f"lightning_logs/version_{version}/info_{signal}_FT.txt", "w") as f:
        f.write(f"Full tagging power: {combined_full}\n")
        f.write(f"One B events: {np.sum(event_counts == 1)}\n")
        f.write(f"Two B events: {np.sum(event_counts == 2)}\n")
        f.write(f"One B tagging power: {combined_one_b}\n")
        f.write(f"Two B tagging power: {combined_two_b}\n")
        f.write(f"#OS B+/- events: {len(os_bpm_df_sig)}\n")
        f.write(f"OS B = B+/- tagging power (combined): {combined_bpm}\n")
        f.write(f"OS B+/- tagging power (combined): {combined_bpm_non}\n")
        f.write(f"Charge based B+/- correct: {np.sum(res) / len(res)}\n")
        f.write(f"Correct q by sum: {combined_bpm_cor_q}")
