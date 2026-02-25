import os

import pandas as pd
import numpy as np
import torch


def acc_four_class(pred, label):
    pred_argmax = torch.argmax(pred, dim=1)

    res = {}
    for i in range(4):
        classi_selbool = label == i

        res[f"LCA_class{i}_num"] = torch.sum(classi_selbool).float().item()
        if res[f"LCA_class{i}_num"] == 0:
            res[f"LCA_class{i}_pred_class0"] = 0.
            res[f"LCA_class{i}_pred_class1"] = 0.
            res[f"LCA_class{i}_pred_class2"] = 0.
            res[f"LCA_class{i}_pred_class3"] = 0.
        else:
            res[f"LCA_class{i}_pred_class0"] = torch.sum(pred_argmax[classi_selbool] == 0).item() / res[
                f"LCA_class{i}_num"]
            res[f"LCA_class{i}_pred_class1"] = torch.sum(pred_argmax[classi_selbool] == 1).item() / res[
                f"LCA_class{i}_num"]
            res[f"LCA_class{i}_pred_class2"] = torch.sum(pred_argmax[classi_selbool] == 2).item() / res[
                f"LCA_class{i}_num"]
            res[f"LCA_class{i}_pred_class3"] = torch.sum(pred_argmax[classi_selbool] == 3).item() / res[
                f"LCA_class{i}_num"]
    return res


def calculate_accuracy(df):
    nevents = len(df)
    all_particles = np.sum(df["AllParticles"])
    all_particles_ratio = all_particles / nevents * 100
    all_particles_ratio_err = np.nan if all_particles == 0 else all_particles_ratio * np.sqrt(
        1 / all_particles + 1 / nevents)
    perfect_reco = np.sum(df["PerfectReco"])
    perfect_reco_ratio = perfect_reco / nevents * 100
    perfect_reco_ratio_err = np.nan if perfect_reco == 0 else perfect_reco_ratio * np.sqrt(
        1 / perfect_reco + 1 / nevents)
    none_iso = np.sum(df["NoneIso"])
    none_iso_ratio = none_iso / nevents * 100
    none_iso_ratio_err = np.nan if none_iso == 0 else none_iso_ratio * np.sqrt(1 / none_iso + 1 / nevents)
    part_reco = np.sum(df["PartReco"])
    part_reco_ratio = part_reco / nevents * 100
    part_reco_ratio_err = np.nan if part_reco == 0 else part_reco_ratio * np.sqrt(1 / part_reco + 1 / nevents)

    return [(all_particles_ratio, all_particles_ratio_err), (perfect_reco_ratio, perfect_reco_ratio_err), (
        none_iso_ratio, none_iso_ratio_err), (part_reco_ratio, part_reco_ratio_err)]


def write_to_file(file, cond, performance, entries=None, label=None):
    all_part, perfect_reco, none_iso, part_reco = performance
    with open(file, cond) as f:
        f.write("=" * 30 + "\n")
        f.write(f"Reconstruction efficiency{label} ({entries}): \n")
        f.write(f"all_particles : {all_part[0]:.2f} +/- {all_part[1]:.2f} \n")
        f.write(f"perfect_reco  : {perfect_reco[0]:.2f} +/- {perfect_reco[1]:.2f} \n")
        f.write(f"none_iso      : {none_iso[0]:.2f} +/- {none_iso[1]:.2f} \n")
        f.write(f"part_reco     : {part_reco[0]:.2f} +/- {part_reco[1]:.2f} \n")


def obtain_reco_accuracy(df, version, signal, log_dir, model):
    # Full signal
    # Full OS inclusive
    # Signal single B events
    # Signal two B events
    # Signal more than two B events
    file = f"{log_dir}/{model}/version_{version}/info_{signal}_reco.txt"
    cond = "a" if os.path.exists(file) else "w"
    with open(file, cond) as f:
        f.write("=" * 50 + "\n")
        f.write("Reconstruction performance \n")

    evtnumber, counts = np.unique(df["EventNumber"], return_counts=True)

    if "inclusive" not in signal:
        sig_df = df[df["SigMatch"] == 1]
        bkg_df = df[df["SigMatch"] != 1]
    else:
        sig_df = df
        bkg_df = None

    performance = calculate_accuracy(sig_df)
    write_to_file(file, "a", performance, entries=len(sig_df), label="")

    if bkg_df is not None:
        performance = calculate_accuracy(bkg_df)
        write_to_file(file, "a", performance, entries=len(bkg_df), label=" inclusive")

    single_evts = evtnumber[counts == 1]
    usage_df = sig_df[sig_df["EventNumber"].isin(single_evts)]
    performance = calculate_accuracy(usage_df)
    write_to_file(file, "a", performance, entries=len(usage_df), label=" only 1 B in event")

    two_evts = evtnumber[counts == 2]
    usage_df = sig_df[sig_df["EventNumber"].isin(two_evts)]
    performance = calculate_accuracy(usage_df)
    write_to_file(file, "a", performance, entries=len(usage_df), label=" only 2 B in event")

    more_evts = evtnumber[counts > 2]
    usage_df = sig_df[sig_df["EventNumber"].isin(more_evts)]
    performance = calculate_accuracy(usage_df)
    write_to_file(file, "a", performance, entries=len(usage_df), label=" more than 2 B in event")

    # bias checking
    usage_df = sig_df[np.sign(sig_df["B_id"]) == -1]
    performance = calculate_accuracy(usage_df)
    write_to_file(file, "a", performance, entries=len(usage_df), label=" negative id")
    usage_df = sig_df[np.sign(sig_df["B_id"]) == 1]
    performance = calculate_accuracy(usage_df)
    write_to_file(file, "a", performance, entries=len(usage_df), label=" positive id")


def acc_pv_asso(pv_perf, version, signal, log_dir, model):
    file = f"{log_dir}/{model}/version_{version}/info_{signal}_reco.txt"
    cond = "a" if os.path.exists(file) else "w"
    with open(file, cond) as f:
        f.write("=" * 50 + "\n")
        f.write("PV association performance \n")

        for key, item in pv_perf.items():
            f.write("=" * 30 + "\n")
            f.write(key + ": \n")
            ntracks, pred, ip = item

            pred_perf = pred / ntracks * 100
            pred_perf_err = pred_perf * np.sqrt(1 / pred + 1 / ntracks)
            f.write(f"HGNN association  : {100 - pred_perf:.2f} +/- {pred_perf_err:.2f} \n")

            if ip is not None:
                ip_perf = ip / ntracks * 100
                ip_perf_err = ip_perf * np.sqrt(1 / ip + 1 / ntracks)
                f.write(f"minIP association : {100 - ip_perf:.2f} +/- {ip_perf_err:.2f} \n")


if __name__ == "__main__":
    df = pd.read_csv(
        "/eos/user/y/yukaiz/DFEI_IFT/IFT_training/wmpgnn/analysis/LHCb_logs/IFT/version_1/signal_df_00299103_Bs_Jpsiphi.csv")
    obtain_reco_accuracy(df, "-1", "Bs_Jpsiphi", "lightning_logs", "IFT")
