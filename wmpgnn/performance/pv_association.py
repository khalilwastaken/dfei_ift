from collections import defaultdict
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.LHCb2)


def pv_asso(_df, b_lvl=False):
    _df = _df.drop_duplicates(subset=['EventNumber'], keep='first')
    npvs = _df["npvs"].values
    keys, counts = np.unique(npvs, return_counts=True)
    pv_log = {"pv_corr_ml": defaultdict(int), "pv_corr_ip": defaultdict(int), "pv_total": defaultdict(int),
              "npvs": dict(zip(keys, counts))}

    if b_lvl:  # Looking at a B as a whole
        true_pv = _df["true_pv_b_lvl"].values
        pred_pv, min_ip_pv = _df["pred_pv_b_lvl"].values, _df["minIP_pv_b_lvl"].values
        for pv in keys:
            selbool = npvs == pv
            pv_log["pv_corr_ml"][pv] = np.sum(pred_pv[selbool] == true_pv[selbool])
            pv_log["pv_corr_ip"][pv] = np.sum(min_ip_pv[selbool] == true_pv[selbool])
            pv_log["pv_total"][pv] = np.sum(selbool)
    else:  # On track level
        true_pv = _df["true_pv"].values
        pred_pv, min_ip_pv = _df["pred_pv"].values, _df["minIP_pv"].values
        parse = lambda s: np.fromstring(s, dtype=int, sep="_")
        for i, npv in enumerate(npvs):
            evt_true = parse(true_pv[i])
            evt_pred = parse(pred_pv[i])
            evt_minIP = parse(min_ip_pv[i])
            pv_log["pv_corr_ml"][npv] += np.sum(evt_true == evt_pred)
            pv_log["pv_corr_ip"][npv] += np.sum(evt_true == evt_minIP)
            pv_log["pv_total"][npv] += len(evt_true)
    return pv_log


def plot_sig_b_pv_missasso(df, version, signal, log_dir="lightning_logs"):
    # B level quantity
    if "inclusive" not in signal:
        sig_df = df[df["SigMatch"] == 1]
    else:
        sig_df = df
    sig_df = sig_df[sig_df["AllParticles"] == 1]
    log = pv_asso(sig_df, b_lvl=True)
    res = plot_pv_missasso(log, version, signal, label='sig_b_lvl', log_dir=log_dir)
    return res


def plot_sig_tracks_pv_missasso(df, version, signal, log_dir="lightning_logs"):
    # Per track quantity
    if "inclusive" not in signal:
        sig_df = df[df["SigMatch"] == 1]
    else:
        sig_df = df
    sig_df = sig_df[sig_df["NotFound"] != 1]

    selections = {
        "sig_tracks_no_selection": None,
        "sig_tracks_perfect_reco": "PerfectReco",
        "sig_tracks_AllParticles": "AllParticles",
        "sig_tracks_NoneIso": "NoneIso",
        "sig_tracks_PartReco": "PartReco",
    }

    return {
        key: plot_pv_missasso(pv_asso(sig_df if sel is None else sig_df[sig_df[sel] == 1]),
                              version, signal, label=key, log_dir=log_dir
                              )
        for key, sel in selections.items()
    }


def plot_pv_missasso(log, version, channel, label=None, log_dir='lightning_logs'):
    def _missasso_stats(asso, ntracks):
        mean = 100 - asso / ntracks * 100
        err = asso / ntracks * np.sqrt(1 / asso + 1 / ntracks) * 100
        return mean, err

    pv_asso_ml, pv_asso_ip, pv_asso_ntracks = log["pv_corr_ml"], log["pv_corr_ip"], log["pv_total"]
    log_npvs = log["npvs"]

    nPV_bins = list(pv_asso_ml.keys())
    try:
        ml_mean, ml_err = zip(*[_missasso_stats(pv_asso_ml[k], pv_asso_ntracks[k]) for k in nPV_bins])
        ip_mean, ip_err = zip(*[_missasso_stats(pv_asso_ip[k], pv_asso_ntracks[k]) for k in nPV_bins])
    except ValueError:
        return (
            sum(pv_asso_ntracks.values()),
            sum(pv_asso_ml.values()),
            sum(pv_asso_ip.values())
        )
    npvs = np.array([v for k in nPV_bins for v in [k] * log_npvs[k]])

    f, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(nPV_bins, ml_mean, yerr=ml_err, fmt='.', color='red', label="DFEI")
    ax.errorbar(nPV_bins, ip_mean, yerr=ip_err, fmt='.', color='black', label="minIP")
    ax.hist(npvs, bins=15, range=(0.5, 15.5), alpha=.3, color='grey',
            weights=np.ones_like(npvs) / len(npvs) * 50)
    ax.set(ylabel="PV miss-association rate [%]", xlabel="# PVs [a.u.]",
           ylim=[0, 30], xlim=[0, 16])
    ax.legend()

    outdir = f"{log_dir}/DFEI/version_{version}/plots_{channel}/pv/pv_miss_asso"
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        plt.savefig(f"{outdir}/pv_asso_{label}.{ext}")
    plt.close()

    return (
        sum(pv_asso_ntracks.values()),
        sum(pv_asso_ml.values()),
        sum(pv_asso_ip.values())
    )


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

            try:
                pred_perf = 100 - pred / ntracks * 100
                pred_perf_err = pred / ntracks * np.sqrt(1 / pred + 1 / ntracks) * 100
            except ZeroDivisionError:
                pred_perf = 0
                pred_perf_err = 100
            f.write(f"HGNN association  : {pred_perf:.2f} +/- {pred_perf_err:.2f} \n")

            if ip is not None:
                try:
                    ip_perf = 100 - ip / ntracks * 100
                    ip_perf_err = ip / ntracks * np.sqrt(1 / ip + 1 / ntracks) * 100
                except ZeroDivisionError:
                    ip_perf = 0
                    ip_perf_err = 100
                f.write(f"minIP association : {ip_perf:.2f} +/- {ip_perf_err:.2f} \n")
