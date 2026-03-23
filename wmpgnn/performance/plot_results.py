import re

import pandas as pd

from wmpgnn.performance.plotter import *


def metrics_eval(metrics_path, configs, version):
    log_dir = configs["log_dir"]
    model = configs["model"]

    # Removing empty row and so on
    metrics = pd.read_csv(metrics_path)
    metrics = metrics.groupby('epoch').agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None).reset_index()
    if configs["inference"].get("LCA", False) and model == "DFEI":
        plot_LCA_acc(metrics, version, log_dir=log_dir)

    loss_val = [
        match.group(1)
        for key in metrics.keys()
        if (match := re.fullmatch(r"train_(.+?)_loss", key))
    ]

    for loss in loss_val:
        plot_loss(metrics, version, loss, mode=model, log_dir=log_dir)


def plot_sig_pv_missasso(df, version, signal, log_dir="lightning_logs"):
    # Per track quantity
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
        _df.drop_duplicates(subset=['EventNumber'], keep='first')
        keys, counts = np.unique(_df["num_pvs"].values, return_counts=True)
        pv_log["npvs"] = dict(zip(keys, counts))

        for i in range(len(true_pv)):
            if npvs[i] not in pv_log["pv_total"].keys():
                pv_log["pv_corr_ml"][npvs[i]], pv_log["pv_corr_ip"][npvs[i]], pv_log["pv_total"][npvs[i]] = 0, 0, 0

            evt_true_pv = np.array(true_pv[i].split("_"), dtype=int)
            evt_pred_pv = np.array(pred_pv[i].split("_"), dtype=int)
            evt_minIP_pv = np.array(min_ip_pv[i].split("_"), dtype=int)
            pv_log["pv_corr_ml"][npvs[i]] += np.sum(evt_true_pv == evt_pred_pv)
            pv_log["pv_corr_ip"][npvs[i]] += np.sum(evt_true_pv == evt_minIP_pv)
            pv_log["pv_total"][npvs[i]] += evt_true_pv.shape[0]

        # write to disk
        label = f"signal_{selbool}" if selbool is not None else "signal_no_selection"
        temp = plot_pv_missasso(pv_log["pv_corr_ml"], pv_log["pv_corr_ip"], pv_log["pv_total"], pv_log["npvs"],
                                _version, _signal, label, log_dir=log_dir)
        return temp

    res = {"sig_no_selection": pv_asso(sig_df, version, signal),
           "sig_perfect_reco": pv_asso(sig_df, version, signal, "PerfectReco"),
           "sig_AllParticles": pv_asso(sig_df, version, signal, "AllParticles"),
           "sig_NoneIso": pv_asso(sig_df, version, signal, "NoneIso"),
           "sig_PartReco": pv_asso(sig_df, version, signal, "PartReco")}
    return res


def plot_sig_b_system_pv_missasso(df, version, signal, log_dir="lightning_logs"):
    # B level quantity
    if "inclusive" not in signal:
        sig_df = df[df["SigMatch"] == 1]
    else:
        sig_df = df
    sig_df = sig_df[sig_df["AllParticles"] == 1]
    true_pv, pred_pv, = sig_df["true_pv"].values, sig_df["pred_pv_b_lvl"].values
    npvs = sig_df["npvs"].values

    pv_log = {"pv_corr_ml": {}, "pv_total": {}}
    sig_df.drop_duplicates(subset=['EventNumber'], keep='first')
    keys, counts = np.unique(sig_df["num_pvs"].values, return_counts=True)
    pv_log["npvs"] = dict(zip(keys, counts))

    for i in range(len(true_pv)):
        if npvs[i] not in pv_log["pv_total"].keys():
            pv_log["pv_corr_ml"][npvs[i]], pv_log["pv_total"][npvs[i]] = 0, 0

        evt_true_pv = np.array(true_pv[i].split("_"), dtype=int)
        evt_pred_pv = int(pred_pv[i])
        pv_log["pv_corr_ml"][npvs[i]] += evt_pred_pv == evt_true_pv[0]
        pv_log["pv_total"][npvs[i]] += 1
    # write to disk
    label = f"signal_b_system"
    res = plot_pv_missasso(pv_log["pv_corr_ml"], None, pv_log["pv_total"], pv_log["npvs"],
                     version, signal, label, log_dir=log_dir)
    return res


def process_ft(df, sig_df, version, signal, log_dir="lightning_logs"):
    pattern = re.compile(r"bbar_ft_score_(\d+)")
    ft_layers = [int(match.group(1)) for k in df for match in [pattern.match(k)] if match]

    # Plot the node level output
    for i in ft_layers:
        bbar_score = 1 - df[f"bbar_ft_score_{i}"]  # optimal 0
        b_score = df[f"b_ft_score_{i}"]  # optimal 1
        plot_weights(b_score, bbar_score, [f"ft_decision_{i}", "b", "bbar"], version,
                     model="IFT", channel=signal, log_dir=log_dir, suffix='tagging_weights')

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
                     model="IFT", channel=signal, log_dir=log_dir, suffix='tagging_weights')

        # Plot the weights of the final state particles
        b_dec_final = np.array(
            [float(x) for item in sig_ch_df["final_b_score"][b_selbool].values for x in item.split(',')])
        bbar_dec_final = 1 - np.array(
            [float(x) for item in sig_ch_df["final_bbar_score"][bbar_selbool].values for x in item.split(',')])
        plot_weights(b_dec_final, bbar_dec_final, [f"signal_b_decision_final", "b", "bbar"], version,
                     model="IFT", channel=signal, log_dir=log_dir, suffix='tagging_weights')
    else:
        rem_B_df = sig_df[selbool]

    b_hadrons = [511, 521, 531]
    for b in b_hadrons:
        bbar_selbool = rem_B_df["B_id"] == b
        b_selbool = rem_B_df["B_id"] == -b
        b_dec = rem_B_df["ft_b_score"][b_selbool]
        bbar_dec = 1 - rem_B_df["ft_bbar_score"][bbar_selbool]
        plot_weights(b_dec, bbar_dec, [f"OS{b}_id_decision", "b", "bbar"], version,
                     model="IFT", channel=signal, log_dir=log_dir, suffix='tagging_weights')

        # Plot the weights of the final state particles
        b_dec_final = np.array(
            [float(x) for item in rem_B_df["final_b_score"][b_selbool].values for x in item.split(',')])
        bbar_dec_final = 1 - np.array(
            [float(x) for item in rem_B_df["final_bbar_score"][bbar_selbool].values for x in item.split(',')])
        plot_weights(b_dec_final, bbar_dec_final, [f"OS{b}_id_decision_final", "b", "bbar"], version,
                     model="IFT", channel=signal, log_dir=log_dir, suffix='tagging_weights')
