import os

import pandas as pd

try:
    from wmpgnn.performance.tagging_power_classes import *
except ImportError:
    from tagging_power_classes import *


def write_tagging_power(file, metric, label, per_event=None):
    file.write(f"{'=' * 20}\n")
    file.write(f"{label}\n")
    file.write(f"Tagging power  : ({metric.power[0]:.4f} +/- {metric.power[1]:.4f})%\n")
    if per_event is not None:
        file.write(f"per event      : ({per_event:.4f})%\n")
    file.write(f"Wrong fraction : ({metric.wrong_fraction[0]:.4f} +/- {metric.wrong_fraction[1]:.4f})%\n")
    file.write(f"Epsilon        : ({metric.epsilon[0]:.4f} +/- {metric.epsilon[1]:.4f})%\n")
    file.write(f"Dsquared       : ({metric.d_squared[0]:.4f} +/- {metric.d_squared[1]:.4f})%\n")


def write_correctness_ratio(file, ratio, frag_results=None):
    file.write(f"Correctness ratio: ({ratio[0] * 100:.2f} +/- {ratio[1] * 100:.2f})%\n")
    if frag_results is not None:
        file.write(f"With fragmentation: ({frag_results['with_frag'][0] * 100:.2f} +/- "
                   f"{frag_results['with_frag'][1] * 100:.2f})%\n")
        file.write(f"Without fragmentation: ({frag_results['without_frag'][0] * 100:.2f} +/- "
                   f"{frag_results['without_frag'][1] * 100:.2f})%\n")


def analyze_tagging_power(df: pd.DataFrame, version: str, signal: str, log_dir: str = "lightning_logs"):
    """
    Main function for tagging power values and plots

    Args:
        df: DataFrame with B meson tagging data
        version: Analysis version identifier
        signal: Signal type identifier
        log_dir: Logging directory
    """
    analyzer = TaggingPowerAnalyzer(version, signal, log_dir)
    classifier = EventClassifier()
    calc = CorrectnessCalculator()

    # Prepare data
    df = analyzer.process_df(df)
    event_ids, event_counts = classifier.get_event_counts(df)

    file = f"{log_dir}/IFT/version_{version}/info_{signal}_FT.txt"

    os.makedirs(os.path.dirname(file), exist_ok=True)
    cond = "a" if os.path.exists(file) else "w"

    with open(file, cond) as f:
        f.write("=" * 50 + "\n")
        f.write("FLAVOUR TAGGING POWER \n")

        """1. Full tagging power (all signal-matched events)"""
        full_df = df[df["SigMatch"] == 1]
        metrics_full = analyzer.compute_tagging_power_per_eta(full_df)
        analyzer.plot_tagging_power(metrics_full, full_df["eta"], "eta", "tagging_power")
        analyzer.plot_tagging_power(metrics_full, full_df["num_pvs"], "npvs", "npvs_tagging_power")
        per_event_tagging_power = analyzer.compute_tagging_power_per_event(full_df)
        label = f"Tagging performance for {signal} ({len(full_df)}):"
        write_tagging_power(f, metrics_full, label, per_event=per_event_tagging_power)

        """2. Single B events"""
        single_b_df = classifier.filter_n_b_events(df, 1, event_ids, event_counts)
        signal_single_b = single_b_df[single_b_df["SigMatch"] == 1]
        metrics_single = analyzer.compute_tagging_power_per_eta(signal_single_b)
        analyzer.plot_tagging_power(metrics_single, signal_single_b["eta"], "eta", "singleB_tagging_power")
        per_event_tagging_power = analyzer.compute_tagging_power_per_event(single_b_df)
        label = f"1 B candidate present in event ({np.sum(event_counts == 1)}):"
        write_tagging_power(f, metrics_single, label, per_event=per_event_tagging_power)
        # Obtaining correctness ratio and with regard to fragmentation tracks
        ratio = calc.calculate_prediction_correctness(signal_single_b)
        frag_results = calc.calculate_by_fragmentation(signal_single_b)
        write_correctness_ratio(f, ratio, frag_results)

        """3. Two B events (opposite-side tagging possible)"""  # The OS can be not found, two B is on true level
        two_b_df = classifier.filter_n_b_events(df, 2, event_ids, event_counts)
        signal_two_b = two_b_df[two_b_df["SigMatch"] == 1]
        metrics_two = analyzer.compute_tagging_power_per_eta(signal_two_b)
        analyzer.plot_tagging_power(metrics_two, signal_two_b["eta"], "eta", "doubleB_tagging_power")
        per_event_tagging_power = analyzer.compute_tagging_power_per_event(two_b_df)
        label = f"2 B candidate present in event ({np.sum(event_counts == 2)}):"
        write_tagging_power(f, metrics_two, label, per_event=per_event_tagging_power)
        ratio = calc.calculate_prediction_correctness(signal_two_b)
        frag_results = calc.calculate_by_fragmentation(signal_two_b)
        write_correctness_ratio(f, ratio, frag_results)

        """4. B+/- specific analysis"""
        bpm_df = classifier.filter_bpm_events(two_b_df)

        # Looking at the performance if the Signal has an B+/- as an OS B
        bpm_signal = bpm_df[bpm_df["SigMatch"] == 1]
        metrics_bpm_sig = analyzer.compute_tagging_power_per_eta(bpm_signal)
        analyzer.plot_tagging_power(metrics_bpm_sig, bpm_signal["eta"], "eta", "doubleB_ospm_tagging_power")
        per_event_tagging_power = analyzer.compute_tagging_power_per_event(bpm_signal)
        label = f"Signal has OS B+/- Events: {len(bpm_signal)}"
        write_tagging_power(f, metrics_bpm_sig, label, per_event=per_event_tagging_power)

        # The performance of the OS B+/- itself
        bpm_os = bpm_df[bpm_df["SigMatch"] == 0]
        metrics_bpm_os = analyzer.compute_tagging_power_per_eta(bpm_os)
        analyzer.plot_tagging_power(metrics_bpm_os, bpm_os["eta"], "eta", "osB_pm_tagging_power")
        per_event_tagging_power = analyzer.compute_tagging_power_per_event(bpm_os)
        charge_frac, charge_correct_df = calc.verify_charge_reconstruction(bpm_os)
        label = f"OS B+/- Events: {len(bpm_os)}"
        write_tagging_power(f, metrics_bpm_os, label, per_event=per_event_tagging_power)
        f.write(f"B+/- charge reconstruction correctness: {charge_frac * 100:.2f}%\n")

        # If based on sum of charge it matches the OS B+/- charge
        metrics_charge = analyzer.compute_tagging_power_per_eta(charge_correct_df)
        analyzer.plot_tagging_power(metrics_charge, charge_correct_df["eta"], "eta", "osB_pm_true_charge_tagging_power")
        per_event_tagging_power = analyzer.compute_tagging_power_per_event(charge_correct_df)
        label = "Charged correct OS B+/- Events:"
        write_tagging_power(f, metrics_charge, label, per_event=per_event_tagging_power)

        """5. Bias study of signal"""
        # negative mc id
        neg_df = full_df[np.sign(full_df["B_id"]) == -1]
        metrics_full = analyzer.compute_tagging_power_per_eta(neg_df)
        analyzer.plot_tagging_power(metrics_full, neg_df["eta"], "eta", f"neg_tagging_power")
        per_event_tagging_power = analyzer.compute_tagging_power_per_event(neg_df)
        label = f"Tagging performance for negative {signal} ({len(neg_df)}):"
        write_tagging_power(f, metrics_full, label, per_event=per_event_tagging_power)
        ratio = calc.calculate_prediction_correctness(neg_df)
        write_correctness_ratio(f, ratio)

        neg_single_bool = (single_b_df["SigMatch"] == 1) & (np.sign(single_b_df["B_id"]) == -1)
        neg_sig_df = single_b_df[neg_single_bool]
        metrics_full = analyzer.compute_tagging_power_per_eta(neg_sig_df)
        analyzer.plot_tagging_power(metrics_full, neg_sig_df["eta"], "eta", f"neg_single_tagging_power")
        per_event_tagging_power = analyzer.compute_tagging_power_per_event(neg_sig_df)
        label = f"Tagging performance for negative single B {signal} ({len(neg_sig_df)}):"
        write_tagging_power(f, metrics_full, label, per_event=per_event_tagging_power)
        ratio = calc.calculate_prediction_correctness(neg_sig_df)
        write_correctness_ratio(f, ratio)

        neg_double_bool = (two_b_df["SigMatch"] == 1) & (np.sign(two_b_df["B_id"]) == -1)
        neg_sig_df = two_b_df[neg_double_bool]
        metrics_full = analyzer.compute_tagging_power_per_eta(neg_sig_df)
        analyzer.plot_tagging_power(metrics_full, neg_sig_df["eta"], "eta", f"neg_double_tagging_power")
        per_event_tagging_power = analyzer.compute_tagging_power_per_event(neg_sig_df)
        label = f"Tagging performance for negative double B {signal} ({len(neg_sig_df)}):"
        write_tagging_power(f, metrics_full, label, per_event=per_event_tagging_power)
        ratio = calc.calculate_prediction_correctness(neg_sig_df)
        write_correctness_ratio(f, ratio)

        # pos mc id
        pos_df = full_df[np.sign(full_df["B_id"]) == 1]
        metrics_full = analyzer.compute_tagging_power_per_eta(pos_df)
        analyzer.plot_tagging_power(metrics_full, pos_df["eta"], "eta", f"pos_tagging_power")
        per_event_tagging_power = analyzer.compute_tagging_power_per_event(pos_df)
        label = f"Tagging performance for positive {signal} ({len(pos_df)}):"
        write_tagging_power(f, metrics_full, label, per_event=per_event_tagging_power)
        ratio = calc.calculate_prediction_correctness(pos_df)
        write_correctness_ratio(f, ratio)

        pos_single_bool = (single_b_df["SigMatch"] == 1) & (np.sign(single_b_df["B_id"]) == 1)
        pos_sig_df = single_b_df[pos_single_bool]
        metrics_full = analyzer.compute_tagging_power_per_eta(pos_sig_df)
        analyzer.plot_tagging_power(metrics_full, pos_sig_df["eta"], "eta", f"pos_single_tagging_power")
        per_event_tagging_power = analyzer.compute_tagging_power_per_event(pos_sig_df)
        label = f"Tagging performance for positive single B {signal} ({len(pos_sig_df)}):"
        write_tagging_power(f, metrics_full, label, per_event=per_event_tagging_power)
        ratio = calc.calculate_prediction_correctness(pos_sig_df)
        write_correctness_ratio(f, ratio)

        pos_double_bool = (two_b_df["SigMatch"] == 1) & (np.sign(two_b_df["B_id"]) == 1)
        pos_sig_df = two_b_df[pos_double_bool]
        metrics_full = analyzer.compute_tagging_power_per_eta(pos_sig_df)
        analyzer.plot_tagging_power(metrics_full, pos_sig_df["eta"], "eta", f"pos_double_tagging_power")
        per_event_tagging_power = analyzer.compute_tagging_power_per_event(pos_sig_df)
        label = f"Tagging performance for positive double B {signal} ({len(pos_sig_df)}):"
        write_tagging_power(f, metrics_full, label, per_event=per_event_tagging_power)
        ratio = calc.calculate_prediction_correctness(pos_sig_df)
        write_correctness_ratio(f, ratio)

        """7. Performance based on OS reconstruction type"""
        os_incl_df = two_b_df[two_b_df["SigMatch"] != 1]
        os_incl_df.loc[os_incl_df["PerfectReco"] == 1, "AllParticles"] = 0
        sig_df = two_b_df[two_b_df["SigMatch"] == 1]
        conditions = ["PerfectReco", "AllParticles", "NoneIso", "PartReco", "NotFound"]
        for condition in conditions:
            evts = os_incl_df[os_incl_df[condition] == 1]["EventNumber"]
            usage_df = sig_df[sig_df["EventNumber"].isin(evts)]
            metrics_two = analyzer.compute_tagging_power_per_eta(usage_df)
            analyzer.plot_tagging_power(metrics_two, usage_df["eta"], "eta", f"doubleB_OS{condition}_tagging_power")
            per_event_tagging_power = analyzer.compute_tagging_power_per_event(usage_df)
            label = f"2 B candidate present in event with OS being {condition} ({len(usage_df)}):"
            write_tagging_power(f, metrics_two, label, per_event=per_event_tagging_power)
            ratio = calc.calculate_prediction_correctness(usage_df)
            frag_results = calc.calculate_by_fragmentation(usage_df)
            write_correctness_ratio(f, ratio, frag_results)


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

if __name__ == "__main__":
    df = pd.read_csv(
        "/eos/user/y/yukaiz/DFEI_IFT/IFT_training/wmpgnn/analysis/LHCb_logs/IFT/version_1/signal_df_00299103_Bs_Jpsiphi.csv")
    analyze_tagging_power(df, "-1", "Bs_Jpsiphi")
