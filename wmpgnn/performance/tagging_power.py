import os

import pandas as pd

try:
    from wmpgnn.performance.tagging_power_classes import *
except ImportError:
    from tagging_power_classes import *



def write_tagging_power(file, metric, label):
    file.write(f"{'='*20}\n")
    file.write(f"{label}\n")
    file.write(f"Tagging power  : ({metric.power[0]:.4f} +/- {metric.power[1]:.4f})%\n")
    file.write(f"Wrong fraction : ({metric.wrong_fraction[0]:.4f} +/- {metric.wrong_fraction[1]:.4f})%\n")
    file.write(f"Epsilon        : ({metric.epsilon[0]:.4f} +/- {metric.epsilon[1]:.4f})%\n")
    file.write(f"Dsquared       : ({metric.d_squared[0]:.4f} +/- {metric.d_squared[1]:.4f})%\n")


def write_correctness_ratio(file, ratio, frag_results):
    file.write(f"Correctness ratio: ({ratio[0] * 100:.2f} +/- {ratio[1] * 100:.2f})%\n")
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
        analyzer.plot_tagging_power(metrics_full, full_df["eta"], "tagging_power")
        label = f"Tagging performance for {signal} ({len(full_df)}):"
        write_tagging_power(f, metrics_full, label)

        """2. Single B events"""
        single_b_df = classifier.filter_n_b_events(df, 1, event_ids, event_counts)
        signal_single_b = single_b_df[single_b_df["SigMatch"] == 1]
        metrics_single = analyzer.compute_tagging_power_per_eta(signal_single_b)
        analyzer.plot_tagging_power(metrics_single, signal_single_b["eta"], "singleB_tagging_power")
        label = f"1 B candidate present in event ({np.sum(event_counts == 1)}):"
        write_tagging_power(f, metrics_single, label)
        # Obtaining correctness ratio and with regard to fragmentation tracks
        ratio = calc.calculate_prediction_correctness(signal_single_b)
        frag_results = calc.calculate_by_fragmentation(signal_single_b)
        write_correctness_ratio(f, ratio, frag_results)

        """3. Two B events (opposite-side tagging possible)"""
        two_b_df = classifier.filter_n_b_events(df, 2, event_ids, event_counts)
        signal_two_b = two_b_df[two_b_df["SigMatch"] == 1]
        metrics_two = analyzer.compute_tagging_power_per_eta(signal_two_b)
        analyzer.plot_tagging_power(metrics_two, signal_two_b["eta"], "doubleB_tagging_power")
        label = f"2 B candidate present in event ({np.sum(event_counts == 2)}):"
        write_tagging_power(f, metrics_two, label)
        ratio = calc.calculate_prediction_correctness(signal_two_b)
        frag_results = calc.calculate_by_fragmentation(signal_two_b)
        write_correctness_ratio(f, ratio, frag_results)

        """4. B+/- specific analysis"""
        bpm_df = classifier.filter_bpm_events(two_b_df)

        # Looking at the performance if the Signal has an B+/- as an OS B
        bpm_signal = bpm_df[bpm_df["SigMatch"] == 1]
        metrics_bpm_sig = analyzer.compute_tagging_power_per_eta(bpm_signal)
        analyzer.plot_tagging_power(metrics_bpm_sig, bpm_signal["eta"], "doubleB_ospm_tagging_power")
        label = f"Signal has OS B+/- Events: {len(bpm_signal)}"
        write_tagging_power(f, metrics_bpm_sig, label)

        # The performance of the OS B+/- itself
        bpm_os = bpm_df[bpm_df["SigMatch"] == 0]
        metrics_bpm_os = analyzer.compute_tagging_power_per_eta(bpm_os)
        analyzer.plot_tagging_power(metrics_bpm_os, bpm_os["eta"], "osB_pm_tagging_power")
        charge_frac, charge_correct_df = calc.verify_charge_reconstruction(bpm_os)
        label = f"OS B+/- Events: {len(bpm_os)}"
        write_tagging_power(f, metrics_bpm_os, label)
        f.write(f"B+/- charge reconstruction correctness: {charge_frac * 100:.2f}%\n")

        # If based on sum of charge it matches the OS B+/- charge
        metrics_charge = analyzer.compute_tagging_power_per_eta(charge_correct_df)
        analyzer.plot_tagging_power(metrics_charge, charge_correct_df["eta"],"osB_pm_true_charge_tagging_power")
        label = "Charged correct OS B+/- Events:"
        write_tagging_power(f, metrics_charge, label)


        """5. Bias study of signal"""
        neg_df = full_df[np.sign(full_df["B_id"]) == -1]
        metrics_full = analyzer.compute_tagging_power_per_eta(neg_df)
        analyzer.plot_tagging_power(metrics_full, full_df["eta"], f"neg_tagging_power")
        label = f"Tagging performance for negative {signal} ({len(neg_df)}):"
        write_tagging_power(f, metrics_full, label)

        pos_df = full_df[np.sign(full_df["B_id"]) == 1]
        metrics_full = analyzer.compute_tagging_power_per_eta(pos_df)
        analyzer.plot_tagging_power(metrics_full, full_df["eta"], f"pos_tagging_power")
        label = f"Tagging performance for positive {signal} ({len(pos_df)}):"
        write_tagging_power(f, metrics_full, label)


if __name__ == "__main__":
    df = pd.read_csv(
        "/eos/user/y/yukaiz/DFEI_IFT/IFT_training/wmpgnn/analysis/LHCb_logs/IFT/version_1/signal_df_00299103_Bs_Jpsiphi.csv")
    analyze_tagging_power(df, "-1", "Bs_Jpsiphi")
