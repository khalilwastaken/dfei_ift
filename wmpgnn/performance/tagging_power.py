import numpy as np
import pandas as pd

import re
import os
from dataclasses import dataclass
from typing import Tuple, List

import matplotlib.pyplot as plt
import mplhep as hep
import warnings

warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in divide",
    category=RuntimeWarning
)

hep.style.use(hep.style.LHCb2)


@dataclass
class TaggingMetrics:
    """Container for tagging performance metrics with uncertainties."""
    wrong_fraction: np.ndarray
    wrong_fraction_err: np.ndarray
    power: np.ndarray
    power_err: np.ndarray
    combined_wrong_fraction: float
    combined_wrong_fraction_err: float
    combined_power: float
    combined_power_err: float


class TaggingPowerAnalyzer:
    """Analyzes flavor tagging performance across different event configurations."""

    def __init__(self, version, channel, eta_centers: List[float] = None, eta_bins: List[float] = None):
        self.eta_centers = eta_centers or [0.05, 0.15, 0.25, 0.35, 0.45, 0.55]
        self.eta_bins = eta_bins or [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1]
        x_err_lower = [center - self.eta_bins[i] for i, center in enumerate(self.eta_centers)]
        x_err_upper = [self.eta_bins[i + 1] - center for i, center in enumerate(self.eta_centers)]
        self.x_err = [x_err_lower, x_err_upper]

        self.outdir = f"lightning_logs/IFT/version_{version}/plots_{channel}"
        os.makedirs(self.outdir, exist_ok=True)

    @staticmethod
    def calculate_tagging_power(num_right: np.ndarray, num_wrong: np.ndarray,
                                num_unclassified: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate tagging efficiency and power metrics."""
        total_classified = num_right + num_wrong
        total_events = total_classified + num_unclassified

        efficiency = total_classified / total_events
        defficiency = efficiency * np.sqrt(1 / total_classified + 1 / total_events)

        wrong_fraction = num_wrong / total_classified
        dwrong_fraction = wrong_fraction * (1 / np.sqrt(num_wrong) + 1 / np.sqrt(total_classified))

        power = efficiency * (1 - 2 * wrong_fraction) ** 2
        dp_deff = (1 - 2 * wrong_fraction) ** 2
        dp_dw = -4 * efficiency * (1 - 2 * wrong_fraction)
        dpower = np.sqrt((dp_deff * defficiency) ** 2 + (dp_dw * dwrong_fraction) ** 2)

        return wrong_fraction * 100, dwrong_fraction * 100, power * 100, dpower * 100

    @staticmethod
    def process_df(df: pd.DataFrame) -> pd.DataFrame:
        """Process df to contain only signal. Add eta column based on maximum score."""
        sig_selbool = df["SigMatch"] == 1
        reco_selbool = df["AllParticles"] == 1
        evts = np.unique(df[sig_selbool * reco_selbool]["EventNumber"])
        df = df[df["EventNumber"].isin(evts)]

        df = df.copy()
        df["eta"] = 1 - np.max(df[["ft_b_score", "ft_bbar_score"]], axis=1)
        return df

    def _calculate_per_eta_bin(self, df: pd.DataFrame) -> Tuple[List, List, List]:
        """Calculate tagging statistics for each eta bin."""
        num_right, num_wrong, num_unclassified = [], [], []

        for i in range(len(self.eta_centers)):
            in_bin = (df["eta"] >= self.eta_bins[i]) & (df["eta"] < self.eta_bins[i + 1])
            bin_df = df[in_bin]

            # Count unclassified events
            classified_mask = (bin_df["ft_b_score"] > 0.5) | (bin_df["ft_bbar_score"] > 0.5)
            unclassified = len(bin_df) - np.sum(classified_mask)

            # Determine true and predicted tags
            true_tag = np.sign(bin_df["B_id"])
            predicted_tag = np.argmax(bin_df[["ft_b_score", "ft_bbar_score"]], axis=1) * 2 - 1

            num_right.append(np.sum(true_tag == predicted_tag))
            num_wrong.append(np.sum(true_tag != predicted_tag))
            num_unclassified.append(unclassified)

        return num_right, num_wrong, num_unclassified

    def compute_tagging_power_per_eta(self, df: pd.DataFrame) -> TaggingMetrics:
        """Compute tagging power across eta bins."""
        num_right, num_wrong, num_unclassified = self._calculate_per_eta_bin(df)

        # Per-bin metrics
        wrong_frac, wrong_frac_err, power, power_err = self.calculate_tagging_power(
            np.array(num_right), np.array(num_wrong), np.array(num_unclassified)
        )

        # Combined metrics
        combined_wrong_frac, combined_wrong_frac_err, combined_power, combined_power_err = self.calculate_tagging_power(
            np.sum(num_right), np.sum(num_wrong), np.sum(num_unclassified)
        )

        return TaggingMetrics(wrong_frac, wrong_frac_err, power, power_err, combined_wrong_frac,
                              combined_wrong_frac_err, combined_power, combined_power_err)

    def plot_tagging_power(self, metrics: TaggingMetrics, eta_dist: np.ndarray,
                           label: str = "tagging_power"):
        """Plot tagging power/wrong fraction vs misstag probability."""
        fig, ax = plt.subplots(figsize=(9, 6))

        # Plot underlying distribution
        weights = np.ones_like(eta_dist) / len(eta_dist)
        ax.hist(eta_dist, bins=self.eta_bins,
                label=r"Underlying $\eta$ distribution",
                color="grey", weights=weights / 2 * 100)

        # Plot metrics
        ax.errorbar(self.eta_centers, metrics.power, xerr=self.x_err, yerr=metrics.power_err,
                    fmt='o', color="red", label="Tagging Power")
        ax.errorbar(self.eta_centers, metrics.wrong_fraction, xerr=self.x_err,
                    yerr=metrics.wrong_fraction_err,
                    fmt='o', color="blue", label="Wrong Fraction")

        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 105)
        ax.set_xlabel(r"Predicted mistag $\eta$")
        ax.set_ylabel("[%]")
        ax.legend()
        plt.savefig(f"{self.outdir}/{label}.pdf")
        plt.savefig(f"{self.outdir}/{label}.png")
        plt.close()


class EventClassifier:
    """Classifies events based on B meson configuration."""

    @staticmethod
    def get_event_counts(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Get unique event IDs and their B meson counts."""
        return np.unique(df["EventNumber"], return_counts=True)

    @staticmethod
    def filter_n_b_events(df: pd.DataFrame, n_b_cand: int, event_ids: np.ndarray,
                          event_counts: np.ndarray) -> pd.DataFrame:
        """Filter events with exactly n B meson."""
        return df[df["EventNumber"].isin(event_ids[event_counts == n_b_cand])]

    @staticmethod
    def filter_bpm_events(df: pd.DataFrame) -> pd.DataFrame:
        """Filter events containing B+/- mesons (PDG ID 521)."""
        is_bpm = np.abs(df["B_id"]) == 521
        return df[df["EventNumber"].isin(df["EventNumber"][is_bpm])]


class CorrectnessCalculator:
    """Calculate correctness ratios and uncertainties."""

    @staticmethod
    def calculate_prediction_correctness(df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate correctness ratio and uncertainty for predictions."""
        pred_tag = np.argmax(df[["ft_b_score", "ft_bbar_score"]], axis=1) * 2 - 1
        true_tag = np.sign(df["B_id"])

        num_correct = np.sum(pred_tag == true_tag)
        num_total = len(true_tag)
        ratio = num_correct / num_total

        try:
            uncertainty = ratio * np.sqrt(1 / num_correct + 1 / num_total)
        except:
            uncertainty = -1

        return ratio, uncertainty

    @staticmethod
    def calculate_by_fragmentation(df: pd.DataFrame) -> dict:
        """Calculate correctness separately for events with/without fragmentation."""
        pred_tag = np.argmax(df[["ft_b_score", "ft_bbar_score"]], axis=1) * 2 - 1
        true_tag = np.sign(df["B_id"])

        has_frag = df["frags"].notna() & (df["frags"] != "")

        results = {}
        for label, mask in [("with_frag", has_frag), ("without_frag", ~has_frag)]:
            num_correct = np.sum(pred_tag[mask] == true_tag[mask])
            num_total = len(true_tag[mask])
            ratio = num_correct / num_total
            uncertainty = ratio * np.sqrt(1 / num_correct + 1 / num_total)
            results[label] = (ratio, uncertainty)

        return results

    @staticmethod
    def verify_charge_reconstruction(df: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
        """Verify B flavor determination from summed charges."""
        pid_strings = np.array(df["final_pid"])
        true_flavor = np.sign(np.array(df["B_id"]))

        charge_matches = []
        for i, (pid_str, true_flav) in enumerate(zip(pid_strings, true_flavor)):
            try:
                charges = np.sign(list(map(int, re.findall(r'-?\d+', pid_str))))
                charge_matches.append(true_flav == np.sum(charges))
            except:
                charge_matches.append(False)

        charge_correct_fraction = np.sum(charge_matches) / len(charge_matches)
        charge_correct_df = df[charge_matches]

        return charge_correct_fraction, charge_correct_df


def analyze_tagging_power(df: pd.DataFrame, version: str, signal: str, log_dir: str = "lightning_logs"):
    """
    Main function for tagging power values and plots

    Args:
        df: DataFrame with B meson tagging data
        version: Analysis version identifier
        signal: Signal type identifier
        log_dir: Logging directory
    """
    analyzer = TaggingPowerAnalyzer(version, signal)
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

        # 1. Full tagging power (all signal-matched events)
        full_df = df[df["SigMatch"] == 1]
        metrics_full = analyzer.compute_tagging_power_per_eta(full_df)
        analyzer.plot_tagging_power(metrics_full, full_df["eta"], "full_tagging_power")

        f.write(
            f"\nFull wrong fraction: ({metrics_full.combined_wrong_fraction:.4f} +/- {metrics_full.combined_wrong_fraction_err:.4f})%\n")
        f.write(f"Full tagging power: ({metrics_full.combined_power:.4f} +/- {metrics_full.combined_power_err:.4f})%\n")

        # 2. Single B events
        single_b_df = classifier.filter_n_b_events(df, 1, event_ids, event_counts)
        signal_single_b = single_b_df[single_b_df["SigMatch"] == 1]

        metrics_single = analyzer.compute_tagging_power_per_eta(signal_single_b)
        analyzer.plot_tagging_power(metrics_single, signal_single_b["eta"], "singleB_tagging_power")

        ratio, ratio_err = calc.calculate_prediction_correctness(signal_single_b)
        try:
            frag_results = calc.calculate_by_fragmentation(signal_single_b)
        except:
            frag_results = {'with_frag': [-1, -1], "without_frag": [-1, -1]}

        f.write(f"\nSingle B Events: {np.sum(event_counts == 1)}\n")
        f.write(f"Correctness ratio: ({ratio * 100:.2f} +/- {ratio_err * 100:.2f})%\n")
        f.write(f"With fragmentation: ({frag_results['with_frag'][0] * 100:.2f} +/- "
                f"{frag_results['with_frag'][1] * 100:.2f})%\n")
        f.write(f"Without fragmentation: ({frag_results['without_frag'][0] * 100:.2f} +/- "
                f"{frag_results['without_frag'][1] * 100:.2f})%\n")
        f.write(
            f"Combined wrong fraction: ({metrics_single.combined_wrong_fraction:.4f} +/- {metrics_single.combined_wrong_fraction_err:.4f})%\n")
        f.write(
            f"Combined tagging power: ({metrics_single.combined_power:.4f} +/- {metrics_single.combined_power_err:.4f})%\n")

        # 3. Two B events (opposite-side tagging)
        two_b_df = classifier.filter_n_b_events(df, 2, event_ids, event_counts)
        signal_two_b = two_b_df[two_b_df["SigMatch"] == 1]

        metrics_two = analyzer.compute_tagging_power_per_eta(signal_two_b)
        analyzer.plot_tagging_power(metrics_two, signal_two_b["eta"],
                                    "doubleB_tagging_power")

        ratio_two, ratio_two_err = calc.calculate_prediction_correctness(signal_two_b)
        try:
            frag_two_results = calc.calculate_by_fragmentation(signal_two_b)
        except:
            frag_two_results = {'with_frag': [-1, -1], "without_frag": [-1, -1]}

        f.write(f"\nDouble B Events: {np.sum(event_counts == 2)}\n")
        f.write(f"Correctness ratio: ({ratio_two * 100:.2f} +/- {ratio_two_err * 100:.2f})%\n")
        f.write(f"With fragmentation: ({frag_two_results['with_frag'][0] * 100:.2f} +/- "
                f"{frag_two_results['with_frag'][1] * 100:.2f})%\n")
        f.write(f"Without fragmentation: ({frag_two_results['without_frag'][0] * 100:.2f} +/- "
                f"{frag_two_results['without_frag'][1] * 100:.2f})%\n")
        f.write(
            f"Combined wrong fraction: ({metrics_two.combined_wrong_fraction:.4f} +/- {metrics_two.combined_wrong_fraction_err:.4f})%\n")
        f.write(
            f"Combined tagging power: ({metrics_two.combined_power:.4f} +/- {metrics_two.combined_power_err:.4f})%\n")

        # 4. B+/- specific analysis
        bpm_df = classifier.filter_bpm_events(two_b_df)

        # Signal B+/-
        bpm_signal = bpm_df[bpm_df["SigMatch"] == 1]
        metrics_bpm_sig = analyzer.compute_tagging_power_per_eta(bpm_signal)
        analyzer.plot_tagging_power(metrics_bpm_sig, bpm_signal["eta"],
                                    "doubleB_ospm_tagging_power")

        # Opposite-side B+/-
        bpm_os = bpm_df[bpm_df["SigMatch"] == 0]
        metrics_bpm_os = analyzer.compute_tagging_power_per_eta(bpm_os)
        analyzer.plot_tagging_power(metrics_bpm_os, bpm_os["eta"],
                                    "osB_pm_tagging_power")

        # Charge-based verification
        charge_frac, charge_correct_df = calc.verify_charge_reconstruction(bpm_os)
        metrics_charge = analyzer.compute_tagging_power_per_eta(charge_correct_df)
        analyzer.plot_tagging_power(metrics_charge, charge_correct_df["eta"],
                                    "osB_pm_true_charge_tagging_power")

        f.write(f"\nB+/- Events: {len(bpm_signal)}\n")
        f.write(f"Signal has OS B+/- wrong fraction: ({metrics_bpm_sig.combined_wrong_fraction:.4f} +/- "
                f"{metrics_bpm_sig.combined_wrong_fraction_err:.4f})%\n")
        f.write(f"Signal has OS B+/- tagging power: ({metrics_bpm_sig.combined_power:.4f} +/- "
                f"{metrics_bpm_sig.combined_power_err:.4f})%\n")
        f.write(f"OS B+/- wrong fraction: ({metrics_bpm_os.combined_wrong_fraction:.4f} +/- "
                f"{metrics_bpm_os.combined_wrong_fraction_err:.4f})%\n")
        f.write(f"OS B+/- tagging power: ({metrics_bpm_os.combined_power:.4f} +/- "
                f"{metrics_bpm_os.combined_power_err:.4f})%\n")
        f.write(f"B+/- charge reconstruction correctness: {charge_frac * 100:.2f}%\n")
        f.write(
            f"Charged correct wrong fraction: ({metrics_charge.combined_wrong_fraction:.4f} +/- {metrics_charge.combined_wrong_fraction_err:.4f})%\n")
        f.write(
            f"Charged correct tagging power: ({metrics_charge.combined_power:.4f} +/- {metrics_charge.combined_power_err:.4f})%\n")
