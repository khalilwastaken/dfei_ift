import numpy as np
import pandas as pd

import re
import os
from dataclasses import dataclass, field
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
    # eta mistag binned
    eta_wrong_fraction: tuple[np.array, np.array]
    eta_power: tuple[np.array, np.array]
    # binned in npvs
    npvs_wrong_fraction: tuple[np.array, np.array]
    npvs_power: tuple[np.array, np.array]
    # Combined quantity
    wrong_fraction: tuple[float, float]
    power: tuple[float, float]
    epsilon: tuple[float, float]
    d_squared: tuple[float, float]


@dataclass
class EtaConfig:
    name: str = r"$\eta$"
    xlabel: str = r"Predicted mistag $\eta$"
    centers: list = field(default_factory=lambda: [0.05, 0.15, 0.25, 0.35, 0.45, 0.55])
    bins: list = field(default_factory=lambda: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1])
    x_err_lower: list = field(init=False)
    x_err_upper: list = field(init=False)
    x_err: list = field(init=False)

    def __post_init__(self):
        self.x_err_lower = [c - self.bins[i] for i, c in enumerate(self.centers)]
        self.x_err_upper = [self.bins[i + 1] - c for i, c in enumerate(self.centers)]
        self.x_err = [self.x_err_lower, self.x_err_upper]


@dataclass
class NPVConfig:
    name: str = "npvs"
    xlabel: str = "#Pvs"
    centers: np.ndarray = field(default_factory=lambda: np.arange(1, 16))
    bins: np.ndarray = field(default_factory=lambda: np.arange(0, 16) + 0.5)
    x_err_lower: list = field(init=False)
    x_err_upper: list = field(init=False)
    x_err: list = field(init=False)

    def __post_init__(self):
        self.x_err_lower = [c - self.bins[i] for i, c in enumerate(self.centers)]
        self.x_err_upper = [self.bins[i + 1] - c for i, c in enumerate(self.centers)]
        self.x_err = [self.x_err_lower, self.x_err_upper]


class TaggingPowerAnalyzer:
    """Analyzes flavor tagging performance across different event configurations."""

    def __init__(self, version, channel, log_dir):
        self.outdir = f"{log_dir}/IFT/version_{version}/plots_{channel}/tagging_power"
        os.makedirs(self.outdir, exist_ok=True)

    @staticmethod
    def calculate_tagging_power(num_right: np.ndarray, num_wrong: np.ndarray,
                                num_unclassified: np.ndarray) -> Tuple[
        Tuple[np.ndarray, np.ndarray],  # wrong_fraction
        Tuple[np.ndarray, np.ndarray],  # power
        Tuple[np.ndarray, np.ndarray],  # efficiency
        Tuple[np.ndarray, np.ndarray],  # d_squared
    ]:
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

        d_squared = (1 - 2 * wrong_fraction) ** 2
        d_squared_err = 4 * (1 - 2 * wrong_fraction) * dwrong_fraction

        wrong_fraction = (wrong_fraction * 100, dwrong_fraction * 100)
        power = (power * 100, dpower * 100)
        efficiency = (efficiency * 100, defficiency * 100)
        d_squared = (d_squared * 100, d_squared_err * 100)
        return wrong_fraction, power, efficiency, d_squared

    @staticmethod
    def process_df(df: pd.DataFrame) -> pd.DataFrame:
        """Process df to contain only signal. Add eta column based on maximum score."""
        sig_selbool = df["SigMatch"] == 1
        reco_selbool = df["AllParticles"] == 1
        evts = np.unique(df[sig_selbool * reco_selbool]["EventNumber"])
        df = df[df["EventNumber"].isin(evts)]

        df = df.copy()
        tag_decision = sig_df[["ft_b_score", "ft_bbar_score"]]
        df["eta"] = 1 - np.max(tag_decision, axis=1) / np.sum(tag_decision, axis=1)
        return df

    def _calculate_per_eta_bin(self, df: pd.DataFrame, mode: str) -> Tuple[
        List, List, List]:  # this we can adapt for npvs and eta
        """Calculate tagging statistics for each eta bin."""
        num_right, num_wrong, num_unclassified = [], [], []
        if mode == "eta":
            configs = EtaConfig()
            var = df["eta"]
        elif mode == "npvs":
            configs = NPVConfig()
            var = df["num_pvs"]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        for i in range(len(configs.centers)):
            in_bin = (var >= configs.bins[i]) & (var < configs.bins[i + 1])
            bin_df = df[in_bin]

            # Count unclassified events
            classified_mask = (bin_df["ft_b_score"] > 0.5) | (bin_df["ft_bbar_score"] > 0.5)
            unclassified = len(bin_df) - np.sum(classified_mask)

            # Get the true tag of the classified events and predicted tag
            true_tag = np.sign(bin_df[classified_mask]["B_id"])
            predicted_tag = np.argmax(bin_df[classified_mask][["ft_b_score", "ft_bbar_score"]], axis=1) * 2 - 1

            num_right.append(np.sum(true_tag == predicted_tag))
            num_wrong.append(np.sum(true_tag != predicted_tag))
            num_unclassified.append(unclassified)

        return np.array(num_right), np.array(num_wrong), np.array(num_unclassified)

    def compute_tagging_power_per_eta(self, df: pd.DataFrame) -> TaggingMetrics:
        """Compute tagging power across eta bins."""
        eta_num_right, eta_num_wrong, eta_num_unclassified = self._calculate_per_eta_bin(df, "eta")
        npvs_num_right, npvs_num_wrong, npvs_num_unclassified = self._calculate_per_eta_bin(df, "npvs")

        # Per-bin metrics
        eta_res = self.calculate_tagging_power(eta_num_right, eta_num_wrong, eta_num_unclassified)
        npvs_res = self.calculate_tagging_power(npvs_num_right, npvs_num_wrong, npvs_num_unclassified)

        # Combined metrics
        comb_res = self.calculate_tagging_power(
            np.sum(eta_num_right), np.sum(eta_num_wrong), np.sum(eta_num_unclassified)
        )
        return TaggingMetrics(eta_res[0], eta_res[1], npvs_res[0], npvs_res[1],
                              comb_res[0], comb_res[1], comb_res[2], comb_res[3])
    @staticmethod
    def compute_tagging_power_per_event(df: pd.DataFrame) -> float:
        classified_mask = (df["ft_b_score"] > 0.5) | (df["ft_bbar_score"] > 0.5)
        sel_df = df[classified_mask]
        return np.sum((1 - 2 * sel_df["eta"]) ** 2) / len(sel_df["eta"]) * 100

    def plot_tagging_power(self, metrics: TaggingMetrics, underlying_dist: np.ndarray, var: str = "eta",
                           label: str = "tagging_power"):

        # here we can switch between eta and npvs
        """Plot tagging power/wrong fraction vs misstag/npvs probability."""
        if var == "eta":
            power = metrics.eta_power
            wrong_fraction = metrics.eta_wrong_fraction
            config = EtaConfig()
        elif var == "npvs":
            power = metrics.npvs_power
            wrong_fraction = metrics.npvs_wrong_fraction
            config = NPVConfig()
        else:
            raise ValueError(f"Unknown var: {var}")

        fig, ax = plt.subplots(figsize=(9, 6))

        # Plot underlying distribution
        weights = np.ones_like(underlying_dist) / len(underlying_dist)
        ax.hist(underlying_dist, bins=config.bins,
                label=rf"Underlying {config.name} distribution",
                color="grey", weights=weights / 2 * 100)

        # Plot metrics
        ax.errorbar(config.centers, power[0], xerr=config.x_err, yerr=power[1],
                    fmt='o', color="#B22222", label="Tagging Power")
        ax.errorbar(config.centers, wrong_fraction[0], xerr=config.x_err, yerr=wrong_fraction[1],
                    fmt='o', color="#4169E1", label="Wrong Fraction")

        if var == "eta":
            ax.set_xlim(0, 0.5)
        else:
            ax.set_xlim(0, 15)
        ax.set_ylim(0, 105)
        ax.set_xlabel(config.xlabel)
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
        if "frags" in df.keys():
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
        else:
            results = {'with_frag': [-1, -1], "without_frag": [-1, -1]}

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
