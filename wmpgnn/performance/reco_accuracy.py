import numpy as np
import os


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


def write_to_file(file, cond, performance, label=None):
    all_part, perfect_reco, none_iso, part_reco = performance
    with open(file, cond) as f:
        f.write("=" * 30 + "\n")
        f.write(f"Reconstruction efficiency{label}: \n")
        f.write(f"all_particles : {all_part[0]:.2f} +/- {all_part[1]:.2f} \n")
        f.write(f"perfect_reco  : {perfect_reco[0]:.2f} +/- {perfect_reco[1]:.2f} \n")
        f.write(f"none_iso      : {none_iso[0]:.2f} +/- {none_iso[1]:.2f} \n")
        f.write(f"part_reco     : {part_reco[0]:.2f} +/- {part_reco[1]:.2f} \n")


def obtain_reco_accuracy(df, version, signal, log_dir):
    # Full signal
    # Full OS inclusive
    # Signal single B events
    # Signal two B events
    # Signal more than two B events
    file = f"{log_dir}/DFEI/version_{version}/info_{signal}_reco.txt"
    cond = "a" if os.path.exists(file) else "w"

    evtnumber, counts = np.unique(df["EventNumber"], return_counts=True)

    if "inclusive" not in signal:
        sig_df = df[df["SigMatch"] == 1]
        bkg_df = df[df["SigMatch"] != 1]
    else:
        sig_df = df
        bkg_df = None

    performance = calculate_accuracy(sig_df)
    write_to_file(file, cond, performance, label="")

    if bkg_df is not None:
        performance = calculate_accuracy(bkg_df)
        write_to_file(file, "a", performance, label=" inclusive")

    single_evts = evtnumber[counts == 1]
    performance = calculate_accuracy(sig_df[sig_df["EventNumber"].isin(single_evts)])
    write_to_file(file, "a", performance, label=" only 1 B in event")

    two_evts = evtnumber[counts == 2]
    performance = calculate_accuracy(sig_df[sig_df["EventNumber"].isin(two_evts)])
    write_to_file(file, "a", performance, label=" only 2 B in event")

    more_evts = evtnumber[counts > 2]
    performance = calculate_accuracy(sig_df[sig_df["EventNumber"].isin(more_evts)])
    write_to_file(file, "a", performance, label=" more than 2 B in event")
