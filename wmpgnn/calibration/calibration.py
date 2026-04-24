import pandas as pd
import numpy as np
import uproot
import awkward as ak


def create_calib_root(df: pd.DataFrame, version: str, signal: str, log_dir: str = "lightning_logs"):
    outfile = f"{log_dir}/IFT/version_{version}/{signal}_calib.root"

    sig_df = df[(df["SigMatch"] == 1) & (df["PerfectReco"] == 1)].copy()

    # tag decision
    tag_decision = sig_df[["ft_b_score", "ft_bbar_score"]]

    # add quantities which are needed in calibration
    sig_df["B_IFT_TAGETA"] = 1 - np.max(tag_decision, axis=1) / np.sum(tag_decision, axis=1)
    sig_df["B_IFT_TAGDEC"] = np.argmax(tag_decision, axis=1) * 2 - 1
    sig_df["B_PARTICLE_ID"] = sig_df["B_id"]
    save_df = sig_df[["EVENTNUMBER", "RUNNUMBER", "num_pvs", "B_IFT_TAGETA", "B_IFT_TAGDEC", "B_PARTICLE_ID"]]

    with uproot.recreate(outfile) as f:
        f["DecayTree"] = {col: sig_df[col].values for col in save_df.columns}
