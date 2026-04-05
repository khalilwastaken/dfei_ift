from argparse import ArgumentParser
import os

import pandas as pd
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--version", type=str, required=True,
                        dest="VERSION", help="LHCb_logs/IFT/version8")
    parser.add_argument("--sample", type=str, required=True,
                        dest="SAMPLE", help="00342638_Bs_Jpsiphi_00342629_Bs_Jpsiphi")
    args = parser.parse_args()
    print("=" * 45)
    print("Starting the evaluating the reco efficiency")

    # Output file
    file = f"{args.VERSION}/info_{args.SAMPLE}_reco.txt"
    cond = "a" if os.path.exists(file) else "w"

    reco_df = pd.read_csv(f"{args.VERSION}/signal_reco_data_df_{args.SAMPLE}.csv")
    true_df = pd.read_csv(f"{args.VERSION}/signal_reco_df_{args.SAMPLE}.csv")

    # Filter to event which are classified to be signal like
    true_sig_df = true_df[true_df["SigMatch"] == 1].copy()
    reco_sig_df = reco_df[reco_df["SigLike"] == 1].copy()

    true_sig_df["EVENTNUMBER"] = true_sig_df["EVENTNUMBER"].astype(int)
    true_sig_df["RUNNUMBER"] = true_sig_df["RUNNUMBER"].astype(int)
    reco_sig_df["EVENTNUMBER"] = reco_sig_df["EVENTNUMBER"].astype(int)
    reco_sig_df["RUNNUMBER"] = reco_sig_df["RUNNUMBER"].astype(int)

    # Doing a event number matching by cantor pairs instead of tuple lvl
    true_sig_df["key"] = true_sig_df["EVENTNUMBER"].astype(str) + "_" + true_sig_df["RUNNUMBER"].astype(str)
    reco_sig_df["key"] = reco_sig_df["EVENTNUMBER"].astype(str) + "_" + reco_sig_df["RUNNUMBER"].astype(str)

    truth_found = true_sig_df["key"].isin(reco_sig_df["key"])
    reco_wrong = reco_sig_df["key"].isin(true_sig_df["key"])

    with open(file, cond) as f:
        f.write("=" * 50 + "\n")
        f.write("Reconstruction efficiency truth matched \n")
        # One needs to care about dupes:
        # An event which contain 2 signal decay modes are not treated
        f.write(f"Number of true signal channels                : {len(truth_found)}" + "\n")
        f.write(f"Number of true signal channels found by reco  : {np.sum(truth_found)}" + "\n")
        f.write(f"Signal finding efficiency                     : {(np.sum(truth_found) / len(truth_found) * 100):.2f}%" + "\n")
        f.write("=" * 30 + "\n")
        f.write(f"Number of reco signal channels                : {len(reco_wrong)}" + "\n")
        f.write(f"Number of reco signal channels truth matched  : {np.sum(reco_wrong)}" + "\n")
        f.write(f"Background contamination rate                 : {((len(reco_wrong)-np.sum(truth_found)) / len(reco_wrong) * 100):.2f}%" + "\n")

    print("Done")
    print("=" * 45)