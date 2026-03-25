from argparse import ArgumentParser


import pandas as pd
import numpy as np

from channel_dict import *




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--version", type=str, required=True,
                        dest="VERSION", help="LHCb_logs/IFT/version8")
    parser.add_argument("--sample", type=str, required=True,
                        dest="SAMPLE", help="00342638_Bs_Jpsiphi_00342629_Bs_Jpsiphi")
    args = parser.parse_args()
    print("=" * 45)
    print("Plotting the mass distribution")

    # Get the signal channel
    if args.SAMPLE.split("_")[1] == "inclusive":
        signal = "inclusive"
    else:
        signal = args.SAMPLE.split("_")[1] + "_" + args.SAMPLE.split("_")[2]

    # Get the reco df
    reco_df = pd.read_csv(f"../analysis/{args.VERSION}/signal_reco_data_df_{args.SAMPLE}.csv")
    sig_df = reco_df[reco_df["SigLike"] == 1].copy()
    reco_df["M_B"] = reco_df.apply(compute_b_kinematics, axis=1)



    import pdb; pdb.set_trace()