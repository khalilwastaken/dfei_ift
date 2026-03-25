import numpy as np
import pandas as pd

sig_dict = {
    "Bs_Jpsiphi": {"Jpsi": "ok"}
}

mass_dict = {
    211: 139.57039,
    13: 105.6583755,
    321: 493.677,
    2212: 938.27208816,
    11: 0.51099895000
}


def compute_b_kinematics(df_row):
    pid_signed = np.array([int(x) for x in df_row["final_pid"].split(",")])
    pid = np.abs(pid_signed)

    # Handle unknown PIDs → return -1 sentinel row
    try:
        mass = np.array([mass_dict[part] for part in pid])
    except KeyError:
        return pd.Series({"M_B": -1})

    px = np.array([float(x) for x in df_row["final_px"].split(",")]) * 1000
    py = np.array([float(x) for x in df_row["final_py"].split(",")]) * 1000
    pz = np.array([float(x) for x in df_row["final_pz"].split(",")]) * 1000

    E = np.sqrt(mass ** 2 + px ** 2 + py ** 2 + pz ** 2)

    # --- B meson: sum all particles ---
    E_B = E.sum()
    px_B = px.sum()
    py_B = py.sum()
    pz_B = pz.sum()
    M_B = np.sqrt(E_B ** 2 - px_B ** 2 - py_B ** 2 - pz_B ** 2)

    return pd.Series({"M_B": M_B})