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
