import os, sys, glob
import time
import yaml
from optparse import OptionParser

from tqdm import tqdm

from multiprocessing.pool import ThreadPool
from functools import partial

from torch_geometric.loader import DataLoader

from trainer_helper import *
from model import DFEI_HGNN
from lightning_module import evaluate

if __name__ == "__main__":
    # python trainer.py  --config  ../../config_files/lightning.yaml
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option("", "--config", type=str, default=None,
                      dest="CONFIG", help="Config file path")
    (option, args) = parser.parse_args()
    if len(args) != 0:
        raise RuntimeError("Got undefined arguments", " ".join(args))

    # Load config file
    with open(option.CONFIG, "r") as file:
        config = adjust_config(yaml.safe_load(file))

    # Load model
    DFEI_model = DFEI_HGNN(config["model"])
    print(DFEI_model)
    print("=" * 30)

    # Get dataset
    sample = config["evaluate"]["sample"]
    print("Start reading in the data")
    load_val_dataset = partial(load_dataset, config=config["training"], mode="val")
    start = time.time()
    print("Testing:")
    tst_dataset = []
    print(f"Loading {sample}")
    tst_paths = sorted(glob.glob(f'{config["data_dir"]}/{sample}/tst_data_*'))[:config["evaluate"]["nfiles"]]
    with ThreadPool(processes=config["training"]["ncpu"]) as pool:
        results = list(
            tqdm(pool.imap(load_val_dataset, tst_paths), total=len(tst_paths), desc="Test dataset"))
    for r in results:
        tst_dataset.extend(r[0])
    end = time.time()
    print(f"data read in, time needed {(end - start):.2f}")
    print(f"Test dataset        : {len(tst_dataset)}")
    print("=" * 30)

    tst_loader = DataLoader(tst_dataset, batch_size=1,
                            num_workers=config["training"]["ncpu"] * 2, drop_last=True)
    pos_weights = transform_pos_weight(None, None, mode="eval")

    metrics, version = evaluate(DFEI_model, tst_loader, config, pos_weights)

    """Evaluate the output metrics"""
    metrics_eval(metrics, config["training"]["infer"], version, config["evaluate"]["sample"])
