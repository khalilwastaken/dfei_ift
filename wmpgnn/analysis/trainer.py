import os, sys, glob
import time
import yaml
from optparse import OptionParser
from tqdm import tqdm

from multiprocessing.pool import ThreadPool
from functools import partial

from torch_geometric.loader import DataLoader

from trainer_helper import *
from model import HeteroGNN
from lightning_module import training

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
        config = yaml.safe_load(file)

    # Load model
    model = HeteroGNN(config["model"])
    checkpoint_path = config["training"]["cpt"]  # load the previous last model to retrain
    print(model)
    print("=" * 30)

    # Get dataset
    samples = config["training"]["sample"]
    print("Start reading in the data")
    load_train_dataset = partial(load_dataset, config=config["training"], mode="train")
    load_val_dataset = partial(load_dataset, config=config["training"], mode="val")
    # Training
    print("Training:")
    start = time.time()
    trn_dataset = []
    weights = []
    for sample in samples:
        print(f"Loading {sample}")
        trn_paths = sorted(glob.glob(f'{config["data_dir"]}/{sample}/trn_data_*'))[:config["training"]["nfiles"]]
        with ThreadPool(processes=config["training"]["ncpu"]) as pool:
            results = list(
                tqdm(pool.imap(load_train_dataset, trn_paths), total=len(trn_paths), desc="Training dataset"))
        for r in results:
            trn_dataset.extend(r[0])
            weights.append(r[1])
    # Validation
    print("Validation:")
    val_dataset = []
    for sample in samples:
        print(f"Loading {sample}")
        val_paths = sorted(glob.glob(f'{config["data_dir"]}/{sample}/val_data_*'))[:config["training"]["nfiles"]]
        with ThreadPool(processes=config["training"]["ncpu"]) as pool:
            results = list(
                tqdm(pool.imap(load_val_dataset, val_paths), total=len(val_paths), desc="Validation dataset"))
        for r in results:
            val_dataset.extend(r[0])
    end = time.time()

    print(f"data read in, time needed {(end - start):.2f}")
    print(f"Train dataset       : {len(trn_dataset)}")
    print(f"Validation dataset  : {len(val_dataset)}")
    print("=" * 30)

    # Transform pos weight
    pos_weights = transform_pos_weight(weights, config["training"]["weights"])

    # Here we can check what kind of gpu it is to specify bs, also num_workers = num_cpu * 2
    trn_loader = DataLoader(trn_dataset, batch_size=config["training"]["batch_size"],
                            num_workers=config["training"]["ncpu"] * 2, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"],
                            num_workers=config["training"]["ncpu"] * 2, drop_last=True)

    training(model, trn_loader, val_loader, config, pos_weights)
