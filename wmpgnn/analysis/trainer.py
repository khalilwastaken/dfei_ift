import os, sys, glob
import time
import yaml
from optparse import OptionParser
from tqdm import tqdm

from multiprocessing.pool import ThreadPool
from functools import partial

from torch_geometric.loader import DataLoader

from trainer_helper import *
from lightning_module import training

from wmpgnn.model.model import DFEI_HGNN, FT_HGNN, DFEIFT

if __name__ == "__main__":
    # python trainer.py  --config  ../../config_files/lightning.yaml
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option("", "--config", type=str, default=None,
                      dest="CONFIG", help="Config file path")
    (option, args) = parser.parse_args()
    if len(args) != 0:
        raise RuntimeError("Got undefined arguments", " ".join(args))

    print("Training script started")
    print("=" * 30)
    # Load config file
    with open(option.CONFIG, "r") as file:
        config = adjust_config(yaml.safe_load(file))

    # Load model
    model = DFEIFT(config["model"])
    print(model)
    print("=" * 30)

    # Get dataset
    samples = config["training"]["sample"]
    nfiles = {}
    for i, sample in enumerate(samples):
        nfiles[sample] = config["training"]["nfiles"][i]
    run_test = any(value for key, value in config["training"]["infer"].items() if key != "LCA")
    nevts = {"training": {}, "validation": {}}
    print("Start reading in the data")
    load_train_dataset = partial(load_dataset, config=config["training"], mode="train")
    load_val_dataset = partial(load_dataset, config=config["training"], mode="val")
    # Training
    start = time.time()
    print("Training:")
    trn_dataset = []
    weights = []
    for sample in samples:
        nevts["training"][sample] = 0
        trn_paths = sorted(glob.glob(f'{config["data_dir"]}/{sample}/trn_data_*'))[:nfiles[sample]]
        with ThreadPool(processes=config["training"]["ncpu"]) as pool:
            results = list(
                tqdm(pool.imap(load_train_dataset, trn_paths), total=len(trn_paths),
                     desc=f"Loading {sample} training dataset"))
        for r in results:
            trn_dataset.extend(r[0])
            weights.append(r[1])
            nevts["training"][sample] += len(r[0])
    # Validation
    print("Validation:")
    val_dataset = []
    for sample in samples:
        nevts["validation"][sample] = 0
        val_paths = sorted(glob.glob(f'{config["data_dir"]}/{sample}/val_data_*'))[:nfiles[sample]]
        with ThreadPool(processes=config["training"]["ncpu"]) as pool:
            results = list(
                tqdm(pool.imap(load_val_dataset, val_paths), total=len(val_paths),
                     desc=f"Loading {sample} validation dataset"))
        for r in results:
            val_dataset.extend(r[0])
            nevts["validation"][sample] += len(r[0])
    # Tests
    if run_test:
        print("Testing:")
        sample = config["evaluate"]["sample"]
        nevts["testing"] = {sample: 0}
        tst_dataset = []
        tst_paths = sorted(glob.glob(f'{config["data_dir"]}/{sample}/tst_data_*'))[:config["evaluate"]["nfiles"]]
        with ThreadPool(processes=config["training"]["ncpu"]) as pool:
            results = list(
                tqdm(pool.imap(load_val_dataset, tst_paths), total=len(tst_paths),
                     desc=f"Loading {sample} test dataset"))
        for r in results:
            tst_dataset.extend(r[0])
            nevts["testing"][sample] += len(r[0])
    end = time.time()

    config.update({"num_events": nevts})
    print(f"data read in, time needed {(end - start):.2f}")
    print(f"Train dataset       : {len(trn_dataset)}")
    print(f"Validation dataset  : {len(val_dataset)}")
    if run_test:
        print(f"Test dataset        : {len(tst_dataset)}")
    print("=" * 30)

    # Transform pos weight
    pos_weights = transform_pos_weight(weights, config["training"]["weights"])

    # Here we can check what kind of gpu it is to specify bs, also num_workers = num_cpu * 2
    trn_loader = DataLoader(trn_dataset, batch_size=config["training"]["batch_size"],
                            num_workers=config["training"]["ncpu"] * 2, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"],
                            num_workers=config["training"]["ncpu"] * 2, drop_last=True)
    if run_test:
        tst_loader = DataLoader(tst_dataset, batch_size=1,
                                num_workers=config["training"]["ncpu"] * 2, drop_last=True)
    else:
        tst_loader = None

    # TODO: Some issue with the tst loader when passing
    metrics, version = training(model, trn_loader, val_loader, tst_loader, config, pos_weights)

    """Evaluate the output metrics"""
    metrics_eval(metrics, config["training"]["infer"], version, config["training"]["sample"][0])
    import pdb;

    pdb.set_trace()
