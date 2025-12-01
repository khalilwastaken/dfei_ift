from tqdm import tqdm

import time
import glob

from multiprocessing.pool import ThreadPool
from functools import partial

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from wmpgnn.analysis.weights_calculator import get_hetero_weight
from wmpgnn.util.pruners import *


def get_trn_val_loaders(configs, model="DFEI"):
    samples = configs["settings"]["sample"]
    nfiles = {}
    for sample, nfile in zip(samples, configs["settings"]["nfiles"]):
        nfiles[sample] = nfile

    nevts = {"training": {}, "validation": {}}
    print("Start reading in the data")
    data_dir = configs["settings"]["data_dir"]
    ncpus = configs["settings"]["ncpu"]

    start = time.time()
    print("Training:")
    load_train_dataset = partial(load_dataset, _configs=configs, mode="train", model=model)
    trn_dataset = []
    weights = {}
    for sample in samples:
        nevts["training"][sample] = 0
        trn_paths = sorted(glob.glob(f'{data_dir}/{sample}/trn_data_*'))[:nfiles[sample]]
        with ThreadPool(processes=ncpus) as pool:
            results = list(tqdm(pool.imap(load_train_dataset, trn_paths), total=len(trn_paths),
                                desc=f"Loading {sample} training dataset"))
        for r in results:
            trn_dataset.extend(r[0])
            for key, value in r.items():
                if key not in self.weights[sample]:
                    weights[sample][key] = value
                else:
                    weights[sample][key] += value
            nevts["training"][sample] += len(r[0])

    print("Validation:")
    load_val_dataset = partial(load_dataset, _configs=configs, mode="val", model=model)
    val_dataset = []
    for sample in samples:
        nevts["validation"][sample] = 0
        val_paths = sorted(glob.glob(f'{data_dir}/{sample}/val_data_*'))[:nfiles[sample]]
        with ThreadPool(processes=ncpus) as pool:
            results = list(tqdm(pool.imap(load_val_dataset, val_paths), total=len(val_paths),
                                desc=f"Loading {sample} validation dataset"))
        for r in results:
            val_dataset.extend(r[0])
            nevts["validation"][sample] += len(r[0])
    end = time.time()
    print(f"data read in, time needed {(end - start):.2f}")
    print(f"Train dataset       : {len(trn_dataset)}")
    print(f"Validation dataset  : {len(val_dataset)}")
    print("=" * 30)

    batch_size = configs["settings"]["batch_size"]
    trn_loader = DataLoader(trn_dataset, batch_size=batch_size,
                            num_workers=ncpus * 2, drop_last=True, shuffle=True)

    # Shuffle the initial dataset of validation as it is currently sorted by the samples
    generator = torch.Generator()
    shuffled_indices = torch.randperm(len(val_dataset), generator=generator).tolist()
    val_dataset_shuffled = Subset(val_dataset, shuffled_indices)
    val_loader = DataLoader(val_dataset_shuffled, batch_size=batch_size,
                            num_workers=ncpus * 2, drop_last=True)

    return trn_loader, val_loader, weights, nevts


def get_tst_loader(configs, model="DFEI"):
    sample = configs["evaluate"]["sample"]
    nfiles = configs["evaluate"]["nfiles"]
    configs = configs[model]
    nevts = {"testing": {sample: 0}}

    print("Testing:")
    load_tst_dataset = partial(load_dataset, _configs=configs, mode="val", model=model)
    tst_dataset = []
    tst_paths = sorted(glob.glob(f'{configs["settings"]["data_dir"]}/{sample}/tst_data_*'))[:nfiles]
    with ThreadPool(processes=configs["settings"]["ncpu"]) as pool:
        results = list(
            tqdm(pool.imap(load_tst_dataset, tst_paths), total=len(tst_paths),
                 desc=f"Loading {sample} test dataset"))
    for r in results:
        tst_dataset.extend(r[0])
        nevts["testing"][sample] += len(r[0])

    # Shuffle the initial dataset of validation as it is currently sorted by the samples
    generator = torch.Generator()
    shuffled_indices = torch.randperm(len(tst_dataset), generator=generator).tolist()
    tst_dataset_shuffled = Subset(tst_dataset, shuffled_indices)
    tst_loader = DataLoader(tst_dataset_shuffled, batch_size=1,
                            num_workers=configs["settings"]["ncpu"] * 2, drop_last=True)
    return tst_loader, nevts


def load_dataset(path, _configs, mode, model):
    with open(path, "rb") as f:
        data = torch.load(f, weights_only=False)

    """Applying pruning for different using truth pruning intially""" # add here pruning from pv asso
    if "true" in _configs["settings"]["graph_mode"]:
        data_selbool = torch.ones(len(data))
        for i, evt in enumerate(data):
            y_nodes = evt["tracks"].ft != 1
            if "frag" in _configs["graph_mode"]:
                frag_selbool = evt["tracks"].frag != 0
                y_nodes = y_nodes | frag_selbool
            if _configs["node_sel"] == "true":
                true_node_pruning(y_nodes, evt, "tracks", [('tracks', 'to', 'tracks')])
            elif _configs["node_sel"] == "default":
                _ = node_pruning(y_nodes, evt, "tracks", [('tracks', 'to', 'tracks')])
            elif _configs["node_sel"] == "zeros":
                _ = test_node_pruning(y_nodes, evt, "tracks", [('tracks', 'to', 'tracks')])
            if evt[("tracks", "to", "tracks")].y.shape[0] == 0 or torch.all(evt[("tracks", "to", "tracks")].y == 0):
                data_selbool[i] = 0
        filtered_data = [d for d, sel in zip(data, data_selbool) if sel]
    else:
        filtered_data = data

    """Adding pid information as a node feature for IFT"""
    if model == "IFT":
        for i in range(len(filtered_data)):
            filtered_data[i]["tracks"].x = torch.cat([filtered_data[i]["tracks"].x, filtered_data[i]["tracks"].pid], dim=1)
    # pid info is only used within the nodes, remove it as a feature
    for item in filtered_data:
        del item["tracks"].pid

    """Obtain the weights"""
    if _configs["inference"]["get_weights"] and mode == "train":
        weights = get_hetero_weight(filtered_data, _configs)
    else:
        weights = {}
    return filtered_data, weights
