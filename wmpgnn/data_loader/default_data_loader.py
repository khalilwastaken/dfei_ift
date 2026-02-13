from tqdm import tqdm

import glob

from multiprocessing.pool import ThreadPool
from functools import partial

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from wmpgnn.data_loader.helper import *


def get_trn_val_loaders(configs):
    data_dir = configs["settings"]["data_dir"]
    ncpus = configs["settings"]["ncpu"]
    nfiles = get_nfiles(configs["settings"])

    nevts = {"training": {}, "validation": {}}

    print("Training:")
    load_train_dataset = partial(load_dataset, configs=configs, mode="train_weights")
    trn_dataset = []
    weights = {}
    for sample, files in nfiles.items():
        nevts["training"][sample] = 0
        trn_paths = sorted(glob.glob(f'{data_dir}/{sample}/trn_data_*'))[:files]
        with ThreadPool(processes=ncpus) as pool:
            results = list(tqdm(pool.imap(load_train_dataset, trn_paths), total=len(trn_paths),
                                desc=f"Loading {sample} training dataset"))
        for r in results:
            trn_dataset.extend(r[0])
            for key, value in r[1].items():
                if key not in weights:
                    weights[key] = value
                else:
                    weights[key] += value
            nevts["training"][sample] += len(r[0])

    print("Validation:")
    load_val_dataset = partial(load_dataset, configs=configs, mode="val")
    val_dataset = []
    for sample, files in nfiles.items():
        nevts["validation"][sample] = 0
        val_paths = sorted(glob.glob(f'{data_dir}/{sample}/val_data_*'))[:files]
        with ThreadPool(processes=ncpus) as pool:
            results = list(tqdm(pool.imap(load_val_dataset, val_paths), total=len(val_paths),
                                desc=f"Loading {sample} validation dataset"))
        for r in results:
            val_dataset.extend(r)
            nevts["validation"][sample] += len(r)
    print(f"Train dataset       : {len(trn_dataset)}")
    print(f"Validation dataset  : {len(val_dataset)}")
    print("=" * 30)

    batch_size = configs["settings"]["batch_size"]
    trn_loader = DataLoader(trn_dataset, batch_size=batch_size, num_workers=ncpus * 2, drop_last=True, shuffle=True)

    # Shuffle the initial dataset of validation as it is currently sorted by the samples
    generator = torch.Generator()
    shuffled_indices = torch.randperm(len(val_dataset), generator=generator).tolist()
    val_dataset_shuffled = Subset(val_dataset, shuffled_indices)
    val_loader = DataLoader(val_dataset_shuffled, batch_size=batch_size,num_workers=ncpus * 2, drop_last=True)

    return trn_loader, val_loader, weights, nevts


def get_tst_loader(configs, model="DFEI"):
    data_dir = configs["settings"]["data_dir"]
    ncpus = configs["settings"]["ncpu"]
    nfiles = get_nfiles(configs["evaluate"])

    nevts = {"testing": {}}

    print("Testing:")
    load_tst_dataset = partial(load_dataset, configs=configs, mode="val")
    tst_dataset = []
    for sample, files in nfiles.items():
        nevts["testing"][sample] = 0
        tst_paths = sorted(glob.glob(f'{data_dir}/{sample}/tst_data_*'))[:files]
        with ThreadPool(processes=ncpus) as pool:
            results = list(tqdm(pool.imap(load_tst_dataset, tst_paths), total=len(tst_paths),
                                desc=f"Loading {sample} validation dataset"))
        for r in results:
            tst_dataset.extend(r)
            nevts["testing"][sample] += len(r)

    # Shuffle the initial dataset of validation as it is currently sorted by the samples
    generator = torch.Generator()
    shuffled_indices = torch.randperm(len(tst_dataset), generator=generator).tolist()
    tst_dataset_shuffled = Subset(tst_dataset, shuffled_indices)
    tst_loader = DataLoader(tst_dataset_shuffled, batch_size=1, num_workers=ncpus * 2)

    return tst_loader, nevts



