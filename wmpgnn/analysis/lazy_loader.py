from tqdm import tqdm
import time
import glob

from itertools import chain
from multiprocessing.pool import ThreadPool
from functools import partial

import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch_geometric.loader import DataLoader

from wmpgnn.analysis.weights_calculator import get_hetero_weight
from wmpgnn.util.pruners import *


class LazyGraphDataset(Dataset):
    """Lazy loading dataset that loads files on-demand instead of pre-loading everything"""

    def __init__(self, file_paths, configs, mode, model):
        self.file_paths = file_paths
        self.configs = configs
        self.model = model

        self.nevnts = {}
        self.cumulative_sizes = []
        self.weights = {}
        load_dataset = partial(self._load_dataset, mode=mode)
        total = 0
        sample_order = list(self.file_paths.keys())
        for sample in sample_order:
            weights = []
            with ThreadPool(processes=configs["settings"]["ncpu"] * 2) as pool:
                results = list(
                    tqdm(pool.imap(load_dataset, file_paths[sample]), total=len(file_paths[sample]),
                         desc=f"  Indexing {sample} dataset files"))
            sample_length = 0
            for r in results:
                filtered_length = len(r[0])
                weights.append(r[1])
                total += filtered_length
                sample_length += filtered_length
                self.cumulative_sizes.append(total)

            self.nevnts[sample] = sample_length

            tensor_keys = {'LCA', 'FT'}
            self.weights[sample] = {}
            for d in weights:
                for key, value in d.items():
                    if key not in self.weights[sample]:
                        self.weights[sample][key] = value.clone() if key in tensor_keys else value
                    else:
                        self.weights[sample][key] += value
        print(f"  Total samples after filtering: {total}")
        self.file_paths = list(chain.from_iterable(self.file_paths[sample] for sample in sample_order))

    def _load_dataset(self, path, mode):
        with open(path, "rb") as f:
            data = torch.load(f, weights_only=False)

        """Applying pruning for different using truth pruning initially"""
        if "true" in self.configs["settings"]["graph_mode"]:
            data_selbool = torch.ones(len(data))
            for i, evt in enumerate(data):
                y_nodes = evt["tracks"].ft != 1
                if "frag" in self.configs["graph_mode"]:
                    frag_selbool = evt["tracks"].frag != 0
                    y_nodes = y_nodes | frag_selbool
                if self.configs["node_sel"] == "true":
                    true_node_pruning(y_nodes, evt, "tracks", [('tracks', 'to', 'tracks')])
                elif self.configs["node_sel"] == "default":
                    _ = node_pruning(y_nodes, evt, "tracks", [('tracks', 'to', 'tracks')])
                elif self.configs["node_sel"] == "zeros":
                    _ = test_node_pruning(y_nodes, evt, "tracks", [('tracks', 'to', 'tracks')])
                if evt[("tracks", "to", "tracks")].y.shape[0] == 0 or torch.all(evt[("tracks", "to", "tracks")].y == 0):
                    data_selbool[i] = 0
            filtered_data = [d for d, sel in zip(data, data_selbool) if sel]
        else:
            filtered_data = data

        """Obtain the weights"""
        if self.configs["inference"]["get_weights"] and mode == "train":
            weights = get_hetero_weight(filtered_data, self.configs)
        else:
            weights = {}
        return filtered_data, weights

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        # Find which file contains this index
        file_idx = 0
        for i, cumsum in enumerate(self.cumulative_sizes):
            if idx < cumsum:
                file_idx = i
                break

        # Calculate local index within the file
        local_idx = idx if file_idx == 0 else idx - self.cumulative_sizes[file_idx - 1]

        # Apply filtering and get the correct item
        filtered_data = self._load_dataset(self.file_paths[file_idx], mode="val")[0]
        item = filtered_data[local_idx]

        # Apply model-specific processing
        # add here pruning from pv asso and model for ft
        if self.model == "IFT":
            item["tracks"].x = torch.cat([item["tracks"].x, item["tracks"].pid], dim=1)

        # Remove pid info, might be fine to keep
        if "pid" in item["tracks"]:
            del item["tracks"].pid

        return item


def get_trn_val_loaders(configs, model="DFEI"):
    samples = configs["settings"]["sample"]
    nfiles = {}
    for sample, nfile in zip(samples, configs["settings"]["nfiles"]):
        nfiles[sample] = nfile

    nevts = {"training": {}, "validation": {}}
    print("Start collecting file paths")
    data_dir = configs["settings"]["data_dir"]
    ncpus = configs["settings"]["ncpu"]
    batch_size = configs["settings"]["batch_size"]

    start = time.time()

    # Collect all file paths without loading data
    print("Training:")
    trn_path_set = {}
    for sample in samples:
        trn_path_set[sample] = sorted(glob.glob(f'{data_dir}/{sample}/trn_data_*'))[:nfiles[sample]]
        print(f"  {sample}: {len(trn_path_set[sample])} files")
    print("  Creating lazy datasets...")
    trn_dataset = LazyGraphDataset(trn_path_set, configs, "train", model)

    print("Validation:")
    val_path_set = {}
    for sample in samples:
        val_path_set[sample] = sorted(glob.glob(f'{data_dir}/{sample}/val_data_*'))[:nfiles[sample]]
        print(f"  {sample}: {len(val_path_set[sample])} files")
    print("  Creating lazy datasets...")
    val_dataset = LazyGraphDataset(val_path_set, configs, "val", model)

    end = time.time()
    print(f"Dataset preparation completed, time needed {(end - start):.2f}s")
    print(f"Train dataset       : {len(trn_dataset)}")
    print(f"Validation dataset  : {len(val_dataset)}")
    print("=" * 30)

    # Create DataLoaders with reduced num_workers to save memory
    num_workers = ncpus

    trn_loader = DataLoader(trn_dataset, batch_size=batch_size, num_workers=num_workers,
                            drop_last=True, shuffle=True, persistent_workers=True if num_workers > 0 else False,
                            pin_memory=False)


    # Shuffle validation dataset
    generator = torch.Generator()
    shuffled_indices = torch.randperm(len(val_dataset), generator=generator).tolist()
    val_dataset_shuffled = Subset(val_dataset, shuffled_indices)

    val_loader = DataLoader(val_dataset_shuffled, batch_size=batch_size, num_workers=num_workers,
                            drop_last=True, persistent_workers=True if num_workers > 0 else False, pin_memory=False)

    nevts["training"] = trn_dataset.nevnts
    nevts["validation"] = val_dataset.nevnts
    return trn_loader, val_loader, trn_dataset.weights, nevts


def get_tst_loaders(configs, model="DFEI"):
    sample = configs["evaluate"]["sample"]
    nfiles = configs["evaluate"]["nfiles"]
    configs = configs[model]
    nevts = {"testing": {sample: 0}}

    print("Testing:")
    data_dir = configs["settings"]["data_dir"]
    ncpus = configs["settings"]["ncpu"]

    tst_paths = sorted(glob.glob(f'{data_dir}/{sample}/tst_data_*'))[:nfiles]

    # Create lazy test dataset
    tst_dataset = LazyGraphDataset(tst_paths, configs, "val", model)
    nevts["testing"][sample] = len(tst_dataset)

    # Shuffle the dataset
    generator = torch.Generator()
    shuffled_indices = torch.randperm(len(tst_dataset), generator=generator).tolist()
    tst_dataset_shuffled = Subset(tst_dataset, shuffled_indices)

    num_workers = min(4, ncpus)
    tst_loader = DataLoader(
        tst_dataset_shuffled,
        batch_size=1,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )

    return tst_loader, nevts
