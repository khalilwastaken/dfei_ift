import gc
import os
from optparse import OptionParser
import yaml
import glob
from tqdm import tqdm

from multiprocessing.pool import ThreadPool
from functools import partial
from itertools import chain

import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import IterableDataset
from torch_geometric.loader import DataLoader

from wmpgnn.analysis.weights_calculator import get_hetero_weight
from wmpgnn.util.pruners import *


class ChunkDataset(IterableDataset):
    # if find a way to load the data during training -> need twice amount of cpu mem
    def __init__(self, file_paths, configs, mode="train", n_chunks=40):
        super().__init__()
        self.file_paths = file_paths
        self.n_chunks = n_chunks
        self.mode = mode
        self.configs = configs
        cumulative_sizes = [0]
        total = 0
        file_index = []

        for i, key in enumerate(self.file_paths.keys()):
            total += len(self.file_paths[key])
            cumulative_sizes.append(total)
            file_index.append(self._generate_groups(n_chunks=self.n_chunks,
                                                    low=cumulative_sizes[i],
                                                    high=cumulative_sizes[i + 1]))
        self.chunk_index = torch.cat(file_index, dim=1)
        self.files_per_chunk = self.chunk_index.shape[1]
        self.n_files = {}
        for sample in self.file_paths.keys():
            self.n_files[sample] = len(self.file_paths[sample])
        self.file_paths = list(chain.from_iterable(self.file_paths[sample] for sample in self.file_paths.keys()))

        self.seeds = torch.randint(0, 1000, (1000,))
        self.seed_tracker = 0

    @staticmethod
    def _generate_groups(n_chunks=20, low=0, high=80):
        groups = []
        available = np.arange(low, high)
        interval_size = high - low
        chunksize = int(np.ceil(interval_size / n_chunks))

        for i in range(n_chunks):
            if len(available) >= chunksize:
                indices = np.random.choice(len(available), size=chunksize, replace=False)
                group = available[indices]
                available = np.delete(available, indices)
            else:
                if len(available) > 0:
                    group = np.concatenate([
                        available,
                        np.random.choice(np.arange(low, high), size=chunksize - len(available), replace=True)
                    ])
                else:
                    group = np.random.choice(np.arange(low, high), size=chunksize, replace=True)
                available = np.array([])
            groups.append(group.tolist())
        return torch.tensor(groups)

    def _load_dataset(self, path, mode="train"):
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

        if mode == "train":
            return filtered_data
        elif mode == "weights":
            weights = get_hetero_weight(filtered_data, self.configs)
            return weights
        else:
            raise NotImplementedError

    def _load_chunk(self, chunk_number, mode="loading"):
        nevnts = 800
        if self.mode == "validation" or self.mode == "test":
            nevnts = 200

        chunk = self.chunk_index[chunk_number]
        files = [self.file_paths[i] for i in chunk]
        desc = f"Loading {self.mode} chunk {chunk_number + 1}/{self.n_chunks} ({self.files_per_chunk} files, ~{self.files_per_chunk * nevnts} events)"
        if mode == "loading":
            dataset = []
            load_dataset = partial(self._load_dataset)
            with ThreadPool(processes=self.configs["settings"]["ncpu"]) as pool:
                for r in tqdm(pool.imap(load_dataset, files), total=len(files), desc=desc):
                    dataset.extend(r)
            return dataset
        elif mode == "weights":
            weights = {}
            load_dataset = partial(self._load_dataset, mode=mode)
            with ThreadPool(processes=self.configs["settings"]["ncpu"]) as pool:
                for r in tqdm(pool.imap(load_dataset, files), total=len(files),desc=desc):
                    for key, value in r.items():
                        if key not in weights:
                            weights[key] = value
                        else:
                            weights[key] += value
            return weights
        else:
            raise NotImplementedError

    def __iter__(self):
        # assuming 4 num workers which each want to load a chunk
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = self.n_chunks
        else:
            worker_id = worker_info.id
            per_worker = int(np.ceil(self.n_chunks / worker_info.num_workers))
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.n_chunks)

        for chunk_number in range(iter_start, iter_end):
            # for chunk_number in range(self.n_chunks):
            chunk_events = self._load_chunk(chunk_number)

            # create index
            idx = torch.arange(0, len(chunk_events))
            # we always shuffle val not sure if this is correct or only in the initial one
            if self.mode == "train":
                g = torch.Generator()
                g.manual_seed(self.seeds[self.seed_tracker].item())
                self.seed_tracker += 1
                idx = idx[torch.randperm(len(idx), generator=g)]
            else:
                idx = idx[torch.randperm(len(idx))]

            # Yield events in shuffled order
            for i in idx:
                yield chunk_events[i]

            del chunk_events
            gc.collect()

    def get_weights(self):
        # pass n entries
        weights = {}
        for i in range(10):
            weights[i] = self._load_chunk(i, mode="weights")
        return weights


class ChunkLoader(pl.LightningDataModule):
    # Data container for pytorch lightning module
    def __init__(self, configs, trn_dataset=None, val_dataset=None, tst_dataset=None, batchsize=None, num_workers=None):
        super().__init__()
        self.trn_dataset = trn_dataset
        self.val_dataset = val_dataset
        self.tst_dataset = tst_dataset
        self.configs = configs

        if isinstance(batchsize, int):
            self.batchsize = batchsize
        else:
            self.batch_size = self.configs["settings"]["batch_size"]
        if num_workers is not None:
            self.num_workers = num_workers
        else:
            self.num_workers = 8  # lets test with 4
        # self.num_workers = self.configs["settings"]["ncpu"] * 2
        # persistent workers take too much mem

    def train_dataloader(self):
        if not isinstance(self.trn_dataset, type(None)):
            trn_loader = DataLoader(self.trn_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                    drop_last=True, persistent_workers=False,
                                    pin_memory=False)
        else:
            raise ValueError("trn_dataset must not be None")
        return trn_loader

    def val_dataloader(self):
        if not isinstance(self.val_dataset, type(None)):
            val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                    drop_last=True, persistent_workers=False,
                                    pin_memory=False)
        else:
            raise ValueError("val_dataset must not be None")
        return val_loader

    def test_dataloader(self):
        if not isinstance(self.tst_dataset, type(None)):
            val_loader = DataLoader(self.tst_dataset, batch_size=1, num_workers=self.num_workers,
                                    drop_last=True, persistent_workers=False,
                                    pin_memory=False)
        else:
            raise ValueError("tst_dataset not must be None")
        return val_loader


def get_trn_val_loaders(configs):
    print("Start collecting file paths")
    samples = configs["settings"]["sample"]
    nfiles = {}
    for sample, nfile in zip(samples, configs["settings"]["nfiles"]):
        nfiles[sample] = nfile
    data_dir = configs["settings"]["data_dir"]

    """Training"""
    path_dict = {}
    for sample in samples:
        path_dict[sample] = sorted(glob.glob(f'{data_dir}/{sample}/trn_data_*'))[:nfiles[sample]]
    trn_dataset = ChunkDataset(path_dict, configs, mode="train")

    """Validation"""
    path_dict = {}
    for sample in samples:
        path_dict[sample] = sorted(glob.glob(f'{data_dir}/{sample}/val_data_*'))[:nfiles[sample]]
    val_dataset = ChunkDataset(path_dict, configs, mode="validation")

    return ChunkLoader(configs, trn_dataset=trn_dataset, val_dataset=val_dataset)


def get_tst_loader(configs, model):
    sample = configs["evaluate"]["sample"]
    nfiles = configs["evaluate"]["nfiles"]
    data_dir = configs[model]["settings"]["data_dir"]
    configs = configs[model]

    path_dict = {}
    path_dict[sample] = sorted(glob.glob(f'{data_dir}/{sample}/tst_data_*'))[:nfiles]
    tst_dataset = ChunkDataset(path_dict, configs, mode="test", n_chunks=nfiles)

    return ChunkLoader(configs, tst_dataset=tst_dataset, batchsize=1, num_workers=1)


if __name__ == "__main__":
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option("", "--config", type=str, default=None,
                      dest="CONFIG", help="Config file path")
    (option, args) = parser.parse_args()
    if len(args) != 0:
        raise RuntimeError("Got undefined arguments", " ".join(args))

    with open(option.CONFIG, "r") as file:
        configs = yaml.safe_load(file)

    samples = configs["settings"]["sample"]
    data_dir = configs["settings"]["data_dir"]
    path_dict = {}
    for sample in samples:
        path_dict[sample] = sorted(glob.glob(f'{data_dir}/{sample}/trn_data_*'))

    dataset = ChunkDataset(path_dict)
    import pdb;

    pdb.set_trace()

# 80 + 80 + 200 = 360 -> factor 10

# 36 -> 18 -> 20 chunks
