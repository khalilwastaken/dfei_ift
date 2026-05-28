import gc
import glob
from tqdm import tqdm

from multiprocessing.pool import ThreadPool
from functools import partial
from itertools import chain

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import IterableDataset
from torch_geometric.loader import DataLoader

from wmpgnn.data_loader.data_loader_class import DataSetLoader
from wmpgnn.data_loader.helper import get_nfiles, load_file


class ChunkDataset(IterableDataset):
    # Loading a chunk of the dataset to cpu memory instead of all files
    def __init__(self, file_paths, configs, data_set_loader, mode="train", n_chunks=32):
        super().__init__()
        self.file_paths = file_paths
        self.n_chunks = n_chunks
        self.mode = mode
        self.configs = configs
        self.data_set_loader = data_set_loader
        cumulative_sizes = [0]
        total = 0
        file_index = []

        for i, key in enumerate(self.file_paths.keys()):
            total += len(self.file_paths[key])
            cumulative_sizes.append(total)
            file_index.append(self._generate_groups(n_chunks=self.n_chunks,
                                                    low=cumulative_sizes[i],
                                                    high=cumulative_sizes[i + 1]))
        self.chunk_index = torch.cat(file_index, dim=1).to(torch.long)
        self.files_per_chunk = self.chunk_index.shape[1]
        self.n_files = {}
        for sample in self.file_paths.keys():
            self.n_files[sample] = len(self.file_paths[sample])
        self.file_paths = list(chain.from_iterable(self.file_paths[sample] for sample in self.file_paths.keys()))

        self.seeds = torch.randint(0, 1000, (1000,))
        self.seed_tracker = 0

    @staticmethod
    def _generate_groups(n_chunks: int, low: int, high: int) -> torch.Tensor:
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

    def _load_chunk(self, chunk_number, mode="loading"):
        nevnts = 800
        if self.mode == "validation" or self.mode == "test":
            nevnts = 200

        chunk = self.chunk_index[chunk_number]
        files = [self.file_paths[i] for i in chunk]
        desc = f"Loading {self.mode} chunk {chunk_number + 1}/{self.n_chunks} ({self.files_per_chunk} files, ~{self.files_per_chunk * nevnts} events)"
        if mode == "loading":
            dataset = []
            load_dataset_part = partial(self.data_set_loader.load_data, mode=mode)
            with ThreadPool(processes=self.configs["settings"]["ncpu"]) as pool:
                for r in tqdm(pool.imap(load_dataset_part, files), total=len(files), desc=desc, leave=False):
                    dataset.extend(r)
            return dataset
        elif mode == "weights":
            weights = {}
            load_dataset_part = partial(self.data_set_loader.load_data, mode="weights_only")
            with ThreadPool(processes=1) as pool: #self.configs["settings"]["ncpu"]) as pool:
                for r in tqdm(pool.imap(load_dataset_part, files), total=len(files), desc=desc, leave=False):
                    for key, value in r.items():
                        weights[key] = weights.get(key, 0) + value
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
            if self.mode == "train":
                idx = idx[torch.randperm(len(idx))]
            else:
                # val and test are shuffled
                g = torch.Generator()
                g.manual_seed(self.seeds[self.seed_tracker].item())
                self.seed_tracker += 1
                idx = idx[torch.randperm(len(idx), generator=g)]

            # Yield events in shuffled order
            for i, i_idx in enumerate(idx):
                yield chunk_events[i_idx]

            del chunk_events
            gc.collect()

    def get_weights(self):
        weights = {}
        try:
            for i in range(10):
                weights[i] = self._load_chunk(i, mode="weights")
        except IndexError:
            print("Using reduced number of chunks for weights:", i)
        print("Done")
        print("=" * 15)
        return weights


class ChunkLoader(pl.LightningDataModule):
    # Data container for pytorch lightning module
    def __init__(self, configs, trn_dataset=None, val_dataset=None, tst_dataset=None, batch_size=None,
                 num_workers=None):
        super().__init__()
        self.trn_dataset = trn_dataset
        self.val_dataset = val_dataset
        self.tst_dataset = tst_dataset
        self.configs = configs

        if isinstance(batch_size, int):
            self.batch_size = batch_size
        else:
            self.batch_size = self.configs["settings"]["batch_size"]
        if isinstance(num_workers, int):  # we need to be cautious to not load too much to cpu mem
            self.num_workers = num_workers
        else:
            self.num_workers = self.configs["settings"]["ncpu"] * 2

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
            tst_loader = DataLoader(self.tst_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                    drop_last=False, persistent_workers=False,
                                    pin_memory=False)
        else:
            raise ValueError("tst_dataset not must be None")
        return tst_loader


def get_trn_val_loaders(_configs) -> ChunkLoader:
    data_dir = _configs["settings"]["data_dir"]
    num_workers = _configs["settings"]["ncpu"] * 2
    nfiles = get_nfiles(_configs["settings"])

    """Paths of the train and validation files"""
    trn_path_dict = {}
    val_path_dict = {}
    for sample, files in nfiles.items():
        trn_path_dict[sample] = sorted(glob.glob(f'{data_dir}/{sample}/trn_data_*'))[:files]
        val_path_dict[sample] = sorted(glob.glob(f'{data_dir}/{sample}/val_data_*'))[:files]

    # Initiate a data set loader
    data_set_loader = DataSetLoader(_configs)

    # Number of chunks definition and safeguard for the files per chunk to be less than 8
    min_files = min(len(v) for v in trn_path_dict.values() if len(v) > 0)
    total_files = sum(len(v) for v in trn_path_dict.values())
    num_chunks = np.ceil(min_files / num_workers).astype(int)
    # Safeguard: increase chunks until files_per_chunk < 8, this can be adapted
    while total_files / num_chunks >= 8:
        num_chunks += num_workers
    print(f"Number of chunks: {num_chunks}")
    print(f"Files per chunk: {np.ceil(total_files / num_chunks).astype(int)}")
    trn_dataset = ChunkDataset(trn_path_dict, _configs, data_set_loader, mode="train", n_chunks=num_chunks)
    val_dataset = ChunkDataset(val_path_dict, _configs, data_set_loader, mode="validation", n_chunks=num_chunks)

    return ChunkLoader(_configs, trn_dataset=trn_dataset, val_dataset=val_dataset)


def get_tst_loader(_configs) -> ChunkLoader:
    data_dir = _configs["settings"]["data_dir"]
    nfiles = get_nfiles(_configs["evaluate"])

    """Testing"""
    path_dict = {"testing": []}
    for sample, files in nfiles.items():
        path_dict["testing"].append(sorted(glob.glob(f'{data_dir}/{sample}/tst_data_*'))[:files])
    path_dict["testing"] = sum(path_dict["testing"], [])

    # Initiate a data set loader
    data_set_loader = DataSetLoader(_configs)
    # Each file is saved individually in a chunk
    batch_size = 512  # increased bs possible during testing
    tst_dataset = ChunkDataset(path_dict, _configs, data_set_loader, mode="test", n_chunks=len(path_dict["testing"]))

    return ChunkLoader(_configs, tst_dataset=tst_dataset, batch_size=batch_size)
