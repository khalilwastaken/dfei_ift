import copy
import io
import zstandard as zstd
from itertools import chain

import torch
from torch_geometric.loader import DataLoader

from wmpgnn.util.pruners import *
from wmpgnn.util.pv_association import pv_associate_data
from wmpgnn.calibration.calibration_mask import *
from wmpgnn.data_loader.weights_calculator import get_hetero_weight


def get_nfiles(_configs):
    samples = _configs["sample"]
    nfiles = {}
    for sample, nfile in zip(samples, _configs["nfiles"]):
        nfiles[sample] = nfile
    return nfiles


def load_dataset(path, configs, mode="train", pv_asso_model=None):
    dctx = zstd.ZstdDecompressor()
    with open(path, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            # Read all decompressed data into memory
            decompressed = reader.read()
            # Load from BytesIO buffer
            data = torch.load(io.BytesIO(decompressed), weights_only=False)

    """Applying pruning for different using truth pruning initially"""
    if "true" in configs["settings"]["graph_mode"]:  # here we need to apply the pv association as well
        data_selbool = torch.ones(len(data))
        edge_types = [("tracks", "to", "tracks"), ("tracks", "to", "pvs")]
        for i, evt in enumerate(data):
            y_nodes = evt["tracks"].ft != 1  # 0 bbar 2 b
            if "frag" in configs["settings"]["graph_mode"]:
                frag_selbool = evt["tracks"].frag != 0
                y_nodes = y_nodes | frag_selbool
            if configs["settings"]["node_sel"] == "true":
                true_node_pruning(y_nodes, evt, "tracks", edge_types)
            elif configs["settings"]["node_sel"] == "default":
                node_pruning(y_nodes, evt, "tracks", edge_types)
            elif configs["settings"]["node_sel"] == "zeros":
                test_node_pruning(y_nodes, evt, "tracks", edge_types)
            if evt[("tracks", "to", "tracks")].y.shape[0] == 0 or torch.all(evt[("tracks", "to", "tracks")].y == 0):
                data_selbool[i] = 0
        filtered_data = [d for d, sel in zip(data, data_selbool) if sel]
    else:
        filtered_data = data

    """Making the graph bidirectional"""
    for data in filtered_data:
        edge_type = ('tracks', 'to', 'tracks')
        store = data[edge_type]

        store.edge_index = torch.cat([store.edge_index, store.edge_index.flip(0)], dim=1)
        store.edges = store.edges.repeat(2, 1)  # More efficient than cat([x]*2)
        store.y = store.y.repeat(2)
        # if exists pred y and lca on edges as well
        if hasattr(store, 'lca') and store.lca is not None:
            store.lca = store.lca.repeat(2, 1)
        if hasattr(store, 'pred_y') and store.lca is not None:
            store.pred_y = store.pred_y.repeat(2)

    """PV asso"""
    if pv_asso_model is not None:
        ncpus = int(configs["settings"]["ncpu"] / 2)
        pv_data = DataLoader(filtered_data, batch_size=512)
        filtered_data = []
        for evt in pv_data:
            if pv_asso_model.name == "pv_asso_module":
                original_data = copy.deepcopy(evt)
                metrics = pv_asso_model.forward(evt)
                res = pv_associate_data(original_data, metrics, node_thr=pv_asso_model.node_thrs, n_cores=ncpus)
            else:
                metrics = pv_asso_model.forward(evt)
                res = pv_associate_data(evt, metrics, n_cores=ncpus)
            filtered_data.append(res)
        filtered_data = list(chain.from_iterable(filtered_data))

    """Whitening for calibration"""
    if configs["settings"]["calibration"]:
        filtered_data = adjust_for_calibration(configs, path, filtered_data)

    if mode == "weights_only":
        weights = get_hetero_weight(filtered_data, configs)
        return weights
    elif "weights" in mode:
        weights = get_hetero_weight(filtered_data, configs)
        return filtered_data, weights
    else:
        return filtered_data
