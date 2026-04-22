import io
import zstandard as zstd

import torch

from wmpgnn.util.pruners import *


def get_nfiles(_configs, prefix=""):
    nfiles = {}
    for sample, nfile in zip(_configs[f"{prefix}sample"], _configs[f"{prefix}nfiles"]):
        nfiles[sample] = nfile
    return nfiles


def load_file(path):
    dctx = zstd.ZstdDecompressor()
    with open(path, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            decompressed = reader.read()
            data = torch.load(io.BytesIO(decompressed), weights_only=False)
    return data


def initial_pruning(data, configs):
    data_selbool = torch.ones(len(data))
    edge_types = [("tracks", "to", "tracks"), ("tracks", "to", "pvs")]
    for i, evt in enumerate(data):
        y_nodes = evt["tracks"].ft != 1  # 0 bbar 2 b
        if "frag" in configs["graph_mode"]:
            frag_selbool = evt["tracks"].frag != 0
            y_nodes = y_nodes | frag_selbool
        if configs["node_sel"] == "true":
            true_node_pruning(y_nodes, evt, "tracks", edge_types)
        elif configs["node_sel"] == "default":
            node_pruning(y_nodes, evt, "tracks", edge_types)
        elif configs["node_sel"] == "zeros":
            test_node_pruning(y_nodes, evt, "tracks", edge_types)
        if evt[("tracks", "to", "tracks")].y.shape[0] == 0 or torch.all(evt[("tracks", "to", "tracks")].y == 0):
            data_selbool[i] = 0
    filtered_data = [d for d, sel in zip(data, data_selbool) if sel]
    return filtered_data
