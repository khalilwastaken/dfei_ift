import torch
from torch_geometric.data import HeteroData

SCHEMA = {
    'nodes': {
        'tracks': {
            'x':   ('tracks', 'x',   1),
            'pid': ('tracks', 'pid', 1),
            'ft':  ('tracks', 'ft',  None),  # 1D, store ref for dtype
        },
        'pvs': {
            'x': ('pvs', 'x', 1),
        },
        'globals': {
            'x': ('globals', 'x', 1),
        },
    },
    'edges': {
        ('tracks', 'to', 'tracks'): {
            'edges': (('tracks', 'to', 'tracks'), 'edges', 1),
            'y':     (('tracks', 'to', 'tracks'), 'y',     None),
        },
        ('tracks', 'to', 'pvs'): {
            'edges': (('tracks', 'to', 'pvs'), 'edges', 1),
            'y':     (('tracks', 'to', 'pvs'), 'y',     1),
            'filter':(('tracks', 'to', 'pvs'), 'filter', None),
        },
    },
    'graph': ['EVENTNUMBER', 'RUNNUMBER'],
}


def _zeros_like_shape(n_rows, ref_val, dim_idx):
    """Create a zero tensor matching ref_val's dtype and shape."""
    dtype = ref_val.dtype
    if dim_idx is None or ref_val.dim() == 1:
        return torch.zeros(n_rows, dtype=dtype)
    else:
        feat_dim = ref_val.shape[dim_idx]
        return torch.zeros(n_rows, feat_dim, dtype=dtype)


def unify_heterodata(data: HeteroData, ref: HeteroData) -> HeteroData:
    # --- Node stores ---
    for node_type, attrs in SCHEMA['nodes'].items():
        store = data[node_type]
        ref_store = ref[node_type]
        n = store.num_nodes

        for attr, spec in attrs.items():
            if store.get(attr) is not None:
                continue
            if spec is None:
                store[attr] = torch.zeros(n)
                continue
            _, ref_attr, dim_idx = spec
            ref_val = ref_store.get(ref_attr)
            if ref_val is None:
                continue
            store[attr] = _zeros_like_shape(n, ref_val, dim_idx)

    # --- Edge stores ---
    for edge_type, attrs in SCHEMA['edges'].items():
        store = data[edge_type]
        ref_store = ref[edge_type]
        n = store.edge_index.shape[1]

        for attr, spec in attrs.items():
            if store.get(attr) is not None:
                continue
            if spec is None:
                store[attr] = torch.zeros(n)
                continue
            _, ref_attr, dim_idx = spec
            ref_val = ref_store.get(ref_attr)
            if ref_val is None:
                continue
            store[attr] = _zeros_like_shape(n, ref_val, dim_idx)

    # --- Graph-level scalars ---
    for attr in SCHEMA['graph']:
        if data.get(attr) is None:
            ref_val = ref.get(attr)
            data[attr] = torch.zeros_like(ref_val) if ref_val is not None \
                         else torch.zeros(1, dtype=torch.long)

    return data