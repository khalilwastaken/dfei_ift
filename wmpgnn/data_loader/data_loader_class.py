import copy
from itertools import chain

from torch_geometric.loader import DataLoader

from wmpgnn.pv_association.pv_association import pv_associate_data
from wmpgnn.calibration.calibration_mask import *
from wmpgnn.data_loader.weights_calculator import get_hetero_weight
from wmpgnn.data_loader.helper import *
from wmpgnn.calibration.calibration_mask import *


class DataSetLoader:
    def __init__(self, configs, pv_model=None):
        self.configs = configs
        self.pv_model = pv_model
        self.calibration_class = None
        if configs['settings'].get('calibration', False):
            # The data passed to calibration needs to hold pred_y on track nodes and tr-tr edges with lca information
            self.calibration_class = CalibrationClass(configs)
        self.edge_types = [("tracks", "tracks"), ("tracks", "pvs")]
        self.make_bidirectional = configs['settings'].get('make_bi', True)
        
    def load_data(self, path, mode="train"):
        data = load_file(path)

        if "true" in self.configs["settings"].get("graph_mode", ""):
            data = initial_pruning(data, self.configs["settings"])

        if self.make_bidirectional:
            for evt in data:
                self._make_bidirectional(evt)

        if self.pv_model is not None:
            data = self._run_pv_association(data)

        if self.calibration_class is not None:
            data = self.calibration_class.remove_sig(path, data)

        if mode == "weights_only":
            return get_hetero_weight(data, self.configs["inference"])
        if "weights" in mode:
            return data, get_hetero_weight(data, self.configs["inference"])
        return data

    @staticmethod
    def _make_bidirectional(evt):
        """Extend track-track edges so the graph is undirected."""
        store = evt[("tracks", "tracks")]
        store.edge_index = torch.cat(
            [store.edge_index, store.edge_index.flip(0)], dim=1
        )
        store.edges = store.edges.repeat(2, 1)
        if getattr(store, "y", None) is not None:
            store.y = store.y.repeat(2)
        if getattr(store, "lca", None) is not None:
            store.lca = store.lca.repeat(2, 1)
        if getattr(store, "pred_y", None) is not None:
            store.pred_y = store.pred_y.repeat(2)

    def _run_pv_association(self, data):
        """Batch-process PV association and return a flat list of events."""
        results = []
        for batch in DataLoader(data, batch_size=1024):
            if self.pv_model.name == "pv_asso_module":
                original = copy.deepcopy(batch)
                metrics = self.pv_model.forward(batch)
                res = pv_associate_data(original, metrics, node_thr=self.pv_model.node_thrs,
                                        n_cores=self.configs["ncpus"]["pv_asso"])
            else:
                metrics = self.pv_model.forward(batch)
                res = pv_associate_data(batch, metrics, n_cores=self.configs["ncpus"]["pv_asso"])
            results.append(res)
        return list(chain.from_iterable(results))

    # Define function to do pre selection
    # define function for da and so on
