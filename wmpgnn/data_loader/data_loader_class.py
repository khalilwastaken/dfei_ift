import copy
from itertools import chain

from torch_geometric.loader import DataLoader

from wmpgnn.pv_association.pv_association import pv_associate_data
from wmpgnn.calibration.calibration_mask import *
from wmpgnn.data_loader.weights_calculator import get_hetero_weight
from wmpgnn.data_loader.helper import *
from wmpgnn.calibration.calibration_mask import *
from wmpgnn.util.pruners import *


class DataSetLoader:
    def __init__(self, configs, pv_model=None):
        self.configs = configs

        # PV model
        self.pv_model = pv_model
        # Whitening module for calibration
        self.calibration_class = None
        if configs['settings']['calibration']:
            # The data passed to calibration needs to hold pred_y on track nodes and tr-tr edges with lca information
            self.calibration_class = CalibrationClass(configs)

        self.edge_types = [("tracks", "tracks"), ("tracks", "pvs")]

    def load_data(self, path, mode="train"):
        data = load_file(path)

        """Applying pruning for different using truth pruning initially"""
        if "true" in self.configs["settings"]["graph_mode"]:
            data = initial_pruning(data, self.configs["settings"])

        """Apply cuts on the graph"""
        """# apply cut currently done manually TODO: change this
        for evt in data:
            ghost_selbool = evt["tracks"].x.T[-1] < 0.6
            angle_selbool = evt[("tracks", "tracks")].edges[:, 1] < 0.0005
            if torch.sum(angle_selbool) != 0:
                indx = evt[("tracks", "tracks")].edge_index[:, angle_selbool]
                for j in indx.T:
                    ghost_selbool[j[torch.argmax(evt["tracks"].x[j].T[-1])].item()] = False
            true_node_pruning(ghost_selbool, evt, "tracks", self.edge_types)"""

        """Making the graph bidirectional"""
        for evt in data:
            store = evt[('tracks', 'tracks')]

            store.edge_index = torch.cat([store.edge_index, store.edge_index.flip(0)], dim=1)
            store.edges = store.edges.repeat(2, 1)  # More efficient than cat([x]*2)
            if hasattr(store, 'y') and store.y is not None:
                store.y = store.y.repeat(2)
            if hasattr(store, 'lca') and store.lca is not None:
                store.lca = store.lca.repeat(2, 1)
            if hasattr(store, 'pred_y') and store.lca is not None:
                store.pred_y = store.pred_y.repeat(2)

        """PV association"""
        if self.pv_model is not None:
            pv_data = DataLoader(data, batch_size=1024)
            data = []
            for evt in pv_data:
                if self.pv_model.name == "pv_asso_module":
                    original_data = copy.deepcopy(evt)
                    metrics = self.pv_model.forward(evt)
                    res = pv_associate_data(original_data, metrics, node_thr=self.pv_model.node_thrs,
                                            n_cores=self.configs['ncpus']['pv_asso'])
                else:
                    metrics = self.pv_model.forward(evt)
                    res = pv_associate_data(evt, metrics, n_cores=self.configs['ncpus']['pv_asso'])
                data.append(res)
            data = list(chain.from_iterable(data))

        """Whitening for calibration"""
        if self.configs["settings"]["calibration"]:
            data = self.calibration_class.remove_sig(path, data)


        """Domain adaptation labeling"""
        if self.configs["settings"].get('domain_adapt', False):
            if self.configs["settings"]["da_data_dir"] in path:
                if 'tst' not in path:
                    label = torch.tensor([1.], dtype=torch.float32)  # data label
                else:
                    label = torch.tensor([0.], dtype=torch.float32)  # set to 0 to run dfei during testing
            else:
                label = torch.tensor([0.], dtype=torch.float32)  # MC label
            for evt in data:
                evt["da_label"] = label

        if mode == "weights_only":
            weights = get_hetero_weight(data, self.configs["inference"])
            return weights
        elif "weights" in mode:
            weights = get_hetero_weight(data, self.configs["inference"])
            return data, weights
        else:
            return data
