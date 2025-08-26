import re, glob

import torch
import pandas as pd


def init_logs(configs, mode="train"):
    loss_config = configs["training"]["infer"]
    gn_blocks = configs["model"]["GNblocks"]["nBlocks"]
    ft_layers = configs["model"]["GNblocks"]["FTlayers"]

    log = {"combined_loss": []}
    if loss_config["LCA"]:
        log["LCA_loss"] = []
        for i in range(4):  # something like num classes in config file
            log[f"LCA_class{i}_num"] = []
            for j in range(4):
                log[f"LCA_class{i}_pred_class{j}"] = []

    if loss_config["node_prune"]:
        log["t_nodes_loss"] = []
        if mode == "test":
            for i in range(gn_blocks):
                log[f"sig_nodes_score_{i}"] = torch.tensor([])
                log[f"bkg_nodes_score_{i}"] = torch.tensor([])

    if loss_config["edge_prune"]:
        log["tt_edges_loss"] = []
        if mode == "test":
            for i in range(gn_blocks):
                log[f"sig_edges_score_{i}"] = torch.tensor([])
                log[f"bkg_edges_score_{i}"] = torch.tensor([])

    if loss_config["FT"]:
        log["ft_loss"] = []
        if mode == "test":
            for i in range(ft_layers):
                log[f"b_ft_score_{gn_blocks - i - 1}"] = torch.tensor([])
                log[f"bbar_ft_score_{gn_blocks - i - 1}"] = torch.tensor([])

    if loss_config["frag"]:
        log["frag_loss"] = []
        if mode == "test":
            for i in range(ft_layers):
                log[f"sig_frag_score_{gn_blocks - i - 1}"] = torch.tensor([])
                log[f"bkg_frag_score_{gn_blocks - i - 1}"] = torch.tensor([])

    if mode == "train":
        return log, log
    elif mode == "test":
        return log


def init_test_df():
    signal_df = pd.DataFrame(
        columns=['EventNumber',
                 'NumParticlesInEvent', 'NumSignalParticles', 'NumBkgParticles_noniso',
                 'PerfectSignalReconstruction', 'AllParticles', 'PerfectReco', 'NoneIso', 'PartReco', 'NotFound',
                 'SigMatch',
                 'B_id', 'Pred_FT',
                 'reco_pv_idx', 'true_pv_idx'
                 ])

    event_df = pd.DataFrame(
        columns=['EventNumber', 'NumParticlesInEvent', 'NumParticlesFromHeavyHadronInEvent',
                 'NumBackgroundParticlesInEvent', 'NumSelectedParticlesInEvent',
                 'NumSelectedParticlesFromHeavyHadronInEvent',
                 'NumSelectedBackgroundParticlesInEvent', 'NumTruthClustersGen1', 'NumTruthClustersGen2',
                 'NumTruthClustersGen3', 'NumTruthClustersGen4', 'NumRecoClustersGen1', 'NumRecoClustersGen2',
                 'NumRecoClustersGen3', 'NumRecoClustersGen4', 'MaxTruthFullChainDepthInEvent',
                 'EfficiencyParticlesFromHeavyHadronInEvent', 'EfficiencyBackgroundParticlesInEvent',
                 'BackgroundRejectionPowerInEvent', 'PerfectEventReconstruction', 'TimeNodeFiltering',
                 'TimeEdgeFiltering',
                 'TimeLCAReconstruction', 'TimeSequence', 'NumTrueSignalsInEvent', 'NumRecoSignalsInEvent',
                 'TimeModel', 'TimeReco', 'TimeTruth'
                 ])

    return signal_df, event_df


def init_loss(device):
    # Later add config
    loss = {"LCA": torch.tensor(0., device=device), "t_nodes": torch.tensor(0., device=device),
            "tt_edges": torch.tensor(0., device=device),
            "tPV_edges": torch.tensor(0., device=device), "frag_nodes": torch.tensor(0., device=device),
            "ft_nodes": torch.tensor(0., device=device)}
    return loss


def epoch_end_loggable(log):
    # keys of the loss functions
    avg_log = {}
    loss_keys = [k for k in log if k.endswith('_loss')]
    for key in loss_keys:
        avg_log[key] = torch.tensor(log[key]).nanmean(dim=0)

    class_numbers = sorted(set(
        int(m.group(1)) for k in log if (m := re.search(r'LCA_class(\d+)_num', k))
    ))

    # LCA logging
    for score in class_numbers:
        selbool = torch.tensor(log[f"LCA_class{score}_num"]) != 0
        avg_log[f"LCA_class{score}_num"] = torch.tensor(log[f"LCA_class{score}_num"]).nanmean(dim=0)
        for s in class_numbers:
            avg_log[f"LCA_class{score}_pred_class{s}"] = torch.nan_to_num(
                torch.tensor(log[f"LCA_class{score}_pred_class{s}"])[selbool].nanmean(dim=0), nan=-1)

    return avg_log


def loss_logging(log, loss, configs):
    if configs["node_prune"]:
        log["t_nodes_loss"].append(loss["t_nodes"].item())
    if configs["edge_prune"]:
        log["tt_edges_loss"].append(loss["tt_edges"].item())
    if configs["frag"]:
        log["frag_loss"].append(loss["frag_nodes"].item())
    if configs["FT"]:
        log["ft_loss"].append(loss["ft_nodes"].item())
    return log


def make_loggable(hparams_dict):
    loggable = {}
    for k, v in hparams_dict.items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                loggable[k] = v.item()  # convert scalar tensor to float
            else:
                loggable[k] = v.tolist()  # convert vector/matrix to list
        else:
            loggable[k] = str(v)  # fallback to string
    return loggable


def get_model_path(config):
    try:
        version, model = config.split("_")
    except:
        raise RuntimeError("Could not obtain version + model in the format of version_model")

    path = f"lightning_logs/version_{version}/checkpoints/"
    if model == "bis":
        path = glob.glob(path + "best-epoch=*")
    else:
        path = path + f"epoch-epoch={str(model).zfill(2)}.ckpt"

    return path
