import re

import torch

import pandas as pd


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


def init_logs(configs, mode="train", model="DFEI"):
    loss_config = configs[model]["inference"]

    log = {}
    if model == "DFEI":
        log["combined_loss"] = []
        gn_blocks = configs[model]["GNblocks"]["nBlocks"]
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

        if  loss_config["pv_asso"]:
            log["tpv_edges_loss"] = []
            if mode == "test":
                for i in range(gn_blocks):
                    log[f"sig_pv_asso_score_{i}"] = torch.tensor([])
                    log[f"bkg_pv_asso_score_{i}"] = torch.tensor([])

                log["pv_corr_ml"] = {}
                log["pv_corr_ip"] = {}
                log["pv_total"] = {}
    elif model == "IFT":
        if loss_config["FT"]:
            log["ft_loss"] = []
    if mode == "train":
        return log, log
    elif mode == "test":
        return log
    else:
        raise Exception(f"Unknown mode {mode}")


def init_loss(device):
    loss = {"LCA": torch.tensor(0., device=device),
            "t_nodes": torch.tensor(0., device=device),
            "tt_edges": torch.tensor(0., device=device),
            "tPV_edges": torch.tensor(0., device=device),
            "ft_nodes": torch.tensor(0., device=device),
            "pv_asso": torch.tensor(0., device=device)}
    return loss


def init_test_df():
    signal_df = pd.DataFrame()
    event_df = pd.DataFrame()
    return signal_df, event_df


def get_block_score(log, weights, y, layer, var):
    sig_selbool = (y == 1).squeeze()
    log[f"sig_{var}_score_{layer}"] = torch.cat([weights[sig_selbool].cpu(), log[f"sig_{var}_score_{layer}"]])
    log[f"bkg_{var}_score_{layer}"] = torch.cat([weights[~sig_selbool].cpu(), log[f"bkg_{var}_score_{layer}"]])


def loss_logging(log, loss, configs, mode="DFEI"):
    if mode == "DFEI":
        if configs["node_prune"]:
            log["t_nodes_loss"].append(loss["t_nodes"].item())
        if configs["edge_prune"]:
            log["tt_edges_loss"].append(loss["tt_edges"].item())
        if configs["pv_asso"]:
            log["tpv_edges_loss"].append(loss["pv_asso"].item())
    elif mode == "IFT":
        if configs["frag"]:
            log["frag_loss"].append(loss["frag_nodes"].item())
        if configs["FT"]:
            log["ft_loss"].append(loss["ft_nodes"].item())
    return log


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
