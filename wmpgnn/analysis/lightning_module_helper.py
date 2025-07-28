import torch

def init_logs(configs):
    trn_log = {}
    if configs["LCA"]:
        trn_log["LCA_loss"] = []
        for i in range(4):  # something like num classes in config file
            trn_log[f"LCA_class{i}_num"] = []
            for j in range(4):
                trn_log[f"LCA_class{i}_pred_class{j}"] = []

    return trn_log, trn_log



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