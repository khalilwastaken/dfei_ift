import torch
from torch_scatter import scatter_add

from tqdm import tqdm



def acc_four_class(pred, label):

    pred_argmax = torch.argmax(pred, dim=1)

    res = {}
    for i in range(4):
        classi_selbool = label == i

        res[f"LCA_class{i}_num"] = torch.sum(classi_selbool).float().item()
        if res[f"LCA_class{i}_num"] == 0:
            res[f"LCA_class{i}_pred_class0"] = 0.
            res[f"LCA_class{i}_pred_class1"] = 0.
            res[f"LCA_class{i}_pred_class2"] = 0.
            res[f"LCA_class{i}_pred_class3"] = 0.
        else:
            res[f"LCA_class{i}_pred_class0"] = torch.sum(pred_argmax[classi_selbool] == 0).item() / res[
                f"LCA_class{i}_num"]
            res[f"LCA_class{i}_pred_class1"] = torch.sum(pred_argmax[classi_selbool] == 1).item() / res[
                f"LCA_class{i}_num"]
            res[f"LCA_class{i}_pred_class2"] = torch.sum(pred_argmax[classi_selbool] == 2).item() / res[
                f"LCA_class{i}_num"]
            res[f"LCA_class{i}_pred_class3"] = torch.sum(pred_argmax[classi_selbool] == 3).item() / res[
                f"LCA_class{i}_num"]
    return res
