import pandas as pd
import torch


def get_normalization(configs):
    if configs['settings']['norm'] == 'None': return None
    return pd.DataFrame(torch.load(configs['settings']['norm']))

def denorm_data(data, norm, var):
    norm = norm.T[var]
    device = data.device
    center = torch.tensor(norm.loc['center'].values, device=device)
    scale = torch.tensor(norm.loc['scale'].values, device=device)
    return data * scale + center
