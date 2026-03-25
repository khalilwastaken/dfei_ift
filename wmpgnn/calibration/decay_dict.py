import torch

# Just as a reminder both Jpsi, Phi, Kst have really short lifetime which can not be resolved thus are merged
decay_prop = {  # Decay properties: True final mc id, lca score
    "Bs_Jpsiphi": [
        torch.tensor([-13, 13, -321, 321]), torch.tensor([1, 1, 1, 1, 1, 1])
    ],
    "Bd_JpsiKst": [
        torch.tensor([-13, 13, 211, -321]), torch.tensor([1, 1, 1, 1, 1, 1])
    ],
    "Bu_JpsiK": [
        torch.tensor([-13, 13, 321]), torch.tensor([1, 1, 1])
    ]

}

# Translate pid name to mc numbering
pid_dict = {
    "Prob_k": 321,
    "Prob_e": 11,
    "Prob_mu": 13,
    "Prob_p": 2212,
    "Prob_pi": 211,
    "Prob_ghost": -1
}
