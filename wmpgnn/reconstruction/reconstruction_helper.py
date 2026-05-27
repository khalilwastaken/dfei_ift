import torch
from wmpgnn.util.bfs import *


def add_lca(components, lca):
    for component in components:
        component['lca'] = lca[component['edge_indices']]
    return components


def apply_reco_mapping(graph, components, edges):
    node_idx = torch.arange(graph['tracks'].x.shape[0])
    idx_to_key = dict(zip(node_idx.tolist(), graph['tracks'].part_keys.tolist()))

    for component in components:
        component['part_keys'] = torch.tensor([idx_to_key.get(node, -1) for node in component['nodes']])
        component['keys_edges'] = torch.tensor(
            [[idx_to_key.get(n.item(), -1) for n in row] for row in edges[:, component['edge_indices']]])
    return components


def apply_true_mapping(graph, components, edges):
    key_to_id = dict(zip(graph['tracks'].sig_keys.tolist(), graph['tracks'].sig_ids.tolist()))
    key_to_head_key = dict(zip(graph['tracks'].sig_keys.tolist(), graph['tracks'].head_keys.tolist()))
    key_to_head_id = dict(zip(graph['tracks'].sig_keys.tolist(), graph['tracks'].head_ids.tolist()))

    for component in components:
        component['part_id'] = torch.tensor([key_to_id.get(node, -1) for node in component['nodes']])
        component['head_key'] = [key_to_head_key.get(node, -1) for node in component['nodes']]
        component['head_id'] = [key_to_head_id.get(node, -1) for node in component['nodes']]

        # Safety checks
        if len(set(component['head_id'])) != 1 or len(set(component['head_key'])) != 1:
            import pdb;
            pdb.set_trace()
        component['head_key'] = component['head_key'][0]
        component['head_id'] = component['head_id'][0]

        component['edges'] = edges[:, component['edge_indices']]
    return components


def get_reco_type(true_component, reco_component, sig_dict):
    # Check if they have all particles
    reco_keys = torch.sort(reco_component['part_keys']).values
    true_keys = torch.sort(torch.tensor(true_component['nodes'])).values
    if torch.equal(reco_keys, true_keys):  # all particles found
        sig_dict["AllParticles"] = 1

        # Check LCA if they are perfectly recoed
        reco_cantor = torch_signed_cantor_pair(reco_component['keys_edges'][0], reco_component['keys_edges'][1])
        reco_values, reco_indices = torch.sort(reco_cantor)

        # make true bidirectional first
        senders = torch.cat([true_component['edges'][0], true_component['edges'][1]])
        receivers = torch.cat([true_component['edges'][1], true_component['edges'][0]])
        true_lca = torch.cat([true_component['lca'], true_component['lca']])
        true_cantor = torch_signed_cantor_pair(senders, receivers)
        true_values, true_indices = torch.sort(true_cantor)

        matching_edges = torch.equal(reco_values, true_values)
        matching_lca = torch.equal(reco_component['lca'][reco_indices], true_lca[true_indices])
        if matching_edges and matching_lca:
            sig_dict["PerfectReco"] = 1
        sig_dict['NumBkgParticles'] = -999
        return sig_dict, True

    # Check if it is part reco or none iso
    true_in_reco = torch.sum(torch.isin(true_keys, reco_keys)) / true_keys.shape[0]
    if true_in_reco == 1:  # background tracks in signal
        sig_dict["NoneIso"], sig_dict["PartReco"] = 1, 0
        sig_dict['NumBkgParticles'] = true_keys.shape[0] - reco_keys.shape[0]
        return sig_dict, True
    elif 0.2 <= true_in_reco:  # at least 20% in part reco
        sig_dict["PartReco"] = 1  # FT decision can not be trusted
        truth_set, recon_set = set(true_keys.tolist()), set(reco_keys.tolist())
        sig_dict['NumBkgParticles'] = float(f"{len(recon_set & truth_set)}.{len(recon_set - truth_set)}")

    return sig_dict, False
