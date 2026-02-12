import torch

def find_components_bfs(edge_index):
    # breadth first search
    edge_index = edge_index.numpy()
    all_nodes = set(edge_index[0]).union(set(edge_index[1]))

    adj_list = {node: set() for node in all_nodes}
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj_list[src].add(dst)
        adj_list[dst].add(src)

    visited = set()
    components = []
    for start_node in all_nodes:
        if start_node in visited:
            continue

        component = set()
        queue = [start_node]

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue

            visited.add(node)
            component.add(node)

            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append([int(node) for node in component])

    return components

def pv_association(edge_index, pv_desc, edge_selbool=None):
    ntracks = edge_index[0].max().item() + 1
    npvs = edge_index[1].max().item() + 1
    pv_desc = pv_desc.view(ntracks, npvs)
    pred_pv = torch.argmax(pv_desc, dim=1)

    if edge_selbool: # doing some correction
        # Should keep the other pvs if passed to a second dfei/ift to reasso and deal with miss asso
        # honestly one should keep the other pvs inside the event as well, as they can help to deal with miss asso

        # Current strategy which should be bis tbh:
        # Full DFEI -> associate ip without any ip correction on B reco
        # Second DFEI on single pp -> reduce noneiso bkg
        # IFT
        # In principal Full DFEI and second DFEI can be combined given how fast the pv asso module on gpu is
        # But at the same time it is "slow"
        # Possible solution: just apply pv asso in between for example GNN layer 2 and 3 and attach the origin graph
        # I think for now we just go with two DFEI model and continue from there
        b_edge_idx = edge_index[:, edge_selbool]
        b_systems = find_components_bfs(b_edge_idx)
        for b in b_systems:
            pred_pv[b] = torch.argmax(torch.sum(pred_desc[b], dim=0))
    import pdb; pdb.set_trace()
    return pred_pv