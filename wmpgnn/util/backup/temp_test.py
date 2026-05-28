# probably rename to data_augmentation

import torch
import copy


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


def adjust_data(configs, path, list_data):
    # copy pi pid response to K in K* decay
    # faking a pi pi or K K decay -> no pid information not selftagging except for kinematics
    if False and configs:
        for data in list_data:
            edges = torch.stack([data['truth_senders'], data['truth_receivers']])
            components = find_components_bfs(edges)
            for comp in components:
                keys = data['truth_part_keys'][comp]
                final_states = torch.sort(data['truth_part_ids'][comp]).values
                edge_mask = (edges[0] == comp[0]) | (edges[1] == comp[0])

                bd_jpsikst = torch.tensor([-13, 13, 211, -321])
                if torch.equal(final_states, torch.sort(bd_jpsikst).values) or torch.equal(final_states, torch.sort(
                        -1 * bd_jpsikst).values):
                    if torch.abs(data['truth_moth_ids'][edge_mask])[0].item() == 511:
                        # replacing K with pi response
                        # source_mask = torch.abs(data['truth_part_ids'][comp]) == 211
                        # target_mask = torch.abs(data['truth_part_ids'][comp]) == 321
                        # replacing pi with k response
                        source_mask = torch.abs(data['truth_part_ids'][comp]) == 321
                        target_mask = torch.abs(data['truth_part_ids'][comp]) == 211

                        source = keys[source_mask].item()  # pion
                        source_bool = data['final_keys'] == source
                        # source_x = data['tracks'].x[source_bool]
                        source_pid = data['tracks'].pid[source_bool]

                        target = keys[target_mask].item()  # Kaon
                        target_bool = data['final_keys'] == target
                        # data['tracks'].x[target_bool] = source_x  # coping pid response is sufficient as it is purely kinematic and this would flip the charge
                        data['tracks'].pid[target_bool] = source_pid
                else:
                    continue
        return list_data

    # flip K* decay mode and dupe
    if False and configs:
        res = []
        for data in list_data:
            charge_flipped_data = copy.deepcopy(data)
            flipped = False

            edges = torch.stack([data['truth_senders'], data['truth_receivers']])
            components = find_components_bfs(edges)
            for comp in components:
                keys = data['truth_part_keys'][comp]
                final_states = torch.sort(data['truth_part_ids'][comp]).values
                edge_mask = (edges[0] == comp[0]) | (edges[1] == comp[0])

                bd_jpsikst = torch.tensor([-13, 13, 211, -321])
                if torch.equal(final_states, torch.sort(bd_jpsikst).values) or torch.equal(final_states, torch.sort(
                        -1 * bd_jpsikst).values):
                    if torch.abs(data['truth_moth_ids'][edge_mask])[0].item() == 511:
                        target_pion_mask = torch.abs(data['truth_part_ids'][comp]) == 211
                        target_pion = keys[target_pion_mask].item()  # pion
                        target_pion_bool = data['final_keys'] == target_pion
                        pion_charge = data['tracks'].x[target_pion_bool][-1][-1].item()
                        target_kaon_mask = torch.abs(data['truth_part_ids'][comp]) == 321
                        target_kaon = keys[target_kaon_mask].item()  # Kaon
                        target_kaon_bool = data['final_keys'] == target_kaon
                        kaon_charge = data['tracks'].x[target_kaon_bool][-1][-1].item()

                        flipped = True
                        indices = target_kaon_bool.nonzero(as_tuple=True)[0]
                        assert len(indices) == 1, "Expected exactly one matching track"
                        target_idx = indices[0]
                        charge_flipped_data["tracks"].x[target_idx, -1] = pion_charge

                        indices = target_pion_bool.nonzero(as_tuple=True)[0]
                        assert len(indices) == 1, "Expected exactly one matching track"
                        target_idx = indices[0]
                        charge_flipped_data["tracks"].x[target_idx, -1] = kaon_charge

            if flipped:
                res.append(data)
                res.append(charge_flipped_data)
        return res

    # Adding a whitening
    # The idea is to take the decay which we want to calibrate on and whiten the charge and pid information
    # and see the performance
    if True and configs:
        if "Bd_JpsiKst" in path:
            for data in list_data:
                edges = torch.stack([data['truth_senders'], data['truth_receivers']])
                components = find_components_bfs(edges)
                for comp in components:
                    final_keys = data['final_keys']
                    final_states = torch.sort(data['truth_part_ids'][comp]).values
                    edge_mask = (edges[0] == comp[0]) | (edges[1] == comp[0])

                    bd_jpsikst = torch.tensor([-13, 13, 211, -321])
                    if torch.equal(final_states, torch.sort(bd_jpsikst).values) or torch.equal(final_states, torch.sort(
                            -1 * bd_jpsikst).values):
                        if torch.abs(data['truth_moth_ids'][edge_mask])[0].item() == 511:
                            truth_key = data["truth_part_keys"][comp]
                            mask = torch.isin(final_keys, truth_key)
                            data["tracks"].x[mask, -1] = 0
                            data["tracks"].pid[mask] = 0
        return list_data


    raise NotImplementedError
