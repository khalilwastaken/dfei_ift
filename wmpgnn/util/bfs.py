import torch

# Implementation of bfs to find isolated systems
def find_components_bfs(edge_index):
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

        node_list = [int(node) for node in component]
        # find edge indices where both src and dst are in this component
        edge_indices = [
            i for i in range(edge_index.shape[1])
            if edge_index[0, i] in component and edge_index[1, i] in component
        ]
        components.append({"nodes": node_list, "edge_indices": edge_indices})

    return components


def torch_signed_cantor_pair(a, b):
    pair = (a + b) * (a + b + 1) // 2 + b
    return torch.where(a < b, -pair, pair)