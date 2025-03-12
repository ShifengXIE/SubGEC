import random
def sam_nexthop(adj_lists, sam_current, current2, k):
    for neigh in sam_current:
        current2.update(adj_lists[int(neigh)])
    current2 = list(current2 - set(sam_current))
    if len(current2) < (k - len(sam_current)):
        return current2
    else:
        sampled = random.sample(current2, k - len(sam_current))
        sampled.extend(sam_current)  # Add current sampled nodes
        return sampled


def sub_sam(nodes, adj_lists, k):
    node_neighbor = [[] for i in range(nodes.shape[0])]
    node_neighbor_cen = [[] for i in range(nodes.shape[0])]
    node_centorr = [[] for i in range(nodes.shape[0])]

    num_nei = 0
    flag = 0
    for node in nodes:
        neighbors = set([int(node)])
        neighs = adj_lists[int(node)]
        node_centorr[num_nei] = [int(node)]
        current1 = adj_lists[int(node)]
        if len(neighs) >= k:
            neighs -= neighbors
            # Convert set to list before sampling
            current1 = random.sample(list(neighs), k - 1)
            node_neighbor[num_nei] = [neg_node for neg_node in current1]
            current1.append(int(node))
            node_neighbor_cen[num_nei] = [neg_node for neg_node in current1]
            num_nei += 1
        else:
            num_while = 0
            while len(current1) < k:
                current2 = set()
                current1 = sam_nexthop(adj_lists, current1, current2, k)
                if num_while > 3:
                    break
                num_while += 1
            if num_while > 3:
                flag += 1
                continue

            node_neighbor_cen[num_nei] = [neg_node for neg_node in current1]
            if int(node) in node_neighbor_cen[num_nei]:
                node_neighbor_cen[num_nei].remove(int(node))
            node_neighbor[num_nei] = random.sample(node_neighbor_cen[num_nei], k - 1)
            node_neighbor_cen[num_nei] = [neg_node for neg_node in node_neighbor[num_nei]]
            node_neighbor_cen[num_nei].append(int(node))
            num_nei += 1

    if flag > 0:
        node_neighbor_new = [[] for i in range(len(node_neighbor) - flag)]
        node_neighbor_cen_new = [[] for i in range(len(node_neighbor) - flag)]
        node_centorr_new = [[] for i in range(len(node_neighbor) - flag)]
        for i in range(len(node_neighbor) - flag):
            node_neighbor_new[i] = node_neighbor[i]
            node_neighbor_cen_new[i] = node_neighbor_cen[i]
            node_centorr_new[i] = node_centorr[i]
        return node_neighbor_cen_new

    return node_neighbor_cen
