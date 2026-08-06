"""Collate per-arena graph snapshots into one batched graph for the critic.

Offsets each arena's edge indices and concatenates, so a single critic forward
pass covers every arena in a step.
"""

import torch


def build_critic_batch(nodes, edge_indices, edge_attrs, zs):
    x = torch.cat(nodes, dim=0)
    z = torch.stack(zs, dim=0)

    batch_parts = []
    edge_parts = []
    attr_parts = []
    offset = 0
    for i in range(len(nodes)):
        m = nodes[i].shape[0]
        batch_parts.append(torch.full((m,), i, dtype=torch.long, device=nodes[i].device))
        edge_parts.append(edge_indices[i] + offset)
        attr_parts.append(edge_attrs[i])
        offset += m

    batch = torch.cat(batch_parts, dim=0)
    edge_index = torch.cat(edge_parts, dim=1)
    edge_attr = torch.cat(attr_parts, dim=0)
    return x, edge_attr, edge_index, z, batch
