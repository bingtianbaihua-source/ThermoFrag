"""Coupling potential V^couple_theta(m).

Graph energy on the fragment-assembly graph. Minimal implementation:
  - integer fragment IDs embedded into node features
  - integer bond types embedded as edge features
  - stack of GINEConv layers, each with residual + LayerNorm
  - Set2Set readout to a graph-level vector, scalar MLP head

Trained with persistent contrastive divergence, see training/pcd.py and
docs/METHOD.md eq (5).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, Set2Set


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_hidden), nn.SiLU(), nn.Linear(d_hidden, d_out)
    )


class CouplingPotential(nn.Module):
    """Scalar graph energy V(m) on fragment-assembly graphs.

    Expects a PyG Batch with:
        frag_id:   [N] long   fragment-vocabulary indices
        edge_index: [2, E]
        bond_type: [E] long   bond-type indices
        batch:     [N] long
    """

    def __init__(
        self,
        n_fragments: int = 1024,
        n_bond_types: int = 8,
        hidden: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.node_embed = nn.Embedding(n_fragments, hidden)
        self.edge_embed = nn.Embedding(n_bond_types, hidden)

        self.convs = nn.ModuleList(
            [GINEConv(_mlp(hidden, hidden, hidden), edge_dim=hidden) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.readout = Set2Set(hidden, processing_steps=3)
        self.head = _mlp(2 * hidden, hidden, 1)

    def forward(self, batch) -> torch.Tensor:
        x = self.node_embed(batch.frag_id)
        ea = self.edge_embed(batch.bond_type)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, batch.edge_index, ea)
            x = norm(x + h)
        g = self.readout(x, batch.batch)  # [B, 2*hidden]
        return self.head(g).squeeze(-1)  # [B]
