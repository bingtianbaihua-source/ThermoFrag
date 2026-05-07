"""PaiNN equivariant backbone (Schütt, Unke, Gastegger 2021).

Minimal self-contained implementation. Per-atom features are a pair
(s_i, v_i) with s_i in R^F and v_i in R^{F,3}. Each layer applies a
message block (edge messages conditioned on radial basis + cosine cutoff)
followed by an equivariant update block (scalar-vector gating).

Reference: docs/METHOD.md eq (4) and section 5.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph
from torch_scatter import scatter_add


@dataclass
class PaiNNConfig:
    hidden: int = 128
    num_layers: int = 4
    cutoff: float = 5.0
    n_radial: int = 20
    elements: tuple[str, ...] = ("H", "C", "N", "O", "F", "P", "S", "Cl", "Br")
    max_z: int = 100  # embedding table size; covers periodic table up to Fm


def _element_to_z(elements: tuple[str, ...]) -> tuple[int, ...]:
    table = {
        "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
        "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
        "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Br": 35, "I": 53,
    }
    return tuple(table[e] for e in elements)


class CosineCutoff(nn.Module):
    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        f = 0.5 * (torch.cos(torch.pi * d / self.cutoff) + 1.0)
        return f * (d < self.cutoff).to(d.dtype)


class BesselRBF(nn.Module):
    """Sine Bessel RBF used in PaiNN: phi_k(d) = sin(k*pi*d/rc) / d."""

    def __init__(self, n_radial: int, cutoff: float):
        super().__init__()
        freqs = torch.arange(1, n_radial + 1, dtype=torch.float32) * torch.pi / cutoff
        self.register_buffer("freqs", freqs)
        self.cutoff = cutoff

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        # d: [E]; output: [E, n_radial]
        d = d.clamp_min(1e-8).unsqueeze(-1)
        return torch.sin(self.freqs * d) / d


class PaiNNMessage(nn.Module):
    def __init__(self, hidden: int, n_radial: int, cutoff: float):
        super().__init__()
        self.hidden = hidden
        self.phi = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3 * hidden)
        )
        self.W = nn.Linear(n_radial, 3 * hidden)
        self.cutoff_fn = CosineCutoff(cutoff)

    def forward(self, s, v, edge_index, r_ij, d_ij, rbf):
        src, dst = edge_index[0], edge_index[1]  # messages from src -> dst
        phi_j = self.phi(s[src])  # [E, 3F]
        W_d = self.W(rbf) * self.cutoff_fn(d_ij).unsqueeze(-1)  # [E, 3F]
        gated = phi_j * W_d
        dss, dvv, dvr = torch.split(gated, self.hidden, dim=-1)
        # vector message: dvv * v_j + dvr * r_hat_ij
        rhat = r_ij / d_ij.clamp_min(1e-8).unsqueeze(-1)  # [E, 3]
        v_src = v[src]  # [E, F, 3]
        msg_v = dvv.unsqueeze(-1) * v_src + dvr.unsqueeze(-1) * rhat.unsqueeze(-2)
        N = s.shape[0]
        ds = scatter_add(dss, dst, dim=0, dim_size=N)
        dv = scatter_add(msg_v, dst, dim=0, dim_size=N)
        return s + ds, v + dv


class PaiNNUpdate(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        # Linear on feature channel of v (R^{F,3} -> R^{F,3}), no bias, equivariant
        self.U = nn.Linear(hidden, hidden, bias=False)
        self.V = nn.Linear(hidden, hidden, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3 * hidden)
        )

    def forward(self, s, v):
        # v: [N, F, 3] -> transpose to apply linear on feature dim
        Uv = self.U(v.transpose(-1, -2)).transpose(-1, -2)  # [N, F, 3]
        Vv = self.V(v.transpose(-1, -2)).transpose(-1, -2)
        # Smooth norm: sqrt(||Vv||^2 + eps). Plain vector_norm is non-differentiable
        # at zero, which hits second-order force training at init when v starts at 0.
        Vv_norm = torch.sqrt(Vv.pow(2).sum(dim=-1) + 1e-8)  # [N, F]
        h = torch.cat([s, Vv_norm], dim=-1)
        a = self.mlp(h)
        a_vv, a_sv, a_ss = torch.split(a, self.hidden, dim=-1)
        dot = (Uv * Vv).sum(dim=-1)  # [N, F]
        ds = a_ss + a_sv * dot
        dv = a_vv.unsqueeze(-1) * Uv
        return s + ds, v + dv


class PaiNNBackbone(nn.Module):
    """Equivariant backbone returning per-atom scalar and vector features.

    Expects a PyG-style batch with attributes:
        z: [N] long atomic numbers
        pos: [N, 3] float coordinates
        batch: [N] long graph assignment
    Edges are built on the fly with radius_graph at `cutoff`.
    """

    def __init__(self, cfg: PaiNNConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.max_z, cfg.hidden)
        self.rbf = BesselRBF(cfg.n_radial, cfg.cutoff)
        self.messages = nn.ModuleList(
            [PaiNNMessage(cfg.hidden, cfg.n_radial, cfg.cutoff) for _ in range(cfg.num_layers)]
        )
        self.updates = nn.ModuleList(
            [PaiNNUpdate(cfg.hidden) for _ in range(cfg.num_layers)]
        )

    def forward(self, batch):
        z, pos, bidx = batch.z, batch.pos, batch.batch
        edge_index = radius_graph(pos, r=self.cfg.cutoff, batch=bidx, loop=False)
        src, dst = edge_index[0], edge_index[1]
        r_ij = pos[src] - pos[dst]  # vector from dst to src: used as r_hat into dst
        d_ij = torch.linalg.vector_norm(r_ij, dim=-1)
        rbf = self.rbf(d_ij)

        s = self.embed(z)
        v = torch.zeros(z.shape[0], self.cfg.hidden, 3, device=z.device, dtype=s.dtype)

        for msg, upd in zip(self.messages, self.updates):
            s, v = msg(s, v, edge_index, r_ij, d_ij, rbf)
            s, v = upd(s, v)
        return s, v
