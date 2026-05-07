"""QM internal-energy head E^QM_theta(m, x).

Wraps a PaiNN backbone with a per-atom energy MLP; total molecular energy is
the scatter-sum of atomic contributions. Forces are obtained as
-nabla_x E_theta via autograd. See docs/METHOD.md eq (4).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter_add

from thermofrag.model.painn import PaiNNBackbone, PaiNNConfig


class QMHead(nn.Module):
    def __init__(self, cfg: PaiNNConfig, max_z: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.backbone = PaiNNBackbone(cfg)
        self.energy_mlp = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, 1),
        )
        # Per-element learnable offset (centers the trivial atom-counting solution out)
        mz = max_z if max_z is not None else cfg.max_z
        self.atom_ref = nn.Embedding(mz, 1)
        nn.init.zeros_(self.atom_ref.weight)

    def forward(
        self,
        batch,
        return_forces: bool = False,
    ):
        if return_forces:
            batch.pos.requires_grad_(True)
        scalar, _vector = self.backbone(batch)
        atom_e = self.energy_mlp(scalar).squeeze(-1) + self.atom_ref(batch.z).squeeze(-1)
        mol_e = scatter_add(atom_e, batch.batch, dim=0)
        if return_forces:
            grad = torch.autograd.grad(
                mol_e.sum(), batch.pos, create_graph=self.training, retain_graph=True
            )[0]
            forces = -grad
            return mol_e, forces
        return mol_e
