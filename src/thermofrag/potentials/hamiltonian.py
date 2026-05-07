"""Total Hamiltonian H_theta(m, x; y) = E^QM + V^couple - mu(y) . phi(m, x).

This is the central object used by both training (loss assembly) and sampling
(MH acceptance and Langevin gradient).

See docs/METHOD.md eq (2).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from thermofrag.potentials.qm import QMHead
from thermofrag.potentials.coupling import CouplingPotential
from thermofrag.potentials.external_field import ChemicalPotentialHead


class Hamiltonian(nn.Module):
    def __init__(self, qm: QMHead, coupling: CouplingPotential, mu: ChemicalPotentialHead):
        super().__init__()
        self.qm = qm
        self.coupling = coupling
        self.mu = mu

    def forward(
        self,
        batch_atomic,
        batch_fragment,
        y: torch.Tensor,
        phi: torch.Tensor,
        return_components: bool = False,
    ):
        """Return H per molecule, shape [B], plus optional component breakdown."""
        e_qm = self.qm(batch_atomic)
        v_couple = self.coupling(batch_fragment)
        mu_y = self.mu(y)
        external = (mu_y * phi).sum(-1)
        H = e_qm + v_couple - external
        if return_components:
            return H, {"qm": e_qm, "couple": v_couple, "external": external}
        return H
