"""V^pocket(m, p): scalar ligand-pocket coupling term for TF-pocket-v2.

Adds a new summand to the conditional Hamiltonian::

    H(m; y, p) = V^couple(m) + V^pocket(m, p) - mu(y, p) . phi(m)

so the sampler's MH acceptance directly rewards molecules that the network
predicts will dock well in pocket ``p``. Unlike mu(y, p), which compresses
the 1280-d pocket into an 8-d property-indexed correction, V^pocket
emits a **scalar energy** — the full pocket vector is relevant for that
one number, sidestepping the 8-d bottleneck that made TF-pocket v1 fail.

Training signal: CrossDocked2020 preprocessed LMDB stores ``vina_dock``
per pose. V^pocket is regressed against the standardized Vina dock score
so the predicted energy has roughly unit-variance scale, multiplied by a
saved ``vina_scale`` to convert back to kcal/mol at sampler time.

The module is deliberately small: it consumes the same standardized
phi_z (fragment-sum of per-fragment property vectors) that the sampler
already computes, plus the frozen pocket embedding. The sampler calls
V^pocket once per MH step per proposal, so cost stays negligible.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PocketLigandCoupling(nn.Module):
    """Scalar ligand-pocket affinity predictor.

    Parameters
    ----------
    phi_dim : int
        Length of the standardized property vector phi_z(m) fed at eval time
        (8 for TF-pocket).
    pocket_dim : int
        Dim of the precomputed pocket embedding (1280 for ESM-2 t33).
    pocket_hidden : int
        Hidden width of the pocket projector. 64 keeps params O(100k).
    mlp_hidden : int
        Hidden width of the joint MLP.
    vina_scale : float
        Training-time std of the Vina dock labels. V^pocket predicts a
        standardized score; the module multiplies by vina_scale at forward
        time so downstream energy units are kcal/mol-scale.
    vina_mean : float
        Training-time mean of the Vina dock labels. Subtracted from the
        regression target during training, re-added at forward time.
    """

    def __init__(
        self,
        phi_dim: int,
        pocket_dim: int,
        pocket_hidden: int = 64,
        mlp_hidden: int = 128,
        vina_scale: float = 1.0,
        vina_mean: float = 0.0,
    ):
        super().__init__()
        self.phi_dim = int(phi_dim)
        self.pocket_dim = int(pocket_dim)
        self.pocket_hidden = int(pocket_hidden)
        self.mlp_hidden = int(mlp_hidden)

        self.pocket_proj = nn.Sequential(
            nn.Linear(self.pocket_dim, self.pocket_hidden),
            nn.SiLU(),
            nn.Linear(self.pocket_hidden, self.pocket_hidden),
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.phi_dim + self.pocket_hidden, self.mlp_hidden),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden, self.mlp_hidden),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden, 1),
        )
        # Calibration buffers so the checkpoint is self-describing.
        self.register_buffer("vina_scale", torch.tensor(float(vina_scale)))
        self.register_buffer("vina_mean", torch.tensor(float(vina_mean)))

    def _join(self, phi_z: torch.Tensor, pocket: torch.Tensor) -> torch.Tensor:
        if pocket.dim() == 1:
            pocket = pocket.unsqueeze(0).expand(phi_z.shape[0], -1)
        p = self.pocket_proj(pocket)
        return torch.cat([phi_z, p], dim=-1)

    def forward_standardized(self, phi_z: torch.Tensor, pocket: torch.Tensor) -> torch.Tensor:
        """Raw (standardized) regression head output before de-normalization.

        This is what ``train_pocket_variant.py`` uses during the MSE loop.
        """
        return self.mlp(self._join(phi_z, pocket)).squeeze(-1)

    def forward(self, phi_z: torch.Tensor, pocket: torch.Tensor) -> torch.Tensor:
        """Return V^pocket in kcal/mol-scale units, shape ``[B]``.

        Applies the saved (mean, scale) calibration so the output is on the
        same scale as CrossDocked2020 Vina dock scores (negative = attractive).
        """
        z = self.forward_standardized(phi_z, pocket)
        return z * self.vina_scale + self.vina_mean

    @torch.no_grad()
    def set_calibration(self, vina_mean: float, vina_scale: float) -> None:
        self.vina_mean.fill_(float(vina_mean))
        self.vina_scale.fill_(float(vina_scale))
