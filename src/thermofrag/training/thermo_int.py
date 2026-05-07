"""Single-step thermodynamic-integration estimate of grad_y F.

We use the finite-difference identity
    grad_y log Z(y) = - beta * E_p(.|y) [ d H / d y ]
and compute d H / d y = - phi(m, x) (since H linear in mu(y) and y enters only
through mu). Thus grad_y F(y) = beta^{-1} * E_p(.|y)[phi].

This gives an O(1) per-step estimator using PCD samples.
"""
from __future__ import annotations

import torch


def free_energy_gradient(phi_buffer_samples: torch.Tensor, beta: float) -> torch.Tensor:
    """Estimate grad_y F at the y the PCD buffer was conditioned on.

    phi_buffer_samples: [N, n_properties] property features of buffer samples.
    Returns: [n_properties] mean property feature, scaled by 1/beta.
    """
    return phi_buffer_samples.mean(dim=0) / beta
