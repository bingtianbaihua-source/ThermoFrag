"""Loss functions, see docs/METHOD.md sec 3."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def qm_loss(e_pred: torch.Tensor, e_true: torch.Tensor,
            f_pred: torch.Tensor | None = None, f_true: torch.Tensor | None = None,
            alpha_force: float = 0.5) -> torch.Tensor:
    """L_QM: energy regression with optional force regression. eq (4)."""
    e_loss = F.mse_loss(e_pred, e_true)
    if f_pred is None or f_true is None:
        return e_loss
    f_loss = F.mse_loss(f_pred, f_true)
    return e_loss + alpha_force * f_loss


def coupling_pcd_loss(v_pos: torch.Tensor, v_neg: torch.Tensor) -> torch.Tensor:
    """L_couple: persistent contrastive divergence. eq (5).

    v_pos: V on data batch; v_neg: V on PCD buffer samples.
    """
    return v_pos.mean() - v_neg.mean()


def mu_calibration_loss(mu_pred: torch.Tensor, free_energy_grad: torch.Tensor) -> torch.Tensor:
    """L_mu: enforce mu(y) = grad_y F. eq (7).

    free_energy_grad is a finite-difference estimate produced by
    training.thermo_int.estimate_free_energy_gradient.
    """
    return F.mse_loss(mu_pred, free_energy_grad)


def detailed_balance_loss(log_q_fwd, log_q_rev, H_old, H_new) -> torch.Tensor:
    """L_DB: regularize proposal kernel toward detailed balance. eq (8)."""
    residual = log_q_fwd + H_old - log_q_rev - H_new
    return (residual ** 2).mean()
