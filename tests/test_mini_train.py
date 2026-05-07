"""End-to-end mini training smoke test.

Trains the full Hamiltonian on synthetic targets with the three main loss
components assembled together (QM regression, coupling contrastive, μ
calibration). Success criterion: each component and the total loss drop
substantially across 80 AdamW steps, and gradients are healthy (no NaNs).

This does not verify physical correctness — the targets are synthetic. It
verifies that the loss assembly, optimizer, and backward graph all compose.
"""
from __future__ import annotations

import torch

from thermofrag.training.losses import (
    qm_loss,
    coupling_pcd_loss,
    mu_calibration_loss,
)
from thermofrag.training.thermo_int import free_energy_gradient
from synth import build_hamiltonian, rand_atomic, rand_fragment, N_PROPS


def _synthetic_qm_targets(atomic_batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic per-molecule energy, standardized to O(1) magnitude so losses
    don't explode before the atom_ref offset has a chance to absorb the mean.
    Forces are small so regression is tractable without elaborate scaling.
    """
    from torch_scatter import scatter_add

    z = atomic_batch.z.float()
    pos = atomic_batch.pos
    atomic_e = 0.1 * z + 0.01 * pos.pow(2).sum(-1)
    E = scatter_add(atomic_e, atomic_batch.batch, dim=0)
    E = (E - E.mean()) / (E.std() + 1e-6)  # standardize to N(0, 1) across batch
    F = -0.02 * pos
    return E.detach(), F.detach()


def test_mini_joint_training_reduces_all_losses():
    torch.manual_seed(0)
    H = build_hamiltonian(hidden=32, layers=2)
    B = 8
    beta = 1.0

    # QM data: fixed batch with deterministic synthetic targets.
    atomic_data = rand_atomic(B, seed=10)
    E_true, F_true = _synthetic_qm_targets(atomic_data)

    # Coupling contrast: "data" graphs small (2-4 frags), "buffer" graphs larger (5-8).
    frag_data = rand_fragment(B, seed=20, nodes_range=(2, 5))
    frag_buffer = rand_fragment(B, seed=21, nodes_range=(5, 9))

    # μ calibration: fixed y, fixed phi samples from the buffer (synthetic).
    y = torch.randn(B, N_PROPS)
    phi_buffer = torch.randn(64, N_PROPS)  # pretend 64 buffer samples

    opt = torch.optim.AdamW(H.parameters(), lr=1e-3)

    def compute_losses():
        # QM part (forward through QMHead only to get energy+forces)
        atomic_fwd = atomic_data.clone()
        atomic_fwd.pos = atomic_fwd.pos.detach().clone().requires_grad_(True)
        scalar, _ = H.qm.backbone(atomic_fwd)
        atom_e = H.qm.energy_mlp(scalar).squeeze(-1) + H.qm.atom_ref(atomic_fwd.z).squeeze(-1)
        from torch_scatter import scatter_add
        E_pred = scatter_add(atom_e, atomic_fwd.batch, dim=0)
        (grad_pos,) = torch.autograd.grad(E_pred.sum(), atomic_fwd.pos, create_graph=True)
        F_pred = -grad_pos
        L_qm = qm_loss(E_pred, E_true, F_pred, F_true, alpha_force=0.5)

        # Coupling PCD
        v_data = H.coupling(frag_data)
        v_buf = H.coupling(frag_buffer)
        L_couple = coupling_pcd_loss(v_data, v_buf)

        # μ calibration: target grad_y F from buffer samples; single finite-difference step
        fe_grad = free_energy_gradient(phi_buffer, beta=beta)  # [n_props]
        mu_pred = H.mu(y)  # [B, n_props]
        L_mu = mu_calibration_loss(mu_pred, fe_grad.unsqueeze(0).expand_as(mu_pred))

        return L_qm, L_couple, L_mu

    # Record initial losses
    L_qm0, L_c0, L_mu0 = compute_losses()
    total0 = L_qm0 + 0.1 * L_c0 + 0.1 * L_mu0

    for _ in range(80):
        opt.zero_grad()
        L_qm, L_c, L_mu = compute_losses()
        # Small L2 on V to keep the unbounded PCD loss from diverging.
        v_all = torch.cat([H.coupling(frag_data), H.coupling(frag_buffer)])
        L_reg = 1e-3 * v_all.pow(2).mean()
        total = L_qm + 0.1 * L_c + 0.1 * L_mu + L_reg
        total.backward()
        for p in H.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), "NaN/Inf in gradients"
        torch.nn.utils.clip_grad_norm_(H.parameters(), max_norm=5.0)
        opt.step()

    L_qm1, L_c1, L_mu1 = compute_losses()
    total1 = L_qm1 + 0.1 * L_c1 + 0.1 * L_mu1

    # Total must drop significantly
    assert total1 < 0.5 * total0, f"total: {total0.item():.3f} -> {total1.item():.3f}"
    # Each component also improves
    assert L_qm1 < 0.5 * L_qm0, f"L_qm: {L_qm0.item():.3f} -> {L_qm1.item():.3f}"
    # Coupling contrast: v_data - v_buf should become more negative (data gets lower V).
    # PCD loss unbounded below without regularization, check it moves in the right direction.
    assert L_c1 < L_c0 - 0.1, f"L_couple: {L_c0.item():.3f} -> {L_c1.item():.3f}"
    assert L_mu1 < 0.5 * L_mu0, f"L_mu: {L_mu0.item():.3f} -> {L_mu1.item():.3f}"
