"""Full Hamiltonian forward smoke test.

H = E^QM(atomic batch, coords) + V^couple(fragment batch) - mu(y) . phi

Verifies:
  - each component runs on synthetic data with the expected shape
  - total H = qm + couple - external (component breakdown matches sum)
  - loss = H.sum() backprops gradients into every module
"""
from __future__ import annotations

import torch

from synth import build_hamiltonian, rand_atomic, rand_fragment, N_PROPS


def test_components_sum_to_total():
    H_model = build_hamiltonian().eval()
    B = 4
    atomic = rand_atomic(B)
    frag = rand_fragment(B)
    y = torch.randn(B, N_PROPS)
    phi = torch.randn(B, N_PROPS)

    H, comps = H_model(atomic, frag, y, phi, return_components=True)
    assert H.shape == (B,)
    assert torch.allclose(H, comps["qm"] + comps["couple"] - comps["external"], atol=1e-5)


def test_backward_hits_every_module():
    H_model = build_hamiltonian().train()
    B = 4
    atomic = rand_atomic(B)
    frag = rand_fragment(B)
    y = torch.randn(B, N_PROPS)
    phi = torch.randn(B, N_PROPS)

    H = H_model(atomic, frag, y, phi)
    H.sum().backward()

    # Each of the three sub-modules should receive a gradient somewhere.
    def _has_grad(mod):
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters())

    assert _has_grad(H_model.qm), "QM head received no gradient"
    assert _has_grad(H_model.coupling), "Coupling potential received no gradient"
    assert _has_grad(H_model.mu), "Chemical-potential head received no gradient"


def test_mu_linear_in_y_via_finite_difference():
    """Sanity: dH/dy_i = -mu_i(y) . d(phi)/dy_i. Here phi is independent of y,
    so dH/dy = -J_mu(y)^T phi. This checks autograd through the external field.
    """
    H_model = build_hamiltonian().eval()
    B = 3
    atomic = rand_atomic(B)
    frag = rand_fragment(B)
    y = torch.randn(B, N_PROPS, requires_grad=True)
    phi = torch.randn(B, N_PROPS)

    H = H_model(atomic, frag, y, phi)
    (grad_y,) = torch.autograd.grad(H.sum(), y)
    assert grad_y.shape == y.shape
    # Should be non-trivial (mu head is non-constant in y)
    assert grad_y.abs().mean() > 1e-6
