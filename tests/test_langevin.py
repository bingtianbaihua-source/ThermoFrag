"""Langevin smoke test.

With a *frozen* random Hamiltonian and high beta, Langevin steps should drive
the mean Hamiltonian H down from its random initial value (relaxation to a
local minimum, up to stochastic noise).
"""
from __future__ import annotations

import torch

from thermofrag.sampling.langevin import LangevinSampler
from synth import build_hamiltonian, rand_atomic, rand_fragment, N_PROPS


def test_langevin_lowers_energy_on_frozen_model():
    torch.manual_seed(0)
    H = build_hamiltonian().eval()
    for p in H.parameters():
        p.requires_grad_(False)

    B = 6
    atomic = rand_atomic(B, seed=7)
    fragment = rand_fragment(B, seed=8)
    y = torch.randn(B, N_PROPS)
    phi = torch.randn(B, N_PROPS)

    sampler = LangevinSampler(H, step_size=0.005, beta=10.0)

    H0 = H(atomic, fragment, y, phi).mean().item()
    state = atomic
    for _ in range(40):
        state = sampler.step(state, fragment, y, phi)
    H1 = H(state, fragment, y, phi).mean().item()

    assert H1 < H0 - 1e-3, (
        f"Langevin did not reduce energy: H0={H0:.4f} -> H1={H1:.4f}"
    )


def test_langevin_preserves_shapes_and_dtype():
    torch.manual_seed(1)
    H = build_hamiltonian().eval()
    B = 3
    atomic = rand_atomic(B, seed=2)
    fragment = rand_fragment(B, seed=3)
    y = torch.randn(B, N_PROPS)
    phi = torch.randn(B, N_PROPS)

    sampler = LangevinSampler(H, step_size=0.003, beta=2.0)
    new = sampler.step(atomic, fragment, y, phi)

    assert new.pos.shape == atomic.pos.shape
    assert new.pos.dtype == atomic.pos.dtype
    assert not new.pos.requires_grad
