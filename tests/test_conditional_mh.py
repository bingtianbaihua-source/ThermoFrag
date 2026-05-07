"""Smoke test for the Phase-4 ConditionalFragmentMH kernel."""
from __future__ import annotations

import torch

from synth import (
    N_FRAG_VOCAB,
    N_BOND_TYPES,
    N_PROPS,
    rand_fragment,
)
from thermofrag.potentials.coupling import CouplingPotential
from thermofrag.potentials.external_field import ChemicalPotentialHead
from thermofrag.sampling.conditional_mh import (
    ConditionalFragmentMH,
    ConditionalMHStats,
    _phi_of_batch,
)


def test_phi_of_batch_shapes_and_sum():
    batch = rand_fragment(4, seed=0, nodes_range=(2, 6))
    frag_phi = torch.randn(N_FRAG_VOCAB, N_PROPS)
    phi_mean = torch.zeros(N_PROPS)
    phi_std = torch.ones(N_PROPS)
    phi_z = _phi_of_batch(batch, frag_phi, phi_mean, phi_std)
    assert phi_z.shape == (4, N_PROPS)
    # Sum over atoms matches manual per-graph sum.
    manual = torch.zeros(4, N_PROPS)
    for b in range(4):
        manual[b] = frag_phi[batch.frag_id[batch.batch == b]].sum(dim=0)
    assert torch.allclose(phi_z, manual, atol=1e-5)


def test_conditional_mh_runs_and_accepts():
    torch.manual_seed(0)
    coupling = CouplingPotential(
        n_fragments=N_FRAG_VOCAB, n_bond_types=N_BOND_TYPES, hidden=16, num_layers=2
    )
    mu = ChemicalPotentialHead(n_properties=N_PROPS, hidden=16)
    frag_phi = torch.randn(N_FRAG_VOCAB, N_PROPS)
    phi_mean = torch.zeros(N_PROPS)
    phi_std = torch.ones(N_PROPS)

    kernel = ConditionalFragmentMH(
        coupling=coupling,
        mu_head=mu,
        frag_phi=frag_phi,
        phi_mean=phi_mean,
        phi_std=phi_std,
        n_fragments=N_FRAG_VOCAB,
        beta=0.5,
    )

    batch = rand_fragment(8, seed=1, nodes_range=(3, 7))
    y = torch.randn(8, N_PROPS)
    stats = ConditionalMHStats(H_mean_history=[])
    for _ in range(20):
        batch = kernel.step(batch, y, stats=stats)
    # We must at least have attempted each chain 20 times.
    assert stats.attempts == 8 * 20
    # Some acceptance should have happened; with random init it's nearly always > 0.
    assert stats.accepts > 0
    # History has one entry per step.
    assert len(stats.H_mean_history) == 20


def test_conditional_mh_prefers_low_H_with_strong_signal():
    """If we set μ(y)·φ so high-φ nodes are strongly attractive, accepted flips
    should bias the marginal fragment distribution toward high-φ ids over time."""
    torch.manual_seed(42)
    hidden = 8
    coupling = CouplingPotential(
        n_fragments=N_FRAG_VOCAB, n_bond_types=N_BOND_TYPES, hidden=hidden, num_layers=2
    )
    # Freeze coupling at 0 (flat V) so μ·φ is the only driver.
    with torch.no_grad():
        for p in coupling.parameters():
            p.zero_()
    mu = ChemicalPotentialHead(n_properties=N_PROPS, hidden=hidden)
    # Force mu(y) = y: zero trunk + identity-ish mean_head.
    with torch.no_grad():
        for p in mu.trunk.parameters():
            p.zero_()
        mu.mean_head.weight.zero_()
        mu.mean_head.bias.copy_(torch.tensor([2.0] + [0.0] * (N_PROPS - 1)))

    # Fragment 0 contributes 0; fragment 7 contributes +5 in dim 0.
    frag_phi = torch.zeros(N_FRAG_VOCAB, N_PROPS)
    frag_phi[7, 0] = 5.0
    phi_mean = torch.zeros(N_PROPS)
    phi_std = torch.ones(N_PROPS)

    kernel = ConditionalFragmentMH(
        coupling=coupling,
        mu_head=mu,
        frag_phi=frag_phi,
        phi_mean=phi_mean,
        phi_std=phi_std,
        n_fragments=N_FRAG_VOCAB,
        beta=2.0,
    )

    batch = rand_fragment(64, seed=2, nodes_range=(4, 6))
    # Start everyone at fragment 0.
    with torch.no_grad():
        batch.frag_id.fill_(0)
    y = torch.zeros(64, N_PROPS)
    # y=0 -> mu(y) = bias = [2, 0, ..., 0]; external = 2*sum_phi0 = 2*5*count7.
    # Higher frag-7 count -> higher external -> lower H -> favored.
    stats = ConditionalMHStats(H_mean_history=[])
    for _ in range(200):
        batch = kernel.step(batch, y, stats=stats)

    count7 = int((batch.frag_id == 7).sum().item())
    total = int(batch.frag_id.numel())
    # Uniform random would give ~total/N_FRAG_VOCAB ~ total/64. We expect a
    # large bias: count7 >> uniform.
    assert count7 > 4 * (total / N_FRAG_VOCAB), f"count7={count7} total={total}"
