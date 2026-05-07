"""Parallel-tempering smoke test.

On a frozen random Hamiltonian, the cold chain should end up with a lower
mean H than the hottest chain, and the swap acceptance should be within a
reasonable band (not stuck at 0 or 1).

For a stronger, analytic check we also run PT on an explicit quadratic-bowl
Hamiltonian H(x) = sum ||pos||^2, where equilibrium variance per temperature
is exactly 3 / (2 beta) so the cold chain converges much tighter than the hot.
"""
from __future__ import annotations

import torch

from thermofrag.sampling.anneal import LinearBetaSchedule, ParallelTempering
from synth import build_hamiltonian, rand_atomic, rand_fragment, N_PROPS


class QuadraticBowlHamiltonian:
    """H(x) = sum_atom_in_graph ||pos||^2, returned per-molecule as Tensor [B].

    Matches the Hamiltonian protocol expected by ParallelTempering /
    LangevinSampler (same signature as ``thermofrag.potentials.hamiltonian.Hamiltonian``).
    """

    def __call__(self, atomic_batch, fragment_batch, y, phi):
        from torch_scatter import scatter_add

        sq = atomic_batch.pos.pow(2).sum(-1)  # [N]
        return scatter_add(sq, atomic_batch.batch, dim=0)  # [B]

    def parameters(self):
        return []


def test_beta_schedule_linear():
    s = LinearBetaSchedule(beta0=0.5, beta_T=5.0, steps=11)
    assert abs(s(0) - 0.5) < 1e-6
    assert abs(s(10) - 5.0) < 1e-6
    assert abs(s(5) - 2.75) < 1e-6


def test_cold_chain_reaches_lower_energy_than_hot():
    torch.manual_seed(7)
    H = build_hamiltonian().eval()
    for p in H.parameters():
        p.requires_grad_(False)

    B = 6
    atomic = rand_atomic(B, seed=11)
    fragment = rand_fragment(B, seed=12)
    y = torch.randn(B, N_PROPS)
    phi = torch.randn(B, N_PROPS)

    betas = [0.3, 1.0, 3.0, 10.0]
    pt = ParallelTempering(H, betas=betas, step_size=0.005, swap_every=20)

    # Record mean H at every swap epoch so we can compare the endpoints.
    cold_state, stats = pt.run(atomic, fragment, y, phi, n_steps=200, record_every=20, seed=0)

    # Hot vs cold ending energies.
    hot_H = stats.H_per_chain[0]
    cold_H = stats.H_per_chain[-1]
    assert len(cold_H) > 0

    hot_final = sum(hot_H[-3:]) / 3
    cold_final = sum(cold_H[-3:]) / 3
    assert cold_final < hot_final - 1e-3, (
        f"cold chain did not reach lower H than hot: hot={hot_final:.4f} cold={cold_final:.4f}"
    )
    # Swap acceptance should be neither zero nor one.
    assert 0.0 < stats.swap_rate < 1.0, f"swap_rate={stats.swap_rate:.3f} looks broken"


def test_pt_separates_temperatures_on_quadratic_bowl():
    """On H(x) = ||pos||^2, equilibrium ⟨H⟩ per-atom ≈ 3/(2β). Four chains at
    β in {0.5, 1, 2, 5} should therefore rank-order cleanly after enough steps.
    """
    torch.manual_seed(0)
    H = QuadraticBowlHamiltonian()

    atomic = rand_atomic(8, seed=1)
    fragment = rand_fragment(8, seed=2)  # unused by the bowl, kept for API
    y = torch.randn(8, N_PROPS)
    phi = torch.randn(8, N_PROPS)

    betas = [0.5, 1.0, 2.0, 5.0]
    pt = ParallelTempering(H, betas=betas, step_size=0.03, swap_every=20)
    _, stats = pt.run(atomic, fragment, y, phi, n_steps=400, record_every=20, seed=0)

    # Average over the second half of the trajectory (after burn-in).
    tails = [sum(t[len(t) // 2:]) / max(len(t) - len(t) // 2, 1) for t in stats.H_per_chain]
    # Cold chain (largest β) should have lowest ⟨H⟩; hot chain highest.
    assert tails[-1] < tails[0] - 0.5, f"cold not colder: {tails}"
    # Rough monotone: tails should be non-increasing (allow small noise tolerance)
    for a, b in zip(tails, tails[1:]):
        assert b < a + 0.5, f"temperatures out of order: {tails}"
    # Swap rate finite and not trivially 0 or 1
    assert 0.05 < stats.swap_rate < 0.95, f"swap_rate={stats.swap_rate:.3f}"


def test_cold_state_shape_preserved():
    torch.manual_seed(3)
    H = build_hamiltonian().eval()
    for p in H.parameters():
        p.requires_grad_(False)

    atomic = rand_atomic(4, seed=1)
    fragment = rand_fragment(4, seed=2)
    y = torch.randn(4, N_PROPS)
    phi = torch.randn(4, N_PROPS)

    pt = ParallelTempering(H, betas=[1.0, 3.0], step_size=0.004, swap_every=10)
    final, stats = pt.run(atomic, fragment, y, phi, n_steps=30, seed=0)
    assert final.pos.shape == atomic.pos.shape
    assert torch.isfinite(final.pos).all()
