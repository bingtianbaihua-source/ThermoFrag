"""Annealed sampling with parallel tempering.

See docs/METHOD.md sec 4.3.

For v0 we support Langevin-only coordinate chains. The discrete fragment-graph
MH moves (proposals.py) are stubbed and come online in Phase 2 once the
fragment library is wired; they plug into :class:`ParallelTempering.run` in
exactly the same position as the Langevin step.

Parallel-tempering swap acceptance between adjacent chains ``i`` and ``j``:

    P_swap = min( 1, exp( (beta_i - beta_j) * (H_i - H_j) ) ).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch

from thermofrag.sampling.langevin import LangevinSampler


@dataclass
class LinearBetaSchedule:
    beta0: float
    beta_T: float
    steps: int

    def __call__(self, t: int) -> float:
        if self.steps <= 1:
            return self.beta_T
        frac = t / max(self.steps - 1, 1)
        return self.beta0 + (self.beta_T - self.beta0) * frac


@dataclass
class PTStats:
    swap_attempts: int = 0
    swap_accepts: int = 0
    H_per_chain: list[list[float]] = field(default_factory=list)

    @property
    def swap_rate(self) -> float:
        return self.swap_accepts / max(self.swap_attempts, 1)


class ParallelTempering:
    """Langevin-only parallel tempering.

    Parameters
    ----------
    hamiltonian : callable producing scalar H per molecule
    betas       : inverse temperatures, in ascending order (coldest last).
    step_size   : Langevin step, shared across chains (can be a list to vary).
    swap_every  : attempt one swap (adjacent pair sweep) every this many steps.
    """

    def __init__(
        self,
        hamiltonian,
        betas: Sequence[float],
        step_size: float | Sequence[float] = 0.005,
        swap_every: int = 20,
    ):
        self.hamiltonian = hamiltonian
        self.betas = list(betas)
        if not all(self.betas[i] <= self.betas[i + 1] for i in range(len(self.betas) - 1)):
            raise ValueError("betas must be ascending (coldest last)")
        if isinstance(step_size, (int, float)):
            step_size = [float(step_size)] * len(self.betas)
        self.samplers = [
            LangevinSampler(hamiltonian, step_size=s, beta=b)
            for s, b in zip(step_size, self.betas)
        ]
        self.swap_every = int(swap_every)

    def _compute_H(self, atomic_batch, fragment_batch, y, phi) -> torch.Tensor:
        """H([B]) per molecule. The Hamiltonian forward may need autograd on
        pos elsewhere; here we only need values, so detach the result."""
        return self.hamiltonian(atomic_batch, fragment_batch, y, phi).detach()

    def _try_swap_pair(
        self,
        i: int,
        j: int,
        chains,
        fragment_batch,
        y,
        phi,
        stats: PTStats,
        rng: torch.Generator,
    ) -> None:
        """Per-sample PT swap between chains i and j.

        Each of the B configurations is swapped independently with probability
        min(1, exp((β_i - β_j)(H_i - H_j))). Accepted swaps exchange the
        corresponding atom slices (via ``batch.ptr``) between the two chains.
        """
        H_i = self._compute_H(chains[i], fragment_batch, y, phi)  # [B]
        H_j = self._compute_H(chains[j], fragment_batch, y, phi)  # [B]
        log_p = (self.betas[i] - self.betas[j]) * (H_i - H_j)  # [B]
        u = torch.rand(H_i.shape[0], generator=rng)
        accept_mask = u.log() < log_p  # [B], bool

        B = H_i.shape[0]
        stats.swap_attempts += B
        n_accepted = int(accept_mask.sum().item())
        if n_accepted == 0:
            return
        stats.swap_accepts += n_accepted

        # Swap per-sample atom slices. Both chains share the same graph
        # connectivity (only `pos` differs across chains), so we can rely on
        # the same ptr for both.
        ptr_i = chains[i].ptr if hasattr(chains[i], "ptr") else None
        ptr_j = chains[j].ptr if hasattr(chains[j], "ptr") else None
        if ptr_i is None or ptr_j is None:
            raise RuntimeError("ParallelTempering requires batched Data with .ptr; use Batch.from_data_list")
        for k in range(B):
            if not accept_mask[k].item():
                continue
            a, b = int(ptr_i[k]), int(ptr_i[k + 1])
            a2, b2 = int(ptr_j[k]), int(ptr_j[k + 1])
            # In our setup atom counts per sample match across chains (same graphs)
            assert (b - a) == (b2 - a2), "chain graphs must share atom counts per sample"
            pos_i_slice = chains[i].pos[a:b].clone()
            pos_j_slice = chains[j].pos[a2:b2].clone()
            chains[i].pos[a:b] = pos_j_slice
            chains[j].pos[a2:b2] = pos_i_slice

    def run(
        self,
        atomic_init,
        fragment_batch,
        y: torch.Tensor,
        phi: torch.Tensor,
        n_steps: int,
        record_every: int = 0,
        seed: int | None = None,
    ) -> tuple[object, PTStats]:
        """Run n_steps of Langevin on every chain, attempting swaps every
        ``swap_every`` steps. Returns the coldest chain's final state and stats.
        """
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(int(seed))

        # Clone the initial atomic batch per chain so they evolve independently.
        chains = [atomic_init.clone() for _ in range(len(self.betas))]
        stats = PTStats(H_per_chain=[[] for _ in self.betas])

        for t in range(n_steps):
            for i, sampler in enumerate(self.samplers):
                chains[i] = sampler.step(chains[i], fragment_batch, y, phi)
                if record_every and (t % record_every == 0):
                    H_i = self._compute_H(chains[i], fragment_batch, y, phi).mean().item()
                    stats.H_per_chain[i].append(H_i)

            if self.swap_every > 0 and (t + 1) % self.swap_every == 0 and len(self.betas) > 1:
                # Sweep adjacent pairs in random order (even/odd alternation keeps each move reversible).
                for i in range(len(self.betas) - 1):
                    self._try_swap_pair(i, i + 1, chains, fragment_batch, y, phi, stats, rng)

        return chains[-1], stats  # coldest
