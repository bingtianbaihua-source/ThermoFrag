"""Minimal Metropolis-Hastings kernels for fragment-assembly graphs.

This is the Phase-2 stand-in for the full fragment add/swap/delete proposals in
:mod:`thermofrag.sampling.proposals`. Those require a vocabulary of BRICS
fragments with anchor-point tracking (Phase 3 deliverable). For Phase 2 — where
we only need to evolve PCD buffer chains so that the Coupling potential's
contrastive loss has meaningful negatives — a simpler symmetric proposal is
sufficient:

    Node-label flip: pick a random node k in graph m, propose to replace its
    ``frag_id`` with a uniformly random id in {0, ..., n_fragments-1}.

The proposal is symmetric, so ``log q(m' -> m) - log q(m -> m') = 0`` and the
MH acceptance reduces to ``min(1, exp(-beta (V(m') - V(m))))``. This lets the
buffer explore the discrete graph space with a single coupling-potential
forward pass per step, without touching edges or anchor validity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch_geometric.data import Batch


@dataclass
class FragmentMHStats:
    attempts: int = 0
    accepts: int = 0

    @property
    def accept_rate(self) -> float:
        return self.accepts / max(self.attempts, 1)


class FragmentNodeFlipMH:
    """Per-graph node-label flip proposal with Metropolis acceptance.

    Parameters
    ----------
    coupling : callable scoring V(batch) -> Tensor[B].
    n_fragments : size of the fragment vocabulary.
    beta : inverse temperature used for acceptance.
    """

    def __init__(self, coupling, n_fragments: int, beta: float):
        self.coupling = coupling
        self.n_fragments = int(n_fragments)
        self.beta = float(beta)

    @torch.no_grad()
    def step(self, batch: Batch, stats: FragmentMHStats | None = None) -> Batch:
        """One MH sweep: propose one flip per graph in the batch, accept independently."""
        if stats is None:
            stats = FragmentMHStats()
        device = batch.frag_id.device
        B = int(batch.num_graphs)

        # Score the current batch.
        V_old = self.coupling(batch)  # [B]

        # Propose a flip per graph: pick one atom per graph, change its frag_id.
        ptr = batch.ptr
        flip_positions = torch.empty(B, dtype=torch.long, device=device)
        for k in range(B):
            a, b = int(ptr[k]), int(ptr[k + 1])
            flip_positions[k] = a + torch.randint(0, b - a, (1,), device=device).item()
        new_ids = torch.randint(0, self.n_fragments, (B,), device=device)
        old_ids = batch.frag_id[flip_positions].clone()

        proposed = batch.clone()
        proposed.frag_id[flip_positions] = new_ids
        V_new = self.coupling(proposed)  # [B]

        log_p = -self.beta * (V_new - V_old)  # [B]
        u = torch.rand(B, device=device)
        accept = u.log() < log_p  # [B]

        stats.attempts += B
        stats.accepts += int(accept.sum().item())

        # Apply accepted flips by overwriting the original; rejected stay.
        for k in range(B):
            if not bool(accept[k]):
                continue
            batch.frag_id[flip_positions[k]] = new_ids[k]
        return batch
