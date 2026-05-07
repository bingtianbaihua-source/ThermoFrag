"""Reversible fragment-graph proposals for Metropolis-Hastings.

Three proposal kinds, each with its reverse:
  add fragment   <->  delete fragment
  swap fragment  <->  swap fragment

Each proposal must return both the new graph m' and the log-ratio
log q(m | m') - log q(m' | m) needed by the MH acceptance.

See docs/METHOD.md eq (9).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

# These will be RDKit Mol or our own Graph wrapper. Kept abstract for now.
Graph = object  # TODO: define proper Graph dataclass


@dataclass
class Proposal:
    new_graph: Graph
    log_q_forward: float
    log_q_reverse: float


class ProposalKernel:
    def __init__(self, library, add_prob: float = 0.4, swap_prob: float = 0.4, delete_prob: float = 0.2):
        s = add_prob + swap_prob + delete_prob
        assert abs(s - 1.0) < 1e-6, "Proposal probabilities must sum to 1"
        self.library = library
        self.add_prob = add_prob
        self.swap_prob = swap_prob
        self.delete_prob = delete_prob

    def propose(self, m: Graph, rng: random.Random | None = None) -> Proposal:
        rng = rng or random
        u = rng.random()
        if u < self.add_prob:
            return self._propose_add(m, rng)
        if u < self.add_prob + self.swap_prob:
            return self._propose_swap(m, rng)
        return self._propose_delete(m, rng)

    def _propose_add(self, m: Graph, rng: random.Random) -> Proposal:
        # TODO: pick a free attachment point on m, sample a fragment from library, attach.
        raise NotImplementedError

    def _propose_swap(self, m: Graph, rng: random.Random) -> Proposal:
        # TODO: pick an existing fragment in m, replace by another from library at the same anchor.
        raise NotImplementedError

    def _propose_delete(self, m: Graph, rng: random.Random) -> Proposal:
        # TODO: pick a leaf fragment in m, remove it.
        raise NotImplementedError
