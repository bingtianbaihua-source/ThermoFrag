"""Metropolis-Hastings acceptance using the Hamiltonian.

See docs/METHOD.md eq (9).
"""
from __future__ import annotations

import math

import torch

from thermofrag.sampling.proposals import Proposal, ProposalKernel


def mh_accept(delta_H: float, log_q_ratio: float, beta: float) -> bool:
    """Accept proposal with prob min(1, exp(-beta * dH) * q_reverse / q_forward)."""
    log_alpha = -beta * delta_H + log_q_ratio
    if log_alpha >= 0:
        return True
    return math.log(torch.rand(1).item()) < log_alpha


class MetropolisSampler:
    def __init__(self, hamiltonian, kernel: ProposalKernel, beta: float):
        self.hamiltonian = hamiltonian
        self.kernel = kernel
        self.beta = beta

    def step(self, state):
        """One MH step on the discrete graph component of the state."""
        # TODO: build proposal, score H(m'), call mh_accept.
        raise NotImplementedError
