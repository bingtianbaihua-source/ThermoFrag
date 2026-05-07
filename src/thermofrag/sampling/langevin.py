"""Langevin coordinate updates.

See docs/METHOD.md eq (10).

    x_{t+1} = x_t - eta * grad_x H + sqrt(2 eta / beta) * xi,   xi ~ N(0, I)

The sampler holds the Hamiltonian and hyperparameters only; the state is
passed in per step as a PyG atomic batch together with the frozen
fragment-graph batch and conditioning (y, phi).
"""
from __future__ import annotations

import math

import torch


class LangevinSampler:
    def __init__(self, hamiltonian, step_size: float, beta: float):
        self.hamiltonian = hamiltonian
        self.step_size = step_size
        self.beta = beta

    def step(self, atomic_batch, fragment_batch, y: torch.Tensor, phi: torch.Tensor):
        """One Langevin coordinate update. Returns a cloned atomic_batch with
        updated positions. Model parameters are untouched; only `pos` sees grads
        here and they are consumed by autograd on the fly.
        """
        batch = atomic_batch.clone()
        batch.pos = batch.pos.detach().clone().requires_grad_(True)
        H = self.hamiltonian(batch, fragment_batch, y, phi)
        (grad_x,) = torch.autograd.grad(H.sum(), batch.pos)
        noise = torch.randn_like(batch.pos) * math.sqrt(2 * self.step_size / self.beta)
        new_pos = (batch.pos - self.step_size * grad_x + noise).detach()
        batch.pos = new_pos
        return batch
