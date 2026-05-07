"""Conditional Metropolis-Hastings on fragment-assembly graphs.

Extends :class:`FragmentNodeFlipMH` from Phase 2 to use the full Hamiltonian

    H_theta(m; y) = V^couple(m) - mu(y) . phi(m),

implementing the Phase-4 sampler used by ``scripts/sample.py`` and the
LIT-PCBA evaluation pipeline (docs/MILESTONES.md Phase 4). Proposals are the
same symmetric node-label flips used in Phase 2 so the MH acceptance reduces
to

    p_accept = min( 1, exp( -beta * (H(m') - H(m)) ) ).

Because the μ head was trained on a seed-level (parent SMILES) phi (never
re-evaluated inside the PCD chain), we mirror that convention at sample time:
phi(m) is the sum of per-fragment property vectors, standardized by the same
phi_mean/phi_std used during training. See ``build_frag_phi_table``.

Notes
-----
* ``phi(m)`` is NOT recomputed via RDKit at every step -- that would require a
  SMILES reassembly on every proposal. Instead we use the additive proxy
  phi(m) = sum_i frag_phi[frag_id_i]. This matches the phi(seed) the μ head
  calibrates against for extensive properties (logP, MW, TPSA, #HBA, #HBD,
  #rotb). QED and SA are intensive -- the sum is a noisier proxy for those,
  but training uses the seed-level SMILES value anyway, so the intensive
  components contribute mostly a constant offset.
* ``mu(y)`` is evaluated once at the start of ``run`` and cached; it only
  depends on y, not on the chain state.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from thermofrag.data.properties import compute_phi
from thermofrag.sampling.fragment_mh import FragmentMHStats


def build_frag_phi_table(
    library_path: str | Path,
    properties: list[str],
) -> np.ndarray:
    """Compute per-fragment property vectors phi_core for every vocabulary entry.

    Parameters
    ----------
    library_path : fragment_library.parquet path (frag_id, fragment_smi, ...).
    properties   : list of property names in compute_phi's REGISTRY.

    Returns
    -------
    ndarray of shape ``[n_fragments, n_properties]``. Row 0 (UNK) is all zeros.
    Rows whose SMILES fails to parse are also zero-filled (rare; <0.1% in
    practice).
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    lib = (
        pd.read_parquet(library_path)
        .sort_values("frag_id")
        .reset_index(drop=True)
    )
    V = int(lib["frag_id"].max()) + 1
    K = len(properties)
    table = np.zeros((V, K), dtype=np.float32)
    n_bad = 0
    for row in lib.itertuples():
        if row.fragment_smi == "__UNK__":
            continue
        mol = Chem.MolFromSmiles(row.fragment_smi)
        if mol is None:
            n_bad += 1
            continue
        try:
            table[int(row.frag_id)] = compute_phi(mol, properties)
        except Exception:
            n_bad += 1
    if n_bad:
        print(f"[frag_phi] {n_bad}/{V} fragments failed phi computation; zero-filled")
    return table


def _phi_of_batch(
    batch: Batch,
    frag_phi: torch.Tensor,  # [V, K]
    phi_mean: torch.Tensor,  # [K]
    phi_std: torch.Tensor,   # [K]
) -> torch.Tensor:
    """Standardized per-graph phi, shape [B, K].

    phi_raw(m) = sum_i frag_phi[frag_id_i]
    phi_std(m) = (phi_raw - phi_mean) / phi_std
    """
    node_phi = frag_phi[batch.frag_id]  # [N, K]
    # Scatter-sum over the graph index.
    B = int(batch.num_graphs)
    out = torch.zeros(B, frag_phi.shape[1], device=node_phi.device, dtype=node_phi.dtype)
    out.index_add_(0, batch.batch, node_phi)
    return (out - phi_mean) / phi_std


@dataclass
class ConditionalMHStats(FragmentMHStats):
    H_mean_history: list[float] | None = None


class ConditionalFragmentMH:
    """Node-flip MH on fragment graphs with conditional Hamiltonian.

    Parameters
    ----------
    coupling    : callable Batch -> [B] returning V(m).
    mu_head     : callable y -> [B, K] returning mu(y).
    frag_phi    : [V, K] tensor of per-fragment property sums.
    phi_mean    : [K] tensor (training phi standardization mean).
    phi_std     : [K] tensor.
    n_fragments : vocabulary size (proposal distribution).
    beta        : inverse temperature used for acceptance.
    v_pocket_fn : optional callable phi_z -> [B] returning V^pocket(m, p).
        When set, the Hamiltonian gains a pocket-ligand coupling term
        on top of the base V^couple + external-field sum (TF-pocket-v2).
        The callable is expected to close over the target pocket
        embedding; it takes only the standardized phi of the current
        batch so the MH driver stays pocket-agnostic.
    """

    def __init__(
        self,
        coupling: Callable[[Batch], torch.Tensor],
        mu_head: Callable[[torch.Tensor], torch.Tensor],
        frag_phi: torch.Tensor,
        phi_mean: torch.Tensor,
        phi_std: torch.Tensor,
        n_fragments: int,
        beta: float = 1.0,
        v_pocket_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.coupling = coupling
        self.mu_head = mu_head
        self.frag_phi = frag_phi
        self.phi_mean = phi_mean
        self.phi_std = phi_std
        self.n_fragments = int(n_fragments)
        self.beta = float(beta)
        self.v_pocket_fn = v_pocket_fn

    def _hamiltonian(self, batch: Batch, mu_y: torch.Tensor) -> torch.Tensor:
        v = self.coupling(batch)  # [B]
        phi_z = _phi_of_batch(batch, self.frag_phi, self.phi_mean, self.phi_std)  # [B, K]
        external = (mu_y * phi_z).sum(dim=-1)  # [B]
        h = v - external  # [B]
        if self.v_pocket_fn is not None:
            h = h + self.v_pocket_fn(phi_z)
        return h

    @torch.no_grad()
    def step(
        self,
        batch: Batch,
        y: torch.Tensor,  # [B, K]  per-graph conditioning target
        stats: ConditionalMHStats | None = None,
    ) -> Batch:
        """One MH sweep: propose one flip per graph, accept independently."""
        if stats is None:
            stats = ConditionalMHStats()
        device = batch.frag_id.device
        B = int(batch.num_graphs)

        mu_y = self.mu_head(y)  # [B, K]

        H_old = self._hamiltonian(batch, mu_y)  # [B]

        # Propose a flip per graph: pick one node per graph, reassign its frag_id.
        ptr = batch.ptr
        flip_positions = torch.empty(B, dtype=torch.long, device=device)
        for k in range(B):
            a, b = int(ptr[k]), int(ptr[k + 1])
            flip_positions[k] = a + torch.randint(0, b - a, (1,), device=device).item()
        new_ids = torch.randint(0, self.n_fragments, (B,), device=device)
        old_ids = batch.frag_id[flip_positions].clone()

        proposed = batch.clone()
        proposed.frag_id[flip_positions] = new_ids
        H_new = self._hamiltonian(proposed, mu_y)  # [B]

        log_p = -self.beta * (H_new - H_old)  # [B]
        u = torch.rand(B, device=device)
        accept = u.log() < log_p  # [B]

        stats.attempts += B
        stats.accepts += int(accept.sum().item())

        if stats.H_mean_history is not None:
            # Record post-acceptance mean H for diagnostics.
            H_applied = torch.where(accept, H_new, H_old)
            stats.H_mean_history.append(float(H_applied.mean().item()))

        # Apply accepted flips in-place on `batch`.
        for k in range(B):
            if not bool(accept[k]):
                continue
            batch.frag_id[flip_positions[k]] = new_ids[k]
        return batch

    @torch.no_grad()
    def run(
        self,
        batch: Batch,
        y: torch.Tensor,
        n_steps: int,
        stats: ConditionalMHStats | None = None,
    ) -> tuple[Batch, ConditionalMHStats]:
        if stats is None:
            stats = ConditionalMHStats(H_mean_history=[])
        for _ in range(int(n_steps)):
            batch = self.step(batch, y, stats=stats)
        return batch, stats
