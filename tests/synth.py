"""Shared synthetic-data builders for smoke tests.

Kept out of any test_*.py to avoid pytest discovery semantics; imported via
`from tests.synth import ...` or `from synth import ...` depending on sys.path.
A `conftest.py` at this directory ensures the `tests/` dir is on sys.path so
`from synth import ...` works.
"""
from __future__ import annotations

import torch
from torch_geometric.data import Data, Batch

from thermofrag.model.painn import PaiNNConfig
from thermofrag.potentials.qm import QMHead
from thermofrag.potentials.coupling import CouplingPotential
from thermofrag.potentials.external_field import ChemicalPotentialHead
from thermofrag.potentials.hamiltonian import Hamiltonian


DRUGLIKE_Z = torch.tensor([1, 6, 7, 8, 9, 15, 16, 17, 35])
N_FRAG_VOCAB = 64
N_BOND_TYPES = 4
N_PROPS = 8


def rand_atomic(n_graphs: int, seed: int = 0, atoms_range: tuple[int, int] = (5, 15)) -> Batch:
    torch.manual_seed(seed)
    mols = []
    for _ in range(n_graphs):
        n = int(torch.randint(*atoms_range, (1,)))
        z = DRUGLIKE_Z[torch.randint(0, len(DRUGLIKE_Z), (n,))]
        pos = torch.randn(n, 3) * 1.8
        mols.append(Data(z=z, pos=pos))
    return Batch.from_data_list(mols)


def rand_fragment(
    n_graphs: int,
    seed: int = 1,
    nodes_range: tuple[int, int] = (2, 8),
) -> Batch:
    torch.manual_seed(seed)
    mols = []
    for _ in range(n_graphs):
        n = int(torch.randint(*nodes_range, (1,)))
        frag_id = torch.randint(0, N_FRAG_VOCAB, (n,))
        src = torch.arange(n - 1)
        dst = torch.arange(1, n)
        edges = torch.stack([src, dst], dim=0) if n > 1 else torch.zeros(2, 0, dtype=torch.long)
        if n >= 3:
            extra = torch.randint(0, n, (2, min(2, n - 1)))
            edges = torch.cat([edges, extra], dim=1)
        edge_index = torch.cat([edges, edges.flip(0)], dim=1) if edges.numel() else edges
        bond_type = torch.randint(0, N_BOND_TYPES, (edge_index.shape[1],))
        mols.append(
            Data(frag_id=frag_id, edge_index=edge_index, bond_type=bond_type, num_nodes=n)
        )
    return Batch.from_data_list(mols)


def build_hamiltonian(hidden: int = 32, layers: int = 2) -> Hamiltonian:
    qm_cfg = PaiNNConfig(hidden=hidden, num_layers=layers, cutoff=5.0, n_radial=16)
    qm = QMHead(qm_cfg)
    coupling = CouplingPotential(
        n_fragments=N_FRAG_VOCAB,
        n_bond_types=N_BOND_TYPES,
        hidden=hidden,
        num_layers=layers,
    )
    mu = ChemicalPotentialHead(n_properties=N_PROPS, hidden=hidden)
    return Hamiltonian(qm, coupling, mu)
