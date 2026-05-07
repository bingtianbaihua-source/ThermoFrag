"""Smoke test for PaiNN + QMHead.

Checks:
  - forward returns scalar energy per molecule with correct shape
  - forces are computed via autograd on positions and have shape [N, 3]
  - energies are rotation-invariant; forces are rotation-equivariant
  - an AdamW loop on a fixed synthetic batch reduces MSE loss by a real margin
"""
from __future__ import annotations

import torch
from torch_geometric.data import Data, Batch

from thermofrag.model.painn import PaiNNConfig
from thermofrag.potentials.qm import QMHead


DRUGLIKE_Z = torch.tensor([1, 6, 7, 8, 9, 15, 16, 17, 35])


def random_molecule(n_atoms: int, seed_tensor: torch.Tensor | None = None) -> Data:
    g = torch.Generator().manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))
    z_idx = torch.randint(0, len(DRUGLIKE_Z), (n_atoms,), generator=g)
    z = DRUGLIKE_Z[z_idx]
    # Spread atoms inside a box large enough that radius_graph sees some edges.
    pos = torch.randn(n_atoms, 3, generator=g) * 1.5
    return Data(z=z, pos=pos)


def make_batch(n_graphs: int, atoms_range: tuple[int, int] = (5, 18)) -> Batch:
    torch.manual_seed(0)
    mols = [random_molecule(int(torch.randint(*atoms_range, (1,)).item())) for _ in range(n_graphs)]
    return Batch.from_data_list(mols)


def synthetic_targets(batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic fake energy = sum of z_i^1.5 + 0.01 * sum ||pos||^2; forces = -d/dx."""
    z = batch.z.float()
    pos = batch.pos
    atomic = z.pow(1.5) + 0.01 * pos.pow(2).sum(-1)
    from torch_scatter import scatter_add

    E = scatter_add(atomic, batch.batch, dim=0)
    # closed-form force under this synthetic: f_i = -d/dx atomic = -0.02 * pos
    F = -0.02 * pos
    return E, F


def build_model(hidden: int = 32, layers: int = 2) -> QMHead:
    cfg = PaiNNConfig(hidden=hidden, num_layers=layers, cutoff=5.0, n_radial=16)
    return QMHead(cfg)


def test_forward_shapes():
    model = build_model()
    batch = make_batch(4)
    E = model(batch)
    assert E.shape == (4,), f"expected [4], got {tuple(E.shape)}"


def test_forces_shape():
    model = build_model()
    batch = make_batch(4)
    E, F = model(batch, return_forces=True)
    assert E.shape == (4,)
    assert F.shape == batch.pos.shape


def test_rotation_equivariance():
    model = build_model().eval()
    batch = make_batch(3)
    theta = 0.7
    R = torch.tensor(
        [[torch.cos(torch.tensor(theta)), -torch.sin(torch.tensor(theta)), 0.0],
         [torch.sin(torch.tensor(theta)),  torch.cos(torch.tensor(theta)), 0.0],
         [0.0, 0.0, 1.0]]
    )

    E1, F1 = model(batch.clone(), return_forces=True)
    rotated = batch.clone()
    rotated.pos = rotated.pos @ R.T
    E2, F2 = model(rotated, return_forces=True)

    assert torch.allclose(E1, E2, atol=1e-4), f"energy not rotation-invariant: max |dE|={((E1-E2).abs().max()):.2e}"
    # forces should rotate accordingly: F2 ≈ F1 @ R.T
    assert torch.allclose(F2, F1 @ R.T, atol=1e-4), (
        f"forces not equivariant: max diff={(F2 - F1 @ R.T).abs().max():.2e}"
    )


def test_training_loop_reduces_loss():
    torch.manual_seed(42)
    model = build_model(hidden=32, layers=2)
    batch = make_batch(8)
    E_true, F_true = synthetic_targets(batch)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    def step():
        opt.zero_grad()
        E_pred, F_pred = model(batch.clone(), return_forces=True)
        loss_e = (E_pred - E_true).pow(2).mean()
        loss_f = (F_pred - F_true).pow(2).mean()
        loss = loss_e + 0.5 * loss_f
        loss.backward()
        opt.step()
        return loss.item()

    loss0 = step()
    for _ in range(80):
        last = step()
    assert last < 0.5 * loss0, f"loss did not halve: {loss0:.3f} -> {last:.3f}"
