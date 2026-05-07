"""Phase-3 joint-Hamiltonian trainer test.

Synthesizes a conditional dataset with simultaneous:
  * atomic batch with coords, self-consistent quadratic E/F targets
  * fragment batch (small graphs, low-id frag vocabulary)
  * property vector y and property features phi

Verifies that over 80 joint steps:
  * L_qm drops substantially
  * L_couple drops (negative samples evolve under MH)
  * L_mu drops (μ head calibrates toward finite-difference target)
  * Hamiltonian checkpoint round-trips through torch.load
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.data import Batch, Data

from thermofrag.sampling.fragment_mh import FragmentNodeFlipMH
from thermofrag.training.pcd import PCDBuffer
from thermofrag.training.trainer import Trainer, build_hamiltonian
from thermofrag.utils.config import load_config
from thermofrag.utils.seed import seed_everything

from synth import rand_atomic, rand_fragment, N_FRAG_VOCAB, N_BOND_TYPES, N_PROPS


def _quadratic_targets(atomic_batch):
    """Self-consistent SE(3)-invariant targets:
      E = 0.01 * sum_{i<j} ||x_i - x_j||^2 per molecule (after scatter by batch.batch)
      F_i = -dE/dx_i = -0.02 * (n * x_i - sum_k x_k) per graph
    """
    from torch_scatter import scatter_add

    pos = atomic_batch.pos
    b = atomic_batch.batch
    B = int(b.max().item()) + 1
    # per-atom position -> per-graph sum
    sums_per_graph = scatter_add(pos, b, dim=0, dim_size=B)  # [B, 3]
    n_per_graph = scatter_add(torch.ones_like(pos[:, :1]).squeeze(-1), b, dim=0, dim_size=B)  # [B]
    sum_per_atom = sums_per_graph[b]  # [N, 3]
    n_per_atom = n_per_graph[b]  # [N]
    # Per-pair E computed per atom as 0.5 * sum_j ||x_i-x_j||^2 (pair double-counted, then halved per mol)
    # = 0.5 * (n*||x_i||^2 + sum_j ||x_j||^2 - 2 x_i . sum_j x_j)
    sq_x = (pos * pos).sum(-1)  # [N]
    sum_sq_per_graph = scatter_add(sq_x, b, dim=0, dim_size=B)  # [B]
    sum_sq_per_atom = sum_sq_per_graph[b]  # [N]
    per_atom_half = 0.5 * (n_per_atom * sq_x + sum_sq_per_atom - 2 * (pos * sum_per_atom).sum(-1))
    E_per_mol = 0.01 * scatter_add(per_atom_half, b, dim=0, dim_size=B)  # [B]
    # F_i
    F = -0.02 * (n_per_atom.unsqueeze(-1) * pos - sum_per_atom)
    return E_per_mol.detach(), F.detach()


class _CondIter:
    """In-memory dataset of fixed joint batches."""

    def __init__(self, n_batches: int, batch_size: int, seed: int = 0):
        self.batches = []
        torch.manual_seed(seed)
        for bi in range(n_batches):
            atomic = rand_atomic(batch_size, seed=seed + bi)
            fragment = rand_fragment(batch_size, seed=seed + 1000 + bi, nodes_range=(2, 5))
            y = torch.randn(batch_size, N_PROPS)
            phi = torch.randn(batch_size, N_PROPS) * 0.5 + 0.1
            E, F = _quadratic_targets(atomic)
            E_norm = (E - E.mean()) / (E.std() + 1e-6)
            self.batches.append(
                SimpleNamespace(atomic=atomic, fragment=fragment, y=y, phi=phi, energy_norm=E_norm, forces=F)
            )

    def __iter__(self):
        return iter(self.batches)


def _build_cfg(repo_root: Path, out_root: Path) -> dict:
    cfg = load_config(repo_root / "configs" / "tiny.yaml")
    cfg["run"]["out_dir"] = str(out_root)
    cfg["run"]["device"] = "cpu"
    cfg["run"]["precision"] = "fp32"
    cfg["run"]["name"] = "trainer_joint_test"
    cfg["training"]["phase"] = "joint"
    cfg["training"]["batch_size"] = 4
    cfg["training"]["epochs"] = 8
    cfg["training"]["log_every"] = 2
    cfg["training"]["ckpt_every"] = 200
    cfg["training"]["lr"] = 3e-3
    cfg["training"]["weight_decay"] = 0.0
    cfg["training"]["grad_clip"] = 2.0
    cfg["training"]["mh_steps"] = 2
    lw = cfg["training"]["loss_weights"]
    lw["qm_force"] = 0.5
    lw["couple"] = 0.05
    lw["mu"] = 0.2
    lw["couple_l2"] = 5e-3
    # match the coupling potential vocabulary to the synth helpers
    cfg["model"]["coupling"]["n_fragments"] = N_FRAG_VOCAB
    cfg["model"]["coupling"]["n_bond_types"] = N_BOND_TYPES
    return cfg


def test_joint_trainer_end_to_end(tmp_path):
    seed_everything(17)

    repo = Path(__file__).resolve().parents[1]
    cfg = _build_cfg(repo, tmp_path / "runs")

    H = build_hamiltonian(cfg)

    # Buffer initialised from a "wrong" distribution to give the contrast something to do.
    data_pool = [rand_fragment(1, seed=1000 + i, nodes_range=(2, 5)).to_data_list()[0] for i in range(64)]
    buffer_init = [rand_fragment(1, seed=2000 + i, nodes_range=(6, 9)).to_data_list()[0] for i in range(64)]
    buffer = PCDBuffer(size=64, refresh_frac=0.0, seed=0)
    buffer._slots = buffer_init

    mh = FragmentNodeFlipMH(coupling=H.coupling, n_fragments=N_FRAG_VOCAB, beta=1.0)

    loader = _CondIter(n_batches=16, batch_size=cfg["training"]["batch_size"], seed=42)

    trainer = Trainer(cfg, H, train_loader=loader, val_loader=None, device="cpu")
    trainer.attach_pcd(buffer, data_pool, mh)
    trainer.fit(max_steps=80)

    # Parse metrics.jsonl
    rows = [json.loads(l) for l in trainer.metrics_path.read_text().strip().splitlines() if "L_qm" in json.loads(l)]
    assert len(rows) >= 10, f"expected ≥10 rows, got {len(rows)}"

    def avg(key, slc):
        return sum(r[key] for r in slc) / len(slc)

    head, tail = rows[:3], rows[-3:]
    L_qm_0, L_qm_1 = avg("L_qm", head), avg("L_qm", tail)
    L_c_0, L_c_1 = avg("L_couple", head), avg("L_couple", tail)
    L_mu_0, L_mu_1 = avg("L_mu", head), avg("L_mu", tail)

    assert L_qm_1 < 0.8 * L_qm_0, f"L_qm flat: {L_qm_0:.3f} -> {L_qm_1:.3f}"
    assert L_c_1 < L_c_0 - 0.3, f"L_couple flat: {L_c_0:.3f} -> {L_c_1:.3f}"
    assert L_mu_1 < 0.8 * L_mu_0, f"L_mu flat: {L_mu_0:.3f} -> {L_mu_1:.3f}"

    # Checkpoint round-trip
    ckpts = list(trainer.ckpt_dir.glob("joint_*.pt"))
    assert ckpts, "no joint checkpoint written"
    sd = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    assert "state_dict" in sd and "step" in sd
    # All three sub-modules' params should be in the state dict
    keys = sd["state_dict"].keys()
    assert any(k.startswith("qm.") for k in keys)
    assert any(k.startswith("coupling.") for k in keys)
    assert any(k.startswith("mu.") for k in keys)
