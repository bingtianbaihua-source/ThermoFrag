"""Phase-2 coupling-PCD trainer integration test.

We create two synthetic fragment-graph distributions:
  * "data"   — small graphs (2–4 fragments), frag_ids drawn from the low half {0..15}
  * "buffer init" — the *opposite* regime: large graphs (6–8), frag_ids from {48..63}

A well-trained CouplingPotential should assign lower V to the data distribution
than the buffer distribution. The PCD loss V(data) - V(buffer) measures exactly
that gap, so it should decrease monotonically during training while the MH
kernel keeps the buffer exploring rather than collapsing.

Verifies:
  - metrics.jsonl rows show L_contrast early > L_contrast late by a real margin
  - MH accept rate stays in [5%, 95%] (kernel is actually working)
  - Final v_pos < final v_neg (data has lower V than buffer)
  - Buffer graphs' frag_ids distribution shifted away from the anti-data init
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch_geometric.data import Data

from thermofrag.potentials.coupling import CouplingPotential
from thermofrag.sampling.fragment_mh import FragmentNodeFlipMH
from thermofrag.training.pcd import PCDBuffer
from thermofrag.training.trainer import Trainer
from thermofrag.utils.config import load_config
from thermofrag.utils.seed import seed_everything

N_FRAG_VOCAB = 64
N_BOND_TYPES = 4


def _make_graph(n_nodes: int, id_low: int, id_high: int, rng: torch.Generator) -> Data:
    frag_id = torch.randint(id_low, id_high, (n_nodes,), generator=rng)
    if n_nodes > 1:
        src = torch.arange(n_nodes - 1)
        dst = torch.arange(1, n_nodes)
        base = torch.stack([src, dst], dim=0)
    else:
        base = torch.zeros(2, 0, dtype=torch.long)
    edge_index = torch.cat([base, base.flip(0)], dim=1) if base.numel() else base
    bond_type = torch.randint(0, N_BOND_TYPES, (edge_index.shape[1],), generator=rng)
    return Data(frag_id=frag_id, edge_index=edge_index, bond_type=bond_type, num_nodes=n_nodes)


def _make_pool(n: int, n_range: tuple[int, int], id_range: tuple[int, int], seed: int) -> list[Data]:
    g = torch.Generator().manual_seed(seed)
    return [
        _make_graph(int(torch.randint(*n_range, (1,), generator=g)), id_range[0], id_range[1], g)
        for _ in range(n)
    ]


def _build_cfg(repo_root: Path, out_root: Path) -> dict:
    cfg = load_config(repo_root / "configs" / "tiny.yaml")
    cfg["run"]["out_dir"] = str(out_root)
    cfg["run"]["device"] = "cpu"
    cfg["run"]["precision"] = "fp32"
    cfg["run"]["name"] = "trainer_coupling_test"
    cfg["training"]["phase"] = "pretrain_coupling"
    cfg["training"]["batch_size"] = 8
    cfg["training"]["epochs"] = 5
    cfg["training"]["log_every"] = 2
    cfg["training"]["ckpt_every"] = 200
    cfg["training"]["lr"] = 3e-3
    cfg["training"]["weight_decay"] = 0.0
    cfg["training"]["grad_clip"] = 2.0
    cfg["training"]["mh_steps"] = 3
    cfg["training"].setdefault("loss_weights", {})["couple_l2"] = 1e-3
    cfg["training"]["pcd"]["buffer_size"] = 64
    cfg["training"]["pcd"]["refresh_frac"] = 0.0  # disable reseeding so we can measure buffer drift cleanly
    return cfg


def test_coupling_pcd_trainer_end_to_end(tmp_path):
    seed_everything(13)

    data_pool = _make_pool(80, n_range=(2, 5), id_range=(0, 16), seed=1)
    buffer_init = _make_pool(64, n_range=(6, 9), id_range=(48, N_FRAG_VOCAB), seed=2)

    repo = Path(__file__).resolve().parents[1]
    cfg = _build_cfg(repo, tmp_path / "runs")

    model = CouplingPotential(
        n_fragments=N_FRAG_VOCAB, n_bond_types=N_BOND_TYPES,
        hidden=cfg["model"]["coupling"]["hidden"], num_layers=cfg["model"]["coupling"]["num_layers"],
    )

    buffer = PCDBuffer(size=cfg["training"]["pcd"]["buffer_size"], refresh_frac=0.0, seed=0)
    buffer._slots = buffer_init[: cfg["training"]["pcd"]["buffer_size"]]
    initial_frag_ids = torch.cat([d.frag_id for d in buffer._slots]).clone()

    mh = FragmentNodeFlipMH(coupling=model, n_fragments=N_FRAG_VOCAB, beta=1.0)

    trainer = Trainer(cfg, model, train_loader=[], val_loader=None, device="cpu")
    trainer.attach_pcd(buffer, data_pool, mh)
    trainer.fit(max_steps=80)

    # -- Metrics trajectory
    lines = trainer.metrics_path.read_text().strip().splitlines()
    rows = [json.loads(ln) for ln in lines if "loss" in json.loads(ln)]
    assert len(rows) >= 10, f"expected many log rows, got {len(rows)}"

    early = sum(r["loss_contrast"] for r in rows[:3]) / 3
    late = sum(r["loss_contrast"] for r in rows[-3:]) / 3
    assert late < early - 0.5, f"coupling contrast loss did not drop: early={early:.3f} late={late:.3f}"

    # -- v_pos should end up below v_neg (data has lower V than buffer)
    last = rows[-1]
    assert last["v_pos_mean"] < last["v_neg_mean"], (
        f"coupling potential did not learn discrimination: v_pos={last['v_pos_mean']} v_neg={last['v_neg_mean']}"
    )

    # -- MH accept rate stays informative (neither frozen nor runaway)
    assert 0.05 <= last["mh_accept_rate"] <= 0.95, f"MH accept_rate={last['mh_accept_rate']:.3f} looks broken"

    # -- Buffer graphs actually evolved (frag_ids changed from the init distribution)
    final_frag_ids = torch.cat([d.frag_id for d in buffer._slots])
    assert initial_frag_ids.shape == final_frag_ids.shape
    drift = (initial_frag_ids != final_frag_ids).float().mean().item()
    assert drift > 0.05, f"buffer did not evolve (drift={drift:.3f})"
