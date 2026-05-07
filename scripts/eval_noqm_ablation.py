"""No-QM ablation (C6 → C1 collapse).

Trains nothing; just builds a fresh QMHead with the same architecture as
``results/checkpoints/qm_final.pt`` but keeps the random-init weights, then
runs ``scripts/eval_qm.py``'s evaluation loop on the SPICE val split. The
Spearman correlation on energies should collapse from ~0.995 (trained) to
~0 (random), making the C1 claim's dependence on QM training falsifiable.

Output::

    results/eval/phase5/c6_noqm.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_add

from thermofrag.data.spice import SPICEShard
from thermofrag.training.trainer import build_qm_head


logger = logging.getLogger("noqm")


def _load_eval(data_dir: Path):
    import json as _json
    shard_dir = data_dir if any(data_dir.glob("shard_*.npz")) else data_dir / "val"
    if not any(shard_dir.glob("shard_*.npz")):
        shard_dir = data_dir / "train"
    mf = shard_dir.parent / "manifest.json"
    stats = _json.loads(mf.read_text()).get("energy_stats", {"mean": 0.0, "std": 1.0}) \
        if mf.is_file() else {"mean": 0.0, "std": 1.0}
    return SPICEShard(shard_dir, energy_mean=stats["mean"], energy_std=stats["std"])


def _spearman(a, b):
    def rank(x):
        order = np.argsort(x, kind="stable")
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(x))
        return r
    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    denom = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def evaluate(model, dataset, device, batch_size, max_samples):
    mean = dataset.energy_mean; std = dataset.energy_std
    n_total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    subset = torch.utils.data.Subset(dataset, list(range(n_total)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    y_true, y_pred = [], []
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            scalar, _ = model.backbone(batch)
            atom_e = model.energy_mlp(scalar).squeeze(-1) + model.atom_ref(batch.z).squeeze(-1)
            E_pred_norm = scatter_add(atom_e, batch.batch, dim=0)
        y_true.append(batch.energy.float().cpu().numpy())
        y_pred.append(E_pred_norm.float().cpu().numpy() * std + mean)
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    err = y_pred - y_true
    return {
        "n": int(y_true.shape[0]),
        "energy_mae_kcal_per_mol": float(np.mean(np.abs(err))),
        "energy_rmse_kcal_per_mol": float(math.sqrt(np.mean(err * err))),
        "energy_spearman": _spearman(y_true, y_pred),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path,
                   default=Path("results/checkpoints/qm_final.pt"))
    p.add_argument("--data", type=Path,
                   default=Path("data/processed/spice_val"))
    p.add_argument("--out", type=Path,
                   default=Path("results/eval/phase5/c6_noqm.json"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-samples", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    # Trained QM reference (recompute, not trust Phase 1 JSON).
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    trained = build_qm_head(blob["cfg"]).to(device).eval()
    trained.load_state_dict(blob["state_dict"], strict=True)

    # Random-init model: identical architecture, untrained.
    torch.manual_seed(123)
    random_head = build_qm_head(blob["cfg"]).to(device).eval()

    dataset = _load_eval(args.data)
    logger.info("dataset: %s  n=%d", args.data, len(dataset))

    m_trained = evaluate(trained, dataset, device,
                         args.batch_size, args.max_samples)
    m_random = evaluate(random_head, dataset, device,
                        args.batch_size, args.max_samples)
    logger.info("trained: %s", m_trained)
    logger.info("random:  %s", m_random)

    spearman_drop = m_trained["energy_spearman"] - m_random["energy_spearman"]
    mae_gap = m_random["energy_mae_kcal_per_mol"] - m_trained["energy_mae_kcal_per_mol"]
    report = {
        "trained":       m_trained,
        "random_init":   m_random,
        "spearman_drop": spearman_drop,
        "mae_gap_kcal":  mae_gap,
        "c6_c1_collapse_confirmed": bool(
            m_trained["energy_spearman"] > 0.5 and m_random["energy_spearman"] < 0.2
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    logger.info("C6-noQM: trained Spearman %.4f → random %.4f  (drop %.4f)  %s",
                m_trained["energy_spearman"], m_random["energy_spearman"],
                spearman_drop,
                "✓" if report["c6_c1_collapse_confirmed"] else "✗")
    logger.info("report → %s", args.out)


if __name__ == "__main__":
    main()
