"""Dump per-target LIT-PCBA pocket embeddings from a trained TF-pocket v3 checkpoint.

Runs the trained ``EGNNPocketEncoder`` on each LIT-PCBA pocket geometry
``.npz`` under ``<in-dir>`` and writes one ``<target>.npy`` under
``<out-dir>``. The sampler (``scripts/sample.py``) reads those ``.npy``
files via ``--pocket-embed`` exactly as in v1/v2, so no sampler-side
change is required for v3.

Usage::

    python scripts/dump_pocket_v3_litpcba.py \\
        --ckpt results/checkpoints/tf_pocket_v3_final.pt \\
        --in-dir  data/processed/pocket_geom/litpcba \\
        --out-dir data/processed/pocket_embeds/litpcba_v3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from thermofrag.potentials.pocket_egnn import EGNNPocketEncoder, load_pocket_geom


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True,
                   help="TF-pocket v3 checkpoint (must carry encoder_state_dict + encoder_cfg)")
    p.add_argument("--in-dir", type=Path, required=True,
                   help="dir of pocket geometry .npz (e.g. data/processed/pocket_geom/litpcba)")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="dir to write <target>.npy pocket vectors")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    blob = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    enc_cfg = blob.get("encoder_cfg")
    if enc_cfg is None:
        raise SystemExit(f"{args.ckpt} missing encoder_cfg — is this a v3 checkpoint?")
    enc_sd = blob["encoder_state_dict"]
    encoder = EGNNPocketEncoder(
        embed_dim=int(enc_cfg["embed_dim"]),
        n_layers=int(enc_cfg["n_layers"]),
        n_rbf=int(enc_cfg.get("n_rbf", 16)),
        cutoff_a=float(enc_cfg.get("cutoff_a", 10.0)),
    )
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    if missing or unexpected:
        print(f"[dump-v3] encoder load: missing={missing} unexpected={unexpected}")
    encoder.to(args.device).eval()

    n_ok = 0
    for npz in sorted(args.in_dir.glob("*.npz")):
        target = npz.stem
        coords, aa = load_pocket_geom(npz)
        if coords.shape[0] == 0:
            print(f"[dump-v3] {target}: empty pocket, skipping")
            continue
        vec = encoder.encode_single(coords, aa).cpu().float().numpy()
        out_path = args.out_dir / f"{target}.npy"
        np.save(str(out_path), vec)
        print(f"[dump-v3] {target}: n_res={coords.shape[0]} dim={vec.shape[0]} -> {out_path}")
        n_ok += 1
    print(f"[dump-v3] wrote {n_ok} .npy vectors to {args.out_dir}")


if __name__ == "__main__":
    main()
