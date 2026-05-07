"""Diagnose why v2's V^pocket failed on the sampler distribution.

Loads v2's ``PocketLigandCoupling`` checkpoint and evaluates it on the
existing phase4 samples + Vina labels. Reports Pearson/Spearman of the
predicted Vina score vs. the ground-truth Vina score under two phi_z
standardization conventions:

(a) CrossDocked phi_mean/std -- the convention v2's training used.
(b) ChEMBL     phi_mean/std -- the convention the sampler uses at eval.

If (b) materially beats (a), v2's failure is a standardization bug and
the fix is a straight retrain with ChEMBL stats. If both are low, the
ligand representation (phi_z alone, 8-d) is too compressed and v4 needs
a richer encoder. Either way, this cheap check decides the next step.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from thermofrag.data.crossdocked import CrossDockedConditionalDataset
from thermofrag.data.zinc_fragments import ZINCFragmentDataset
from thermofrag.potentials.pocket_coupling import PocketLigandCoupling
from thermofrag.sampling.conditional_mh import build_frag_phi_table


LITPCBA = [
    "ADRB2", "ALDH1", "ESR_ago", "ESR_antago", "FEN1", "GBA", "IDH1",
    "KAT2A", "MAPK1", "MTORC1", "OPRK1", "PKM2", "PPARG", "TP53", "VDR",
]


def _load_v_pocket(ckpt: Path, device: str) -> tuple[PocketLigandCoupling, int]:
    blob = torch.load(str(ckpt), map_location=device, weights_only=False)
    sd = blob["v_pocket_state_dict"]
    meta = blob.get("v_pocket_meta", {})
    pocket_dim = int(sd["pocket_proj.0.weight"].shape[1])
    phi_dim = int(sd["mlp.0.weight"].shape[1] - int(meta.get("pocket_hidden", 64)))
    head = PocketLigandCoupling(
        phi_dim=phi_dim,
        pocket_dim=pocket_dim,
        pocket_hidden=int(meta.get("pocket_hidden", 64)),
        mlp_hidden=int(meta.get("mlp_hidden", 128)),
    )
    head.load_state_dict(sd, strict=True)
    head.to(device).eval()
    print(f"[diag] v2 loaded: phi_dim={phi_dim} pocket_dim={pocket_dim}")
    print(
        f"[diag]   calib vina_mean={float(head.vina_mean):.3f} "
        f"vina_scale={float(head.vina_scale):.3f}"
    )
    return head, pocket_dim


def _phi_from_frag_ids(
    frag_ids: list[int], frag_phi: np.ndarray
) -> np.ndarray:
    """phi_raw(m) = sum_i frag_phi[frag_id_i]. Matches _phi_of_batch in conditional_mh.py."""
    if not frag_ids:
        return np.zeros(frag_phi.shape[1], dtype=np.float32)
    return frag_phi[frag_ids].sum(axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v-pocket-ckpt", type=Path,
                   default=Path("results/checkpoints/tf_pocket_v2_final.pt"))
    p.add_argument("--samples-dir", type=Path,
                   default=Path("results/eval/phase4/samples"))
    p.add_argument("--vina-dir", type=Path,
                   default=Path("results/eval/phase4/vina"))
    p.add_argument("--pocket-embeds", type=Path,
                   default=Path("data/processed/pocket_embeds/litpcba"))
    p.add_argument("--library", type=Path,
                   default=Path("data/processed/fragment_library.parquet"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # Model
    head, pocket_dim = _load_v_pocket(args.v_pocket_ckpt, device)

    # phi_mean/std from both sources.
    cd = CrossDockedConditionalDataset(
        "data/processed/crossdocked_conditional.lmdb",
        pocket_embeds_dir="data/processed/pocket_embeds/crossdocked",
    )
    properties = cd.phi_properties
    phi_mean_cd = cd.phi_mean.astype(np.float32)
    phi_std_cd = cd.phi_std.astype(np.float32)

    chembl = ZINCFragmentDataset("data/processed/chembl_conditional.lmdb")
    phi_mean_cb = np.asarray(chembl.meta["phi_mean"], dtype=np.float32)
    phi_std_cb = np.asarray(chembl.meta["phi_std"], dtype=np.float32)
    print(f"[diag] phi CD vs ChEMBL means:\n  CD   : {phi_mean_cd.round(3).tolist()}\n  ChEMBL: {phi_mean_cb.round(3).tolist()}")

    # frag_phi table (matches sampler's _phi_of_batch convention)
    print("[diag] building frag_phi table...")
    frag_phi = build_frag_phi_table(args.library, properties)
    print(f"[diag]   shape={frag_phi.shape}")

    # Aggregate predictions per target.
    records = []
    for target in LITPCBA:
        samp_pkl = args.samples_dir / f"{target}.pkl"
        vina_pq = args.vina_dir / f"{target}.parquet"
        pkt_npy = args.pocket_embeds / f"{target}.npy"
        if not (samp_pkl.exists() and vina_pq.exists() and pkt_npy.exists()):
            print(f"[diag] skip {target}: missing files")
            continue
        with open(samp_pkl, "rb") as f:
            d = pickle.load(f)
        chain_samples = d["samples"]  # list of dicts with frag_id

        vdf = pd.read_parquet(vina_pq)
        vdf = vdf[vdf.status == "ok"][["chain_idx", "vina_score"]]
        if len(vdf) == 0:
            continue

        pocket = np.load(pkt_npy).astype(np.float32)
        if pocket.shape[-1] != pocket_dim:
            raise RuntimeError(f"{target} pocket dim {pocket.shape} != ckpt {pocket_dim}")

        # phi_raw per Vina-labeled chain.
        phi_raws = []
        for _, row in vdf.iterrows():
            ci = int(row.chain_idx)
            fi = chain_samples[ci]["frag_id"]
            phi_raws.append(_phi_from_frag_ids(fi, frag_phi))
        phi_raws = np.asarray(phi_raws, dtype=np.float32)  # [N, K]
        vinas = vdf.vina_score.to_numpy().astype(np.float32)  # [N]

        # Two standardizations.
        phi_z_cd = (phi_raws - phi_mean_cd) / phi_std_cd
        phi_z_cb = (phi_raws - phi_mean_cb) / phi_std_cb
        n = phi_raws.shape[0]
        pkt_t = torch.from_numpy(pocket).to(device).unsqueeze(0).expand(n, -1)

        with torch.no_grad():
            pred_cd = head(torch.from_numpy(phi_z_cd).to(device), pkt_t).cpu().numpy()
            pred_cb = head(torch.from_numpy(phi_z_cb).to(device), pkt_t).cpu().numpy()

        records.append({
            "target": target,
            "n": n,
            "vina_mean": float(vinas.mean()),
            "vina_std": float(vinas.std()),
            "pred_cd_mean": float(pred_cd.mean()),
            "pred_cb_mean": float(pred_cb.mean()),
            "pearson_cd": float(np.corrcoef(vinas, pred_cd)[0, 1]) if n > 2 else float("nan"),
            "pearson_cb": float(np.corrcoef(vinas, pred_cb)[0, 1]) if n > 2 else float("nan"),
        })
        print(
            f"[diag] {target:12s} n={n:4d}  "
            f"pearson CD={records[-1]['pearson_cd']:+.3f}  "
            f"pearson ChEMBL={records[-1]['pearson_cb']:+.3f}  "
            f"vina_mean={records[-1]['vina_mean']:+.2f}  "
            f"pred_cd={records[-1]['pred_cd_mean']:+.2f}  "
            f"pred_cb={records[-1]['pred_cb_mean']:+.2f}"
        )

    # Pooled Pearson: concat all predictions and all truths.
    all_v = []
    all_pcd = []
    all_pcb = []
    for target in LITPCBA:
        samp_pkl = args.samples_dir / f"{target}.pkl"
        vina_pq = args.vina_dir / f"{target}.parquet"
        pkt_npy = args.pocket_embeds / f"{target}.npy"
        if not (samp_pkl.exists() and vina_pq.exists() and pkt_npy.exists()):
            continue
        with open(samp_pkl, "rb") as f:
            d = pickle.load(f)
        chain_samples = d["samples"]
        vdf = pd.read_parquet(vina_pq)
        vdf = vdf[vdf.status == "ok"][["chain_idx", "vina_score"]]
        pocket = np.load(pkt_npy).astype(np.float32)
        phi_raws = np.stack(
            [_phi_from_frag_ids(chain_samples[int(r.chain_idx)]["frag_id"], frag_phi)
             for _, r in vdf.iterrows()], axis=0,
        )
        vinas = vdf.vina_score.to_numpy().astype(np.float32)
        phi_z_cd = (phi_raws - phi_mean_cd) / phi_std_cd
        phi_z_cb = (phi_raws - phi_mean_cb) / phi_std_cb
        pkt_t = torch.from_numpy(pocket).to(device).unsqueeze(0).expand(vinas.size, -1)
        with torch.no_grad():
            pred_cd = head(torch.from_numpy(phi_z_cd).to(device), pkt_t).cpu().numpy()
            pred_cb = head(torch.from_numpy(phi_z_cb).to(device), pkt_t).cpu().numpy()
        all_v.append(vinas)
        all_pcd.append(pred_cd)
        all_pcb.append(pred_cb)
    all_v = np.concatenate(all_v)
    all_pcd = np.concatenate(all_pcd)
    all_pcb = np.concatenate(all_pcb)
    pcd = float(np.corrcoef(all_v, all_pcd)[0, 1])
    pcb = float(np.corrcoef(all_v, all_pcb)[0, 1])

    print()
    print(f"[diag] POOLED Pearson across all {all_v.size} labeled samples:")
    print(f"         CrossDocked standardization (training-consistent): {pcd:+.3f}")
    print(f"         ChEMBL      standardization (sampler-consistent):  {pcb:+.3f}")

    df = pd.DataFrame(records)
    df.to_csv("results/eval/v4_diag_v2_on_phase4.csv", index=False)
    print(f"\n[diag] per-target table -> results/eval/v4_diag_v2_on_phase4.csv")


if __name__ == "__main__":
    main()
