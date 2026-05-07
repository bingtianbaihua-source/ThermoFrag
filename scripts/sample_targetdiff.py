"""TargetDiff baseline sampler for C3 / C4 generator-vs-generator arm.

Per BASELINES.md §3. TargetDiff is a 3D pocket-conditional equivariant
diffusion model. Per target:

1. Build a 10 A pocket PDB by querying residues within 10 A of the
   ``cognate_ligand.pdb`` heavy-atom positions (upstream helper:
   ``utils.data.PDBProtein``).
2. Run the upstream ``sample_diffusion_ligand`` driver for ``--n-request``
   molecules. We follow the upstream default: num_steps=1000,
   sample_num_atoms=prior, center_pos_mode=protein.
3. Reconstruct molecules from the predicted positions + atom classes,
   canonicalize SMILES, drop invalid / disconnected fragments and
   duplicates, keep the first ``--n-keep`` rows.
4. Write:
     results/eval/phase4_baselines/targetdiff/decoded/<t>.parquet
         target, chain_idx, smiles
     results/eval/phase4_baselines/targetdiff/poses/<t>.sdf
         one 3D pose per kept molecule (first conformer)
   and a project-level ``manifest.json``.

Entry point: run inside the ``targetdiff`` conda env
(python 3.8 + torch 1.13.1 + pyg 2.2.0 + torch-scatter 2.1.0 + rdkit 2022.03).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger("sample_targetdiff")

THERMOFRAG_ROOT = Path(__file__).resolve().parents[1]
TARGETDIFF_ROOT = THERMOFRAG_ROOT / "vendor" / "targetdiff"

DEFAULT_TARGETS = [
    "ADRB2", "ALDH1", "ESR_ago", "ESR_antago", "FEN1",
    "GBA",   "IDH1",  "KAT2A",   "MAPK1",     "MTORC1",
    "OPRK1", "PKM2",  "PPARG",   "TP53",      "VDR",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonicalize(mol):
    """Return canonical SMILES and the mol, or (None, None)."""
    from rdkit import Chem
    if mol is None:
        return None, None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None, None
    try:
        Chem.AssignStereochemistryFrom3D(mol)
    except Exception:
        pass
    smi = Chem.MolToSmiles(mol, canonical=True)
    if not smi or "." in smi:
        return None, None
    return smi, mol


def build_pocket_pdb(receptor_pdb: Path, ligand_pdb: Path,
                     out_pocket_pdb: Path, radius: int = 10) -> None:
    """Write a pocket-cropped PDB by selecting residues within ``radius`` A
    of any cognate ligand heavy-atom."""
    import numpy as np
    from rdkit import Chem
    from utils.data import PDBProtein

    receptor_block = receptor_pdb.read_text()
    protein = PDBProtein(receptor_block)

    # Heavy-atom positions of the cognate ligand. cognate_ligand.pdb is in
    # RDKit-readable PDB format; strip hydrogens for symmetry with
    # upstream parse_sdf_file (which also removes Hs).
    lig_mol = Chem.MolFromPDBFile(str(ligand_pdb), removeHs=True)
    if lig_mol is None:
        raise RuntimeError(f"Failed to parse {ligand_pdb}")
    positions = np.asarray(
        lig_mol.GetConformers()[0].GetPositions(), dtype=np.float32
    )
    residues = protein.query_residues_ligand({"pos": positions}, radius)
    out_pocket_pdb.parent.mkdir(parents=True, exist_ok=True)
    out_pocket_pdb.write_text(
        protein.residues_to_pdb_block(residues, name=f"POCKET_{radius}A")
    )


def load_targetdiff_model(ckpt_path: Path, device: str):
    """Instantiate ScorePosNet3D and load the pretrained diffusion checkpoint."""
    import torch
    import utils.transforms as trans
    from models.molopt_score_model import ScorePosNet3D

    ckpt = torch.load(str(ckpt_path), map_location=device)
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt["config"].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)

    model = ScorePosNet3D(
        ckpt["config"].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt, protein_featurizer, ligand_featurizer


def sample_pocket(model, ckpt, pocket_pdb: Path, num_samples: int,
                  batch_size: int, num_steps: int, device: str):
    """Thin wrapper around ``sample_diffusion_ligand`` + reconstruction."""
    import torch
    from torch_geometric.transforms import Compose
    from datasets.pl_data import ProteinLigandData, torchify_dict
    from utils.data import PDBProtein
    from utils import reconstruct
    import utils.transforms as trans
    from scripts.sample_diffusion import sample_diffusion_ligand

    protein_featurizer = trans.FeaturizeProteinAtom()
    transform = Compose([protein_featurizer])

    pocket_dict = PDBProtein(str(pocket_pdb)).to_dict_atom()
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict={
            "element":       torch.empty([0, ],  dtype=torch.long),
            "pos":           torch.empty([0, 3], dtype=torch.float),
            "atom_feature":  torch.empty([0, 8], dtype=torch.float),
            "bond_index":    torch.empty([2, 0], dtype=torch.long),
            "bond_type":     torch.empty([0, ],  dtype=torch.long),
        },
    )
    data = transform(data)

    r = sample_diffusion_ligand(
        model, data, num_samples,
        batch_size=batch_size, device=device,
        num_steps=num_steps,
        pos_only=False,
        center_pos_mode="protein",
        sample_num_atoms="prior",
    )
    # r is (all_pred_pos, all_pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list)
    all_pred_pos, all_pred_v = r[0], r[1]

    # Reconstruct molecules
    from rdkit import Chem
    mols = []
    n_recon_ok = 0
    for pred_pos, pred_v in zip(all_pred_pos, all_pred_v):
        pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode="add_aromatic")
        try:
            pred_aromatic = trans.is_aromatic_from_index(pred_v, mode="add_aromatic")
            mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
        except reconstruct.MolReconsError:
            mols.append(None)
            continue
        n_recon_ok += 1
        mols.append(mol)
    return mols, n_recon_ok


def write_outputs(target: str, mols, out_parquet: Path, out_sdf: Path,
                  n_keep: int):
    """Canonicalize + dedup + keep first ``n_keep``. Writes parquet + SDF.
    Returns stats dict."""
    import pandas as pd
    from rdkit import Chem

    seen: set[str] = set()
    kept_smiles: list[str] = []
    kept_mols = []
    n_invalid = 0
    n_dup = 0
    for mol in mols:
        smi, clean_mol = _canonicalize(mol)
        if smi is None:
            n_invalid += 1
            continue
        if smi in seen:
            n_dup += 1
            continue
        seen.add(smi)
        kept_smiles.append(smi)
        kept_mols.append(clean_mol)
        if len(kept_smiles) >= n_keep:
            break

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "target":    [target] * len(kept_smiles),
        "chain_idx": list(range(len(kept_smiles))),
        "smiles":    kept_smiles,
    }).to_parquet(out_parquet, index=False)

    out_sdf.parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(out_sdf))
    try:
        for idx, mol in enumerate(kept_mols):
            mol.SetProp("_Name", f"{target}_{idx}")
            w.write(mol)
    finally:
        w.close()

    return {
        "n_candidates": len(mols),
        "n_invalid": n_invalid,
        "n_duplicate": n_dup,
        "n_kept": len(kept_smiles),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--receptors", type=Path,
                   default=THERMOFRAG_ROOT / "data" / "external" / "receptors")
    p.add_argument("--out-dir", type=Path,
                   default=THERMOFRAG_ROOT / "results" / "eval"
                           / "phase4_baselines" / "targetdiff")
    p.add_argument("--ckpt", type=Path,
                   default=TARGETDIFF_ROOT / "pretrained_models"
                           / "pretrained_diffusion.pt")
    p.add_argument("--targets", nargs="*", default=None,
                   help="Subset of targets (default: all 15)")
    p.add_argument("--n-request", type=int, default=1200,
                   help="Raw molecules to generate per target (trimmed to "
                        "--n-keep unique canonical after reconstruction)")
    p.add_argument("--n-keep", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=100,
                   help="Diffusion batch size (upstream default 100)")
    p.add_argument("--num-steps", type=int, default=1000,
                   help="Reverse-diffusion steps (upstream default 1000)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--radius", type=int, default=10,
                   help="Pocket crop radius in angstroms (upstream: 10)")
    p.add_argument("--force", action="store_true",
                   help="Re-sample even if target parquet already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # upstream code uses relative imports like `utils.*`, `datasets.*`
    os.chdir(TARGETDIFF_ROOT)
    sys.path.insert(0, str(TARGETDIFF_ROOT))

    # Upstream references numpy 1.19-era aliases (np.long, np.bool, np.int,
    # np.float, ...) that were removed in numpy 1.24. Patch the obvious ones.
    import numpy as np
    _aliases = {"long": int, "bool": bool, "int": int, "float": float,
                "complex": complex, "object": object, "str": str}
    for k, v in _aliases.items():
        if not hasattr(np, k):
            setattr(np, k, v)
    if not hasattr(np, "compat"):
        class _Compat: pass
        np.compat = _Compat()  # type: ignore[attr-defined]
    for k, v in _aliases.items():
        if not hasattr(np.compat, k):
            setattr(np.compat, k, v)

    # Seed
    import utils.misc as misc
    misc.seed_all(args.seed)

    targets = args.targets or DEFAULT_TARGETS
    decoded_dir = args.out_dir / "decoded"
    poses_dir = args.out_dir / "poses"
    pocket_dir = args.out_dir / "pockets"
    decoded_dir.mkdir(parents=True, exist_ok=True)
    poses_dir.mkdir(parents=True, exist_ok=True)
    pocket_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading TargetDiff model from %s", args.ckpt)
    model, ckpt, _, _ = load_targetdiff_model(args.ckpt, args.device)
    ckpt_sha = _sha256(args.ckpt)

    per_target: dict[str, dict] = {}
    total_t0 = time.time()
    for target in targets:
        out_pq = decoded_dir / f"{target}.parquet"
        out_sdf = poses_dir / f"{target}.sdf"
        if out_pq.exists() and out_sdf.exists() and not args.force:
            logger.info("%s: parquet+sdf exist, skip", target)
            continue
        rec_dir = args.receptors / target
        if not rec_dir.exists():
            logger.error("%s: receptor dir missing %s", target, rec_dir)
            continue

        pocket_pdb = pocket_dir / f"{target}_pocket{args.radius}.pdb"
        if not pocket_pdb.exists() or args.force:
            build_pocket_pdb(
                receptor_pdb=rec_dir / "receptor_clean.pdb",
                ligand_pdb=rec_dir / "cognate_ligand.pdb",
                out_pocket_pdb=pocket_pdb,
                radius=args.radius,
            )
            logger.info("%s: wrote pocket pdb → %s (%d bytes)",
                        target, pocket_pdb, pocket_pdb.stat().st_size)

        t0 = time.time()
        try:
            mols, n_recon_ok = sample_pocket(
                model, ckpt, pocket_pdb,
                num_samples=args.n_request,
                batch_size=args.batch_size,
                num_steps=args.num_steps,
                device=args.device,
            )
        except Exception as exc:
            logger.exception("%s: sampling failed: %s", target, exc)
            per_target[target] = {"error": str(exc)}
            continue
        t_sample = time.time() - t0
        stats = write_outputs(target, mols, out_pq, out_sdf, args.n_keep)
        stats.update({
            "n_requested": args.n_request,
            "n_recon_ok": n_recon_ok,
            "seconds": round(t_sample, 2),
        })
        per_target[target] = stats
        logger.info(
            "%s: kept %d / recon_ok %d / requested %d in %.1fs",
            target, stats["n_kept"], n_recon_ok, args.n_request, t_sample,
        )

    total_wall = time.time() - total_t0

    manifest = {
        "baseline": "targetdiff",
        "checkpoint_path": str(args.ckpt),
        "checkpoint_sha256": ckpt_sha,
        "sampling": {
            "n_request_per_target": args.n_request,
            "n_keep_per_target": args.n_keep,
            "batch_size": args.batch_size,
            "num_steps": args.num_steps,
            "seed": args.seed,
            "radius": args.radius,
            "center_pos_mode": "protein",
            "sample_num_atoms": "prior",
        },
        "per_target": per_target,
        "targetdiff_git_sha": _try_git_sha(TARGETDIFF_ROOT),
        "wall_clock_seconds": round(total_wall, 2),
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("manifest → %s", manifest_path)


def _try_git_sha(repo: Path):
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        )
        return out.strip()
    except Exception:
        return None


if __name__ == "__main__":
    main()
