"""Prepare the 15 LIT-PCBA receptor PDBQTs and docking boxes.

The canonical LIT-PCBA site (drugdesign.unistra.fr) is firewalled on this
workstation, so we fall back to the per-target PDB codes that the LIT-PCBA
benchmark paper (Tran-Nguyen et al., JCIM 2020) uses and download them
directly from RCSB. For each target we:

  1. curl the RCSB PDB.
  2. Identify the cognate (largest non-solvent / non-ion) HETATM residue.
  3. Compute its centroid + axis-aligned bounding box.
  4. Strip the structure to protein-only (waters removed, cognate ligand
     removed, non-standard residues removed) and write ``receptor_clean.pdb``.
  5. Call ``prepare_receptor4`` to produce ``receptor.pdbqt``.

Outputs::

    data/external/receptors/<target>/
        <pdbcode>.pdb            # raw from RCSB
        cognate_ligand.pdb        # cognate HETATM residue (for reference)
        receptor_clean.pdb        # protein-only input to prepare_receptor4
        receptor.pdbqt            # for Vina
        box.json                  # { center: [x,y,z], size: [sx,sy,sz], pdb: <code> }

    data/external/receptors/summary.csv

"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import urllib.request

# LIT-PCBA -> (PDB code, expected cognate ligand 3-letter code).  The cognate
# code helps us pick the right HETATM residue when multiple are present. If the
# cognate is unknown / apo we leave it blank and fall back to the largest
# non-solvent HETATM.
TARGET_PDB = {
    "ADRB2":       ("4LDL", "3P0"),   # beta2 adrenergic receptor + BI-167107
    "ALDH1":       ("5L2O", ""),      # ALDH1A1
    "ESR_ago":     ("2P15", "PTI"),   # ERα agonist
    "ESR_antago":  ("2IOK", "369"),   # ERα antagonist
    "FEN1":        ("5FV7", ""),      # FEN1
    "GBA":         ("2V3E", "CBE"),   # β-glucocerebrosidase + conduritol B
    "IDH1":        ("4UMX", "35Q"),   # IDH1 mutant
    "KAT2A":       ("5MLJ", ""),      # KAT2A
    "MAPK1":       ("4FV6", ""),      # MAPK1 / ERK2
    "MTORC1":      ("4DRI", ""),      # FKBP12-rapamycin-FRB
    "OPRK1":       ("6B73", "8ET"),   # κ-opioid receptor (active)
    "PKM2":        ("3GR4", ""),      # PKM2
    "PPARG":       ("3B1M", "570"),   # PPARγ
    "TP53":        ("4HG7", ""),      # MDM2-p53 interface (small-molecule site)
    "VDR":         ("3A2J", "VDX"),   # VDR + vitamin-D analogue
}

SOLVENT = {"HOH", "WAT", "DOD", "SO4", "PO4", "CL", "NA", "MG", "K",
           "CA", "ZN", "MN", "FE", "CU", "NI", "NO3", "ACE", "TRS",
           "EDO", "GOL", "PEG", "DMS", "BME", "IMD", "ACT", "EPE",
           "HEP", "FMT", "CIT", "TAR", "IPA"}

BOX_PADDING = 5.0   # Å added on each side beyond the cognate ligand extent
MIN_BOX_SIDE = 22.5 # Å minimum Vina box side (standard Vina default region)


logger = logging.getLogger("prep_receptors")


@dataclass
class BoxSpec:
    pdb: str
    center: list      # [x, y, z]
    size: list        # [sx, sy, sz]
    cognate_resname: str
    cognate_n_atoms: int
    source: str       # "cognate" | "fallback_centroid"


def _download_pdb(code: str, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 1000:
        return
    url = f"https://files.rcsb.org/download/{code}.pdb"
    urllib.request.urlretrieve(url, out_path)


def _parse_atoms(pdb_path: Path):
    """Return (atom_rows, het_rows) as lists of dicts. Preserves the original
    line text for round-trip writing."""
    atoms, hets = [], []
    with open(pdb_path) as fh:
        for ln in fh:
            if ln.startswith("ATOM "):
                atoms.append(ln)
            elif ln.startswith("HETATM"):
                hets.append(ln)
    return atoms, hets


def _parse_het_rows(rows):
    """Group HETATM rows by (resname, chain, resseq)."""
    groups = {}
    for ln in rows:
        resname = ln[17:20].strip()
        chain = ln[21]
        resseq = ln[22:26].strip()
        key = (resname, chain, resseq)
        try:
            x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
        except ValueError:
            continue
        groups.setdefault(key, []).append((ln, (x, y, z)))
    return groups


def _pick_cognate(het_groups, prefer_resname: str = ""):
    """Return the (key, entries) tuple for the most likely cognate ligand.

    Priority:
      1. The resname matching ``prefer_resname`` (if it exists).
      2. The largest non-solvent HETATM residue.
      3. Otherwise None.
    """
    non_sol = [(k, v) for k, v in het_groups.items() if k[0] not in SOLVENT]
    if not non_sol:
        return None
    if prefer_resname:
        for k, v in non_sol:
            if k[0] == prefer_resname:
                return k, v
    non_sol.sort(key=lambda kv: -len(kv[1]))
    return non_sol[0]


def _box_from_coords(coords: np.ndarray) -> tuple[list, list]:
    center = coords.mean(axis=0)
    extent = coords.max(axis=0) - coords.min(axis=0)
    size = np.maximum(extent + 2 * BOX_PADDING, MIN_BOX_SIDE)
    return center.round(2).tolist(), size.round(2).tolist()


def _write_receptor_clean(atom_rows, out_path: Path):
    """Write ATOM-only PDB with an END record.

    Drops altLoc != ' ' | 'A' (Vina's PDBQT parser trips on digit altLocs, e.g.
    ``HH1A1ARG``). Also normalises any remaining altLoc to ' '.
    """
    kept = []
    for ln in atom_rows:
        if len(ln) < 17:
            kept.append(ln)
            continue
        altloc = ln[16]
        if altloc not in (" ", "A"):
            continue
        if altloc == "A":
            ln = ln[:16] + " " + ln[17:]
        kept.append(ln)
    with open(out_path, "w") as fh:
        fh.writelines(kept)
        fh.write("END\n")


def _write_cognate(cognate_rows, out_path: Path):
    with open(out_path, "w") as fh:
        for ln, _ in cognate_rows:
            fh.write(ln)
        fh.write("END\n")


def _run_prepare_receptor(clean_pdb: Path, out_pdbqt: Path) -> None:
    cmd = [
        "prepare_receptor4", "-r", str(clean_pdb),
        "-o", str(out_pdbqt),
        "-A", "hydrogens",
        "-U", "nphs_lps_waters_nonstdres",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"prepare_receptor4 failed for {clean_pdb}:\n"
            f"STDOUT: {res.stdout}\nSTDERR: {res.stderr}"
        )


def prep_one_target(name: str, code: str, cognate_hint: str,
                    out_root: Path) -> BoxSpec:
    t_dir = out_root / name
    t_dir.mkdir(parents=True, exist_ok=True)
    raw_pdb = t_dir / f"{code}.pdb"
    _download_pdb(code, raw_pdb)

    atoms, hets = _parse_atoms(raw_pdb)
    if not atoms:
        raise RuntimeError(f"{name}: no ATOM records in {code}")

    het_groups = _parse_het_rows(hets)
    cognate = _pick_cognate(het_groups, prefer_resname=cognate_hint)

    if cognate:
        key, entries = cognate
        coords = np.asarray([xyz for _, xyz in entries])
        center, size = _box_from_coords(coords)
        _write_cognate(entries, t_dir / "cognate_ligand.pdb")
        source = "cognate"
        cognate_name = key[0]
        cognate_n = len(entries)
    else:
        coords = np.asarray([
            (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
            for ln in atoms
        ])
        center, size = _box_from_coords(coords)
        # for apo structures we clip to 25Å cube by default
        size = [max(MIN_BOX_SIDE, 25.0)] * 3
        source = "fallback_centroid"
        cognate_name = ""
        cognate_n = 0

    clean_pdb = t_dir / "receptor_clean.pdb"
    _write_receptor_clean(atoms, clean_pdb)

    out_pdbqt = t_dir / "receptor.pdbqt"
    _run_prepare_receptor(clean_pdb, out_pdbqt)

    box = BoxSpec(pdb=code, center=center, size=size,
                  cognate_resname=cognate_name, cognate_n_atoms=cognate_n,
                  source=source)
    with open(t_dir / "box.json", "w") as fh:
        json.dump(asdict(box), fh, indent=2)
    return box


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path,
                   default=Path("data/external/receptors"))
    p.add_argument("--targets", nargs="*",
                   help="Subset of target names to prep (default: all 15).")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args.out.mkdir(parents=True, exist_ok=True)

    picks = TARGET_PDB.items()
    if args.targets:
        keep = set(args.targets)
        picks = [(n, v) for n, v in TARGET_PDB.items() if n in keep]

    rows = []
    for name, (code, hint) in picks:
        logger.info("→ %s (%s, hint=%s)", name, code, hint or "none")
        try:
            box = prep_one_target(name, code, hint, args.out)
        except Exception as exc:
            logger.error("FAILED %s: %s", name, exc)
            rows.append({"target": name, "pdb": code, "status": "failed",
                         "error": str(exc)})
            continue
        rows.append({
            "target": name,
            "pdb": box.pdb,
            "status": "ok",
            "source": box.source,
            "cognate": box.cognate_resname,
            "cognate_n_atoms": box.cognate_n_atoms,
            "center_x": box.center[0],
            "center_y": box.center[1],
            "center_z": box.center[2],
            "size_x": box.size[0],
            "size_y": box.size[1],
            "size_z": box.size[2],
        })
        logger.info("   ok → %s  box=%s size=%s (src=%s)",
                    box.pdb, box.center, box.size, box.source)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "summary.csv", index=False)
    logger.info("summary → %s", args.out / "summary.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
