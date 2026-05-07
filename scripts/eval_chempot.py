"""Phase-3 exit evaluator for C2 (chemical-potential calibration).

Loads the joint_final checkpoint (CouplingMu module: coupling + mu), probes
``mu(y)`` with structured y targets, and computes Spearman correlations against
two empirical weight references:

  * Bickerton QED weights (RDKit rdkit.Chem.QED.WEIGHT_MEAN), 8-dim. Six of the
    eight align with our phi properties (MW, ALOGP, HBA, HBD, PSA, ROTB); the
    remaining two (AROM, ALERTS) don't appear in our phi and are dropped.

  * Empirical logP-proxy: Pearson correlation of logP with each of the other
    7 properties across the conditional dataset. This is a molecule-level
    surrogate for Wildman-Crippen's per-atom contributions, which aren't
    directly tractable without a full conditional sampler.

Reports pass/fail against the C2 target Spearman > 0.6. Writes
``results/eval/phase3/c2_report.json`` and a bar-plot figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr

from thermofrag.data.zinc_fragments import ZINCFragmentDataset
from thermofrag.potentials.external_field import ChemicalPotentialHead
from thermofrag.potentials.coupling import CouplingPotential


# Bickerton weights extracted from rdkit.Chem.QED.WEIGHT_MEAN.
BICKERTON_WEIGHT_MEAN = {
    "MW": 0.66,
    "ALOGP": 0.46,
    "HBA": 0.05,
    "HBD": 0.61,
    "PSA": 0.06,
    "ROTB": 0.65,
    "AROM": 0.48,
    "ALERTS": 0.95,
}

# Map our phi property names -> Bickerton key (for those 6 that align).
PHI_TO_BICKERTON = {
    "logP": "ALOGP",
    "mw": "MW",
    "hba": "HBA",
    "hbd": "HBD",
    "tpsa": "PSA",
    "rotb": "ROTB",
}


def load_joint_checkpoint(ckpt_path: Path, dataset: ZINCFragmentDataset, hidden: int, coupling_cfg: dict):
    """Reconstruct CouplingPotential + ChemicalPotentialHead from a joint checkpoint.

    The checkpoint stores a flat state_dict for ``CouplingMuModule`` (coupling.* + mu.*).
    """
    n_frag = dataset.n_fragments
    n_prop = dataset.phi_dim
    coupling = CouplingPotential(
        n_fragments=n_frag,
        n_bond_types=int(coupling_cfg.get("n_bond_types", 8)),
        hidden=int(coupling_cfg["hidden"]),
        num_layers=int(coupling_cfg["num_layers"]),
    )
    mu = ChemicalPotentialHead(n_properties=n_prop, hidden=hidden)

    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # Split prefixes.
    coupling_sd = {k[len("coupling."):]: v for k, v in sd.items() if k.startswith("coupling.")}
    mu_sd = {k[len("mu."):]: v for k, v in sd.items() if k.startswith("mu.")}
    coupling.load_state_dict(coupling_sd, strict=False)
    mu.load_state_dict(mu_sd, strict=False)
    coupling.eval()
    mu.eval()
    return coupling, mu


def probe_mu_response(mu: ChemicalPotentialHead, property_names: list[str], device: str) -> np.ndarray:
    """Return an 8x8 response matrix M where
        M[i, j] = mu_j(y = +sigma on property i only) - mu_j(y = 0).
    All y's are in standardized (z-score) coordinates, so +1 means "+1 stddev".
    """
    n_prop = len(property_names)
    mu.to(device).eval()
    with torch.no_grad():
        y0 = torch.zeros(1, n_prop, device=device)
        base = mu(y0).cpu().numpy()[0]  # [K]
        rows = []
        for i in range(n_prop):
            y = torch.zeros(1, n_prop, device=device)
            y[0, i] = 1.0
            rows.append(mu(y).cpu().numpy()[0] - base)
    return np.stack(rows, axis=0)  # [K, K]


def _order_by_bickerton(phi_props: list[str], mu_row: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Extract the 6 phi-properties that align with Bickerton and pair them up.

    Returns (aligned_names, mu_values, bickerton_values).
    """
    aligned = []
    mus = []
    bicks = []
    for i, name in enumerate(phi_props):
        if name in PHI_TO_BICKERTON:
            aligned.append(name)
            mus.append(mu_row[i])
            bicks.append(BICKERTON_WEIGHT_MEAN[PHI_TO_BICKERTON[name]])
    return aligned, np.asarray(mus), np.asarray(bicks)


def empirical_logp_correlations(dataset: ZINCFragmentDataset, phi_props: list[str], n_samples: int = 5000) -> dict[str, float]:
    """Pearson corr of each phi_i with logP across a random sample of the conditional LMDB."""
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    phi = np.stack([dataset[int(i)].phi.numpy() for i in idxs], axis=0)  # [N, K]
    logp_idx = phi_props.index("logP")
    out = {}
    for j, name in enumerate(phi_props):
        if name == "logP":
            continue
        r, _ = pearsonr(phi[:, logp_idx], phi[:, j])
        out[name] = float(r)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("results/checkpoints/joint_final.pt"))
    p.add_argument("--lmdb", type=Path, default=Path("data/processed/chembl_conditional.lmdb"))
    p.add_argument("--coupling-hidden", type=int, default=256)
    p.add_argument("--coupling-layers", type=int, default=4)
    p.add_argument("--mu-hidden", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=Path("results/eval/phase3"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    ds = ZINCFragmentDataset(args.lmdb, split="train")
    phi_props = ds.phi_properties
    print(f"[eval] dataset: n={len(ds)}, phi_dim={ds.phi_dim}, properties={phi_props}")

    coupling, mu = load_joint_checkpoint(
        args.ckpt, ds, hidden=args.mu_hidden,
        coupling_cfg={"hidden": args.coupling_hidden, "num_layers": args.coupling_layers, "n_bond_types": 8},
    )
    print(f"[eval] loaded checkpoint {args.ckpt}")

    # 1. Response matrix.
    M = probe_mu_response(mu, phi_props, args.device)  # [K, K]
    print("[eval] mu response matrix (rows = +1sigma on input property; cols = output dims):")
    print(f"       properties: {phi_props}")
    for i, name in enumerate(phi_props):
        print(f"  y=e_{name:<5s}: " + " ".join(f"{v:+7.3f}" for v in M[i]))

    # 2. C2 test A: Bickerton QED weights vs per-property self-response strength.
    # The diagonal M[i,i] measures how strongly mu_i responds to pushing y_i. Under the
    # thermodynamic identity mu = grad F, properties with more variance in the data (higher
    # importance in the QED desirability sense) should receive stronger mu attention.
    # Compare M[i,i] on the 6 Bickerton-aligned axes to Bickerton's desirability weights.
    mu_diag = np.array([M[i, i] for i in range(len(phi_props))])
    aligned = [n for n in phi_props if n in PHI_TO_BICKERTON]
    mu_diag_aligned = np.array([mu_diag[phi_props.index(n)] for n in aligned])
    bick_vals = np.array([BICKERTON_WEIGHT_MEAN[PHI_TO_BICKERTON[n]] for n in aligned])
    rho_qed, p_qed = spearmanr(mu_diag_aligned, bick_vals)
    print(f"\n[eval][C2 Bickerton] aligned properties: {aligned}")
    for n, md, bv in zip(aligned, mu_diag_aligned, bick_vals):
        print(f"  {n:<6s}  mu_diag={md:+.3f}  bickerton={bv:.2f}")
    print(f"[eval][C2 Bickerton] Spearman(mu_diag, Bickerton) = {rho_qed:.3f} (p={p_qed:.3f})")

    # 3. C2 test B: WC-proxy. Wildman-Crippen is per-atom-type; our phi is molecule-level,
    # so a direct weight comparison is not tractable without a full conditional sampler.
    # Instead, we test whether mu's self-response diagonal aligns with each property's
    # variance-scaled importance in the empirical ChEMBL-conditional distribution: properties
    # with higher absolute correlation to logP (our WC axis) should attract stronger mu
    # diagonal response.
    emp_corr = empirical_logp_correlations(ds, phi_props, n_samples=5000)
    other_names = [n for n in phi_props if n != "logP"]
    mu_diag_others = np.asarray([mu_diag[phi_props.index(n)] for n in other_names])
    emp_abs = np.asarray([abs(emp_corr[n]) for n in other_names])
    rho_logp, p_logp = spearmanr(mu_diag_others, emp_abs)
    print(f"\n[eval][C2 WC-proxy] mu_diag vs |empirical Pearson(logP, phi)|:")
    for n, md, ev in zip(other_names, mu_diag_others, emp_abs):
        print(f"  {n:<6s}  mu_diag={md:+.3f}  |emp_corr_logP|={ev:.3f}")
    print(f"[eval][C2 WC-proxy] Spearman = {rho_logp:.3f} (p={p_logp:.3f})")

    # 4. Laplace uncertainty at probe points (C5 / Fig 6 seed).
    # Report predictive variance norm at y=0 (in-distribution baseline) vs
    # y = 3*e_i (far out on each axis). Healthy Laplace-calibrated heads have
    # larger variance at the out-of-support points.
    with torch.no_grad():
        y_zero = torch.zeros(1, len(phi_props), device=args.device)
        var_zero = mu.predictive_variance(y_zero).cpu().numpy()[0]
        var_extreme = []
        for i in range(len(phi_props)):
            y = torch.zeros(1, len(phi_props), device=args.device)
            y[0, i] = 3.0
            var_extreme.append(mu.predictive_variance(y).cpu().numpy()[0])
        var_extreme = np.stack(var_extreme, axis=0)  # [K, K]
    laplace_fitted = bool(mu._laplace_fitted.item()) if hasattr(mu, "_laplace_fitted") else False
    print(f"\n[eval][Laplace] fitted={laplace_fitted}  var(y=0)_norm={np.linalg.norm(var_zero):.4f}")
    for i, name in enumerate(phi_props):
        print(f"  var(y=3e_{name})_norm={np.linalg.norm(var_extreme[i]):.4f}")

    # 5. Report.
    report = {
        "laplace_fitted": laplace_fitted,
        "var_at_zero": var_zero.tolist(),
        "var_at_3sigma": var_extreme.tolist(),
        "ckpt": str(args.ckpt),
        "phi_properties": phi_props,
        "response_matrix": M.tolist(),
        "bickerton": {
            "aligned_properties": aligned,
            "mu_diag": mu_diag_aligned.tolist(),
            "bickerton_weight_mean": bick_vals.tolist(),
            "spearman": float(rho_qed),
            "pvalue": float(p_qed),
            "pass": bool(rho_qed > 0.6),
        },
        "wc_proxy": {
            "other_properties": other_names,
            "mu_diag": mu_diag_others.tolist(),
            "empirical_logp_pearson_abs": emp_abs.tolist(),
            "spearman": float(rho_logp),
            "pvalue": float(p_logp),
            "pass": bool(rho_logp > 0.6),
        },
        "c2_overall_pass": bool((rho_qed > 0.6) and (rho_logp > 0.6)),
    }
    out_path = args.out / "c2_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[eval] report -> {out_path}")
    print(f"[eval] C2 PASS: Bickerton={report['bickerton']['pass']}  WC-proxy={report['wc_proxy']['pass']}  overall={report['c2_overall_pass']}")

    # Plot response matrix heatmap (Fig 5 panel c seed).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        vmax = float(np.abs(M).max())
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(phi_props)))
        ax.set_yticks(range(len(phi_props)))
        ax.set_xticklabels(phi_props, rotation=45, ha="right")
        ax.set_yticklabels(phi_props)
        ax.set_xlabel("output: mu_j")
        ax.set_ylabel("input: y (one-hot +1sigma on y_i)")
        ax.set_title("Chemical-potential response matrix (dmu_j / dy_i)")
        for i in range(len(phi_props)):
            for j in range(len(phi_props)):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(M[i, j]) < vmax * 0.5 else "white")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        heatmap_path = args.out / "mu_response_heatmap.png"
        fig.savefig(heatmap_path, dpi=150)
        plt.close(fig)
        print(f"[eval] heatmap -> {heatmap_path}")
    except Exception as e:  # pragma: no cover
        print(f"[eval] heatmap plotting skipped: {e}")


if __name__ == "__main__":
    main()
