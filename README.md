# ThermoFrag

A statistical-mechanical, SE(3)-equivariant fragment-assembly generative model for multi-objective small-molecule design.

Hardware envelope: a single NVIDIA RTX 4060 16 GB for original development; later experiments (TF-pocket, MM-GBSA) ran on dual RTX 3090. Every design choice is sized to fit consumer GPUs without sacrificing the first-principles claim.

---

## One-paragraph pitch

Existing multi-objective molecular generators treat property targets as opaque conditioning vectors and reward weights, which produces molecules that violate Pareto coherence and physical chemistry once they leave the model. ThermoFrag rewrites fragment-based 3D generation as conditional sampling from a Boltzmann distribution whose Hamiltonian decomposes into a QM-grounded internal energy, a learned coupling potential, and a property external field with a calibrated chemical potential. The sampler satisfies detailed balance, so generated molecules are unbiased samples from a physically meaningful target distribution rather than ad hoc autoregressive outputs. In the zero-temperature limit ThermoFrag recovers BBAR-style greedy fragment assembly; with the QM term off it recovers data-driven density modeling; with the property field off it recovers an unconditional 3D generator. ThermoFrag is therefore a strict generalization of three existing paradigms with new physics put in by hand.

---

## Status (2026-05)

The full pipeline is implemented and benchmarked. Falsifiable claims and verdicts:

| ID  | Claim | Verdict |
|-----|-------|---------|
| C1  | Internal-energy head agrees with DFT single-points (Spearman > 0.9, MAE < 5 kcal/mol) | **PASS** — per-atom MAE 0.49 kcal/mol on drug-like holdout. |
| C2  | Learned chemical-potential vector matches Wildman-Crippen logP / Bickerton QED weights (ρ > 0.6) | **PASS** with surfaced novel HBA↔HBD trade-off (Cat-E, replicated on ChEMBL + LIT-PCBA). |
| C3  | Pocket-agnostic generator beats baselines on docking under matched-pool fairness | **PASS vs TargetDiff** (14/15 sig, mean −0.94 kcal/mol). **TIE vs RxnFlow / BBAR** (2/15 each — score-aware baselines). |
| C4  | Generated molecules have lower post-relaxation strain than baselines | **PARTIAL** — strain not inflated beyond baselines but original "d > 0.3" threshold not met. |
| C5  | Laplace-approx μ uncertainty flags OOD targets (AUROC > 0.8) | **PASS** on ChEMBL Pareto-frontier holdout. |
| C6  | Each Hamiltonian term is independently necessary (QM, coupling, μ) | **PASS** — three independent ablations each collapse the corresponding claim. |

A pocket-conditional follow-on (TF-pocket v1–v4) was explored to close the score-aware-baseline gap. Net result: μ-only and V^pocket couplings regress; EGNN-over-Cα v3 is the best variant but does not flip the C3 score-aware tie.

---

## Repository map

```
ThermoFrag/
  README.md                  This file.
  src/thermofrag/            Library source.
    model/                   SE(3)-equivariant backbone, Hamiltonian heads, μ head.
    sampling/                Discrete MH on fragment graph + Langevin on coordinates, temperature annealing.
    potentials/              QM / coupling / external-field decomposition.
    training/                Loss assembly, optimizer, persistent-CD buffer.
    data/                    Dataset wrappers (SPICE, QMugs, ZINC, ChEMBL, LIT-PCBA, CrossDocked2020).
    eval/                    Evaluation, OOD breakdown, ProLIF, MM-GBSA driver.
    utils/                   Logging, seeding, I/O.
  vendor/                    Vendored upstream code: BBAR (BRICS), TargetDiff, RxnFlow, PharmacoNet, PMNet weights.
  data/
    raw/                     Untouched downloads (gitignored).
    processed/               Cached preprocessed tensors (gitignored).
    external/                BBAR-shipped datasets: LIT-PCBA, ZINC, fragment library.
  configs/                   YAML run configs (default, phase1_large, phase2/3, tiny, tf_pocket{,_v3}).
  scripts/                   74 entrypoints — preprocess, train, sample, dock (Vina/Smina/MM-GBSA), evaluate (C1..C6).
  notebooks/                 Exploratory notebooks.
  results/                   Metrics + summaries (checkpoints / logs / large figures gitignored).
  tests/                     Pytest unit tests (Hamiltonian smoke, Langevin, decoder, conditional MH, mini-train).
```

---

## Quick start

```bash
# environment (CUDA 12.1)
conda env create -f environment.yml
conda activate thermofrag

# preprocess SPICE / QMugs / ZINC / ChEMBL / LIT-PCBA into LMDB caches
python scripts/preprocess.py --config configs/default.yaml

# train (Hamiltonian-head warm-up → joint training → μ head)
python scripts/train.py --config configs/default.yaml

# sample 1000 ligands per target with property vector y
python scripts/sample.py --config configs/default.yaml --n 1000

# evaluate C1..C6
python scripts/eval_qm.py            # C1
python scripts/eval_chempot.py       # C2
python scripts/eval_generator_vs_generator.py  # C3 (matched-pool, three baselines)
python scripts/eval_pareto.py        # C4
python scripts/eval_ood_auroc.py     # C5
python scripts/eval_c6_unified.py    # C6 ablations
```

Pocket-conditional variant:

```bash
python scripts/build_crossdocked_lmdb.py
python scripts/train.py --config configs/tf_pocket_v3.yaml
python scripts/eval_c3_c4_c6_summary.py --variant tf_pocket_v3
```

---

## Citation

Manuscript under preparation. A preprint and final citation will be added here once the submission is filed.

---

## License

MIT (see `pyproject.toml`). Vendored upstream code retains its own licenses — see `vendor/<package>/LICENSE`.
