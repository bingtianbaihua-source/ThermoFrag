# Abstract

Inverse molecular design must simultaneously satisfy competing constraints including binding activity, drug-likeness, and synthesizability, across an intractable chemical space. Existing generative models perform well on benchmarks, yet rarely provide insight into whether learned structure--property associations reflect genuine chemical principles or statistical artifacts. This interpretability gap is especially acute in conditioning-vector-based approaches, where learned embeddings offer no direct correspondence to physicochemical quantities, limiting reliable generalization to new targets. We introduce ThermoFrag, a fragment-based 3D molecular generation framework that decomposes a unified Hamiltonian into three interpretable terms (quantum-mechanical energy, fragment compatibility, and chemical potential) and generates molecules by sampling from a conditional Boltzmann distribution. The chemical potential field achieves Spearman correlations of $0.714$ and $0.607$ with QED and $\log P$, respectively, versus near-zero correlations for conditioning embeddings in baseline methods ($|\rho| < 0.1$). Its off-diagonal gradient field recovers Lipinski, Veber, Bickerton and Ertl couplings and surfaces a candidate novel HBA$\leftrightarrow$HBD allocation trade-off (ChEMBL partial $r=-0.514$; LIT-PCBA actives $r=-0.416$). The energy term reaches per-atom chemical accuracy on SPICE-v2, and the same posterior reliably flags out-of-distribution inputs. In structure-based design, ThermoFrag outperforms the unsupervised baseline TargetDiff on 14 of 15 LIT-PCBA targets.


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
