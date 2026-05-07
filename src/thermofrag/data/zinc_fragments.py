"""PyG Dataset over the Phase-2 ZINC fragment LMDB (and the Phase-3 conditional LMDB).

Reads the LMDB written by ``scripts/build_zinc_fragments.py`` and yields
:class:`torch_geometric.data.Data` objects with the exact shape
:class:`thermofrag.potentials.coupling.CouplingPotential` consumes:

    frag_id:    [N] long
    edge_index: [2, E] long
    bond_type:  [E] long

Plus the following per-graph extras used by Phase-2 evaluation code:

    n_anchors:  [N] long   anchor-slot counts per unit (not used by coupling)
    props:      [4] float  [mw, tpsa, logP, qed] kept for sanity histograms
    smiles:     str        original SMILES (for debugging / property KL)

If the LMDB records carry a ``phi`` field (Phase-3 conditional LMDB built by
``scripts/build_conditional_lmdb.py``), it is also exposed as

    phi:   [K] float  (K = n_properties)

and the Dataset's ``phi_dim`` / ``phi_properties`` attributes are filled from
the LMDB's ``__meta__`` entry.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class ZINCFragmentDataset(Dataset):
    """In-memory-index, on-demand-decode LMDB reader for fragment graphs."""

    def __init__(self, lmdb_path: str | Path, split: str | None = None):
        self.path = Path(lmdb_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._split_filter = {"train": 0, "val": 1, "test": 2}.get(split) if split is not None else None
        self._env: lmdb.Environment | None = None
        self._keys: List[bytes] = []
        self.meta: dict = {}
        self._load_index()

    # ------------------------------------------------------------------ index

    def _open_env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(
                str(self.path),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        return self._env

    def _load_index(self) -> None:
        env = self._open_env()
        with env.begin() as txn:
            meta_bytes = txn.get(b"__meta__")
            if meta_bytes is None:
                raise RuntimeError(f"{self.path} missing __meta__ key")
            self.meta = pickle.loads(meta_bytes)
            cur = txn.cursor()
            keys: List[bytes] = []
            for k, v in cur:
                if k == b"__meta__":
                    continue
                if self._split_filter is not None:
                    rec = pickle.loads(v)
                    if rec.get("split") != self._split_filter:
                        continue
                keys.append(bytes(k))
            self._keys = keys

    # -------------------------------------------------------------- PyG Data

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def n_fragments(self) -> int:
        return int(self.meta["n_fragments"])

    @property
    def phi_dim(self) -> int:
        """Dimensionality of the stored property vector phi. 0 if LMDB is unconditional."""
        return int(self.meta.get("phi_dim", 0))

    @property
    def phi_properties(self) -> list[str]:
        """Property names in the order they appear in phi. Empty if unconditional."""
        return list(self.meta.get("phi_properties", []))

    @property
    def phi_mean(self) -> np.ndarray | None:
        m = self.meta.get("phi_mean")
        return np.asarray(m, dtype=np.float32) if m is not None else None

    @property
    def phi_std(self) -> np.ndarray | None:
        s = self.meta.get("phi_std")
        return np.asarray(s, dtype=np.float32) if s is not None else None

    def _record_to_data(self, rec: dict) -> Data:
        frag_id = torch.as_tensor(rec["frag_id"], dtype=torch.long)
        if len(rec["edge_index"]) == 0:
            # Single-unit graph: empty edge_index must have shape [2, 0].
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            bond_type = torch.zeros(0, dtype=torch.long)
        else:
            ei = np.asarray(rec["edge_index"], dtype=np.int64).T  # [2, E]
            edge_index = torch.from_numpy(ei)
            bond_type = torch.as_tensor(rec["bond_type"], dtype=torch.long)
        n_anchors = torch.as_tensor(rec["n_anchors"], dtype=torch.long)
        props = torch.tensor(
            [rec["props"]["mw"], rec["props"]["tpsa"], rec["props"]["logp"], rec["props"]["qed"]],
            dtype=torch.float32,
        )
        data = Data(
            frag_id=frag_id,
            edge_index=edge_index,
            bond_type=bond_type,
        )
        data.n_anchors = n_anchors
        data.props = props
        data.smiles = rec["smiles"]
        if "phi" in rec:
            data.phi = torch.as_tensor(rec["phi"], dtype=torch.float32)
        # PyG uses num_nodes for batching if `x` is absent.
        data.num_nodes = int(frag_id.numel())
        return data

    def __getitem__(self, idx: int) -> Data:
        env = self._open_env()
        with env.begin() as txn:
            raw = txn.get(self._keys[idx])
        rec = pickle.loads(raw)
        return self._record_to_data(rec)

    # Avoid pickling the LMDB handle when the Dataset is passed to workers.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        return state
