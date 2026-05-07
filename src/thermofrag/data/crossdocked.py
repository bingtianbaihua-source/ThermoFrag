"""Lightweight Dataset over the TF-pocket CrossDocked conditional LMDB.

The LMDB is written by ``scripts/build_crossdocked_lmdb.py``. Each record
holds the ligand SMILES, its 8-dim property vector ``phi``, and a
``pocket_id`` string referencing a precomputed ``.npy`` embedding under
``pocket_embeds_dir``. We do NOT reconstruct a fragment graph here —
TF-pocket fine-tuning freezes coupling + QM and only updates the μ head,
so only ``(phi, pocket_embed, split)`` is needed per batch.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset


_ENV_CACHE: dict[str, lmdb.Environment] = {}


def _shared_env(path: str) -> lmdb.Environment:
    env = _ENV_CACHE.get(path)
    if env is None:
        env = lmdb.open(
            path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        _ENV_CACHE[path] = env
    return env


class CrossDockedConditionalDataset(Dataset):
    def __init__(
        self,
        lmdb_path: str | Path,
        pocket_embeds_dir: str | Path | None = None,
        split: str | None = None,
    ):
        self.path = Path(lmdb_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._split_filter = (
            {"train": 0, "val": 1, "test": 2}.get(split) if split is not None else None
        )
        self._env: lmdb.Environment | None = None
        self._keys: List[bytes] = []
        self.meta: dict = {}
        self._load_index()

        embeds_dir = pocket_embeds_dir or self.meta.get("pocket_embeds_dir")
        if embeds_dir is None:
            raise RuntimeError("pocket_embeds_dir not in meta; pass it explicitly")
        self.pocket_embeds_dir = Path(embeds_dir)
        if not self.pocket_embeds_dir.is_dir():
            raise FileNotFoundError(self.pocket_embeds_dir)

        self._pocket_cache: dict[str, torch.Tensor] = {}

    def _open_env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = _shared_env(str(self.path))
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

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def phi_dim(self) -> int:
        return int(self.meta.get("phi_dim", 0))

    @property
    def phi_properties(self) -> list[str]:
        return list(self.meta.get("phi_properties", []))

    @property
    def phi_mean(self) -> np.ndarray | None:
        m = self.meta.get("phi_mean")
        return np.asarray(m, dtype=np.float32) if m is not None else None

    @property
    def phi_std(self) -> np.ndarray | None:
        s = self.meta.get("phi_std")
        return np.asarray(s, dtype=np.float32) if s is not None else None

    @property
    def pocket_dim(self) -> int:
        # Peek at any cached embedding to discover the dim.
        any_key = next(iter(self._pocket_cache), None)
        if any_key is not None:
            return int(self._pocket_cache[any_key].shape[-1])
        any_npy = next(self.pocket_embeds_dir.glob("*.npy"), None)
        if any_npy is None:
            raise RuntimeError(f"no .npy under {self.pocket_embeds_dir}")
        arr = np.load(str(any_npy))
        return int(arr.shape[-1])

    def _load_pocket(self, pocket_id: str) -> torch.Tensor:
        cached = self._pocket_cache.get(pocket_id)
        if cached is not None:
            return cached
        path = self.pocket_embeds_dir / f"{pocket_id}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        emb = torch.from_numpy(np.load(str(path)).astype(np.float32))
        self._pocket_cache[pocket_id] = emb
        return emb

    def __getitem__(self, idx: int) -> dict:
        env = self._open_env()
        with env.begin() as txn:
            raw = txn.get(self._keys[idx])
        rec = pickle.loads(raw)
        phi = torch.as_tensor(rec["phi"], dtype=torch.float32)
        pocket = self._load_pocket(rec["pocket_id"])
        # vina_dock is only present when the upstream LMDB carried a dock
        # score (TargetDiff preprocessed CrossDocked). Older rows fall back
        # to NaN so V^pocket training can mask them out.
        vina_dock = float(rec.get("vina_dock", float("nan")))
        return {
            "phi": phi,
            "pocket": pocket,
            "pocket_id": rec["pocket_id"],
            "smiles": rec["smiles"],
            "vina_dock": torch.tensor(vina_dock, dtype=torch.float32),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        return state


def collate_pocket_batch(items: list[dict]) -> dict:
    phi = torch.stack([it["phi"] for it in items], dim=0)
    pocket = torch.stack([it["pocket"] for it in items], dim=0)
    vina = torch.stack([it.get("vina_dock", torch.tensor(float("nan"))) for it in items], dim=0)
    return {
        "phi": phi,
        "pocket": pocket,
        "pocket_id": [it["pocket_id"] for it in items],
        "smiles": [it["smiles"] for it in items],
        "vina_dock": vina,
    }


class CrossDockedPocketGeomDataset(Dataset):
    """Geom variant of ``CrossDockedConditionalDataset``.

    Instead of a pre-embedded pocket vector (``.npy`` produced by a frozen
    ESM-2), each record's ``pocket_id`` refers to an ``.npz`` under
    ``pocket_geom_dir`` holding the pocket's Cα coordinates and residue-type
    indices. See ``scripts/precompute_pocket_geometry.py``. Used by
    TF-pocket v3, which trains an EGNN over this geometry end-to-end with
    the μ head.

    ``__getitem__`` returns ``{phi, pocket_coords, pocket_aa, pocket_n,
    pocket_id, smiles, vina_dock}``. Use ``collate_pocket_geom_batch`` to
    pad pockets in a batch.
    """

    def __init__(
        self,
        lmdb_path: str | Path,
        pocket_geom_dir: str | Path,
        split: str | None = None,
    ):
        self.path = Path(lmdb_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.pocket_geom_dir = Path(pocket_geom_dir)
        if not self.pocket_geom_dir.is_dir():
            raise FileNotFoundError(self.pocket_geom_dir)
        self._split_filter = (
            {"train": 0, "val": 1, "test": 2}.get(split) if split is not None else None
        )
        self._env: lmdb.Environment | None = None
        self._keys: List[bytes] = []
        self.meta: dict = {}
        self._load_index()

        self._geom_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _open_env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = _shared_env(str(self.path))
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

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def phi_dim(self) -> int:
        return int(self.meta.get("phi_dim", 0))

    @property
    def phi_properties(self) -> list[str]:
        return list(self.meta.get("phi_properties", []))

    @property
    def phi_mean(self) -> np.ndarray | None:
        m = self.meta.get("phi_mean")
        return np.asarray(m, dtype=np.float32) if m is not None else None

    @property
    def phi_std(self) -> np.ndarray | None:
        s = self.meta.get("phi_std")
        return np.asarray(s, dtype=np.float32) if s is not None else None

    def _load_geom(self, pocket_id: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._geom_cache.get(pocket_id)
        if cached is not None:
            return cached
        path = self.pocket_geom_dir / f"{pocket_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        z = np.load(str(path))
        coords = z["coords"].astype(np.float32, copy=False)
        aa = z["aa_idx"].astype(np.int64, copy=False)
        self._geom_cache[pocket_id] = (coords, aa)
        return coords, aa

    def __getitem__(self, idx: int) -> dict:
        env = self._open_env()
        with env.begin() as txn:
            raw = txn.get(self._keys[idx])
        rec = pickle.loads(raw)
        phi = torch.as_tensor(rec["phi"], dtype=torch.float32)
        coords, aa = self._load_geom(rec["pocket_id"])
        vina_dock = float(rec.get("vina_dock", float("nan")))
        return {
            "phi": phi,
            "pocket_coords": torch.from_numpy(coords),
            "pocket_aa": torch.from_numpy(aa),
            "pocket_n": int(coords.shape[0]),
            "pocket_id": rec["pocket_id"],
            "smiles": rec["smiles"],
            "vina_dock": torch.tensor(vina_dock, dtype=torch.float32),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        return state


def collate_pocket_geom_batch(items: list[dict]) -> dict:
    """Pad a list of geom items into ``{phi (B,K), pocket_coords (B,N,3),
    pocket_aa (B,N), pocket_mask (B,N) bool, pocket_id, smiles, vina_dock}``.
    """
    B = len(items)
    lens = [int(it["pocket_n"]) for it in items]
    N_max = max(max(lens), 1)
    coords = torch.zeros(B, N_max, 3, dtype=torch.float32)
    aa = torch.full((B, N_max), 20, dtype=torch.long)
    mask = torch.zeros(B, N_max, dtype=torch.bool)
    for i, it in enumerate(items):
        n = lens[i]
        if n > 0:
            coords[i, :n] = it["pocket_coords"]
            aa[i, :n] = it["pocket_aa"]
            mask[i, :n] = True
    phi = torch.stack([it["phi"] for it in items], dim=0)
    vina = torch.stack(
        [it.get("vina_dock", torch.tensor(float("nan"))) for it in items], dim=0
    )
    return {
        "phi": phi,
        "pocket_coords": coords,
        "pocket_aa": aa,
        "pocket_mask": mask,
        "pocket_id": [it["pocket_id"] for it in items],
        "smiles": [it["smiles"] for it in items],
        "vina_dock": vina,
    }
