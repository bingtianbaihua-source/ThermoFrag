"""Persistent contrastive divergence buffer.

Tieleman 2008. A fixed-size pool of MCMC chain states; each training step pulls
a batch, runs k moves under the current model, writes the evolved states back.

Design notes:
  * States are PyG :class:`torch_geometric.data.Data` objects (one fragment
    graph per slot). Storing detached CPU-tensored Data keeps memory small and
    lets us push big buffers without hogging GPU.
  * ``sample_batch`` draws a minibatch, returns it as a Batch plus the slot
    indices so ``update_from_batch`` can write evolved states back into the
    same slots.
  * ``refresh_frac`` is the fraction of slots reseeded from the data
    distribution each step (Tieleman's small-refresh trick to prevent chains
    from drifting too far from data support).
"""
from __future__ import annotations

import random
from typing import Iterable, Sequence

from torch_geometric.data import Batch, Data


class PCDBuffer:
    def __init__(self, size: int = 4096, refresh_frac: float = 0.05, seed: int = 0):
        self.size = int(size)
        self.refresh_frac = float(refresh_frac)
        self._rng = random.Random(seed)
        self._slots: list[Data] = []

    # ---- initialization -------------------------------------------------

    def init_from_dataset(self, dataset: Sequence[Data], n: int | None = None) -> None:
        n = self.size if n is None else min(int(n), self.size)
        if len(dataset) == 0:
            raise ValueError("dataset is empty")
        self._slots = [self._detach_cpu(dataset[self._rng.randrange(len(dataset))]) for _ in range(n)]

    @staticmethod
    def _detach_cpu(d: Data) -> Data:
        out = d.clone()
        for k, v in out:
            if hasattr(v, "detach"):
                out[k] = v.detach().cpu()
        return out

    # ---- sampling -------------------------------------------------------

    def sample_batch(self, batch_size: int) -> tuple[Batch, list[int]]:
        if not self._slots:
            raise RuntimeError("PCDBuffer empty; call init_from_dataset first")
        idxs = self._rng.sample(range(len(self._slots)), k=min(batch_size, len(self._slots)))
        batch = Batch.from_data_list([self._slots[i] for i in idxs])
        return batch, idxs

    # ---- writing back ---------------------------------------------------

    def update_from_batch(self, batch: Batch, slot_idxs: list[int], data_pool: Sequence[Data] | None = None) -> None:
        """Write evolved graphs back into their slots, optionally reseeding a small fraction."""
        data_list = batch.cpu().to_data_list()
        assert len(data_list) == len(slot_idxs), "batch size must match slot list"
        for pos, slot_i in enumerate(slot_idxs):
            self._slots[slot_i] = self._detach_cpu(data_list[pos])
        # Reseed a small fraction of total buffer slots from data_pool.
        if data_pool is not None and self.refresh_frac > 0:
            n_refresh = max(1, int(self.refresh_frac * len(self._slots)))
            for _ in range(n_refresh):
                slot_i = self._rng.randrange(len(self._slots))
                self._slots[slot_i] = self._detach_cpu(data_pool[self._rng.randrange(len(data_pool))])

    def __len__(self) -> int:
        return len(self._slots)
