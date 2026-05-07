"""Top-level training loops for the three phases.

Phases (set training.phase in the config):
  pretrain_qm        Train QMHead on SPICE.
  pretrain_coupling  Train CouplingPotential with PCD on ZINC.
  joint              Train all three heads jointly with full Hamiltonian, on ChEMBL conditional.

Each phase saves a checkpoint into results/checkpoints/<phase>.pt and a
metrics.jsonl into results/logs/<run.name>/.
"""
from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_add

from thermofrag.data.spice import SPICEShard
from thermofrag.model.painn import PaiNNConfig
from thermofrag.potentials.coupling import CouplingPotential
from thermofrag.potentials.external_field import ChemicalPotentialHead
from thermofrag.potentials.hamiltonian import Hamiltonian
from thermofrag.potentials.qm import QMHead
from thermofrag.sampling.fragment_mh import FragmentMHStats, FragmentNodeFlipMH
from thermofrag.training.losses import (
    coupling_pcd_loss,
    mu_calibration_loss,
    qm_loss,
)
from thermofrag.training.pcd import PCDBuffer
from thermofrag.training.thermo_int import free_energy_gradient


_DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def build_hamiltonian(cfg: dict) -> Hamiltonian:
    """Assemble the full Hamiltonian from config, with optional warm-start of QM
    and Coupling heads from independent Phase-1 / Phase-2 checkpoints.
    """
    qm = build_qm_head(cfg)

    mc = cfg["model"]["coupling"]
    coupling = CouplingPotential(
        n_fragments=int(mc.get("n_fragments", 1024)),
        n_bond_types=int(mc.get("n_bond_types", 8)),
        hidden=int(mc["hidden"]),
        num_layers=int(mc["num_layers"]),
    )
    cws = mc.get("warm_start")
    if cws:
        coupling.load_state_dict(torch.load(cws, map_location="cpu"), strict=False)

    me = cfg["model"]["external_field"]
    mu = ChemicalPotentialHead(n_properties=len(me["properties"]), hidden=int(me["hidden"]))

    return Hamiltonian(qm, coupling, mu)


def build_qm_head(cfg: dict) -> QMHead:
    mq = cfg["model"]["qm"]
    pcfg = PaiNNConfig(
        hidden=int(mq["hidden"]),
        num_layers=int(mq["num_layers"]),
        cutoff=float(mq["cutoff"]),
        n_radial=int(mq["n_radial"]),
        elements=tuple(mq.get("elements", ("H", "C", "N", "O", "F", "P", "S", "Cl", "Br"))),
    )
    model = QMHead(pcfg)
    ws = mq.get("warm_start")
    if ws:
        sd = torch.load(ws, map_location="cpu")
        model.load_state_dict(sd, strict=False)
    return model


class Trainer:
    """Single-GPU trainer with bf16 autocast, grad clipping, JSONL logging."""

    def __init__(
        self,
        cfg: dict,
        model: torch.nn.Module,
        train_loader,
        val_loader=None,
        device: str | None = None,
    ):
        self.cfg = cfg
        device = device or cfg["run"].get("device", "cuda")
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.step = 0
        self.optimizer = self._build_optimizer()
        self.precision = cfg["run"].get("precision", "fp32")
        self.autocast_dtype = _DTYPE.get(self.precision, torch.float32)

        self.out_root = Path(cfg["run"]["out_dir"])
        self.log_dir = self.out_root / "logs" / cfg["run"]["name"]
        self.ckpt_dir = self.out_root / "checkpoints"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.log_dir / "metrics.jsonl"

    def _build_optimizer(self):
        t = self.cfg["training"]
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=float(t["lr"]),
            weight_decay=float(t["weight_decay"]),
        )

    def _autocast(self):
        if self.device == "cuda" and self.autocast_dtype != torch.float32:
            return torch.autocast("cuda", dtype=self.autocast_dtype)
        return nullcontext()

    def fit(self, max_steps: int | None = None) -> dict:
        phase = self.cfg["training"]["phase"]
        if phase == "pretrain_qm":
            return self._fit_qm(max_steps=max_steps)
        if phase == "pretrain_coupling":
            return self._fit_coupling(max_steps=max_steps)
        if phase == "joint":
            return self._fit_joint(max_steps=max_steps)
        if phase == "joint_conditional":
            return self._fit_joint_conditional(max_steps=max_steps)
        raise ValueError(f"Unknown training phase: {phase}")

    # -- QM pretrain -----------------------------------------------------

    def _fit_qm(self, max_steps: int | None = None) -> dict:
        t = self.cfg["training"]
        epochs = int(t["epochs"])
        log_every = int(t["log_every"])
        ckpt_every = int(t["ckpt_every"])
        eval_every = int(t.get("eval_every", 0))  # 0 = legacy once-at-end behavior
        grad_clip = float(t.get("grad_clip", 5.0))
        alpha_force = float(t["loss_weights"].get("qm_force", 0.5))

        self.model.train()
        t0 = time.time()
        loss_ema = None
        best_val = math.inf

        # If the dataset carries a force_std scale, normalize the predicted force
        # the same way the target was normalized. Our model predicts E in
        # normalized-energy units; its gradient wrt pos is therefore
        # F_pred_norm = -grad_x E_pred = F_true_phys / energy_std. To make that
        # comparable to batch.forces_norm = F_true_phys / force_std we multiply
        # by energy_std / force_std.
        energy_std = getattr(self.train_loader.dataset, "energy_std", 1.0)
        force_std = getattr(self.train_loader.dataset, "force_std", 1.0)
        f_scale = float(energy_std / max(force_std, 1e-9))

        def loop():
            nonlocal loss_ema, best_val
            for epoch in range(epochs):
                for batch in self.train_loader:
                    batch = batch.to(self.device)
                    with self._autocast():
                        E_pred, F_pred = self.model(batch, return_forces=True)
                        E_true = batch.energy_norm if hasattr(batch, "energy_norm") else batch.energy
                        F_true = batch.forces_norm if hasattr(batch, "forces_norm") else batch.forces
                        loss = qm_loss(
                            E_pred.float(), E_true.float(),
                            (F_pred * f_scale).float(), F_true.float(),
                            alpha_force=alpha_force,
                        )
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.optimizer.step()

                    loss_val = loss.item()
                    loss_ema = loss_val if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_val
                    self.step += 1

                    if self.step % log_every == 0:
                        self._log({
                            "phase": "qm",
                            "epoch": epoch,
                            "loss": loss_val,
                            "loss_ema": loss_ema,
                            "lr": self.optimizer.param_groups[0]["lr"],
                            "samples_per_sec": (self.step * self.cfg["training"]["batch_size"]) / max(time.time() - t0, 1e-6),
                        })
                    if self.step % ckpt_every == 0:
                        self._save_ckpt("qm_last")
                    if (
                        eval_every > 0
                        and self.val_loader is not None
                        and self.step % eval_every == 0
                    ):
                        val_metrics = self._eval_qm(alpha_force=alpha_force)
                        self._log({"phase": "qm", "eval": val_metrics, "best_val": best_val})
                        if val_metrics["loss_phys"] < best_val:
                            best_val = val_metrics["loss_phys"]
                            self._save_ckpt("qm_best")
                    if max_steps is not None and self.step >= max_steps:
                        return

        loop()

        if self.val_loader is not None:
            val_metrics = self._eval_qm(alpha_force=alpha_force)
            self._log({"phase": "qm", "eval": val_metrics, "best_val": best_val})
            if val_metrics["loss_phys"] < best_val:
                best_val = val_metrics["loss_phys"]
                self._save_ckpt("qm_best")

        self._save_ckpt("qm_final")
        return {"step": self.step, "loss_ema": loss_ema, "best_val": best_val}

    @torch.no_grad()
    def _eval_qm(self, alpha_force: float) -> dict:
        """Val eval in physical units (kcal/mol, kcal/mol/Å)."""
        self.model.eval()
        energy_mean = getattr(self.val_loader.dataset, "energy_mean", 0.0)
        energy_std = getattr(self.val_loader.dataset, "energy_std", 1.0)

        se_e, se_f, n, n_atoms = 0.0, 0.0, 0, 0
        for batch in self.val_loader:
            batch = batch.to(self.device)
            with torch.enable_grad():
                E_pred_norm, F_pred_norm = self.model(batch, return_forces=True)
            # de-normalize to kcal/mol, kcal/mol/Å
            E_pred = E_pred_norm.detach().float() * energy_std + energy_mean
            F_pred = F_pred_norm.detach().float() * energy_std  # F = -grad E; E scaled by std -> F scaled by std
            E_true = batch.energy.float()
            F_true = batch.forces.float()
            se_e += (E_pred - E_true).pow(2).sum().item()
            se_f += (F_pred - F_true).pow(2).sum().item()
            n += E_pred.shape[0]
            n_atoms += int(batch.z.shape[0])
        self.model.train()
        mse_e = se_e / max(n, 1)
        mse_f = se_f / max(3 * n_atoms, 1)  # per-component
        return {
            "loss_phys": mse_e + alpha_force * mse_f,
            "energy_rmse_kcal": math.sqrt(mse_e),
            "energy_mae_rough_kcal": math.sqrt(mse_e),  # placeholder; full MAE via eval_qm.py
            "force_rmse_kcal_per_A": math.sqrt(mse_f),
            "n": n,
            "n_atoms": n_atoms,
        }

    # -- Coupling PCD ----------------------------------------------------

    def _fit_coupling(self, max_steps: int | None = None) -> dict:
        """Phase 2: train CouplingPotential V with persistent contrastive divergence.

        Requires on ``self``:
          * ``self.pcd_buffer``      PCDBuffer already seeded from the data pool
          * ``self.data_pool``       Sequence[Data] of fragment graphs used to
                                     sample the positive batch and reseed a
                                     small slice of the buffer each step.
          * ``self.mh_kernel``       FragmentNodeFlipMH or compatible.
          * ``self.model``           CouplingPotential instance.
        """
        t = self.cfg["training"]
        batch_size = int(t["batch_size"])
        log_every = int(t["log_every"])
        ckpt_every = int(t["ckpt_every"])
        epochs = int(t["epochs"])
        grad_clip = float(t.get("grad_clip", 5.0))
        l2_reg = float(t["loss_weights"].get("couple_l2", 1e-3))
        mh_steps = int(t.get("mh_steps", 5))

        self.model.train()
        t0 = time.time()
        loss_ema = None
        mh_stats = FragmentMHStats()

        def loop():
            nonlocal loss_ema
            steps_per_epoch = max(1, len(self.data_pool) // batch_size)
            for epoch in range(epochs):
                for _ in range(steps_per_epoch):
                    # --- positive batch from data
                    idxs = [self._coupling_rng.randrange(len(self.data_pool)) for _ in range(batch_size)]
                    from torch_geometric.data import Batch

                    data_batch = Batch.from_data_list([self.data_pool[i] for i in idxs]).to(self.device)

                    # --- negative batch from buffer, evolved with MH
                    buf_batch, slot_idxs = self.pcd_buffer.sample_batch(batch_size)
                    buf_batch = buf_batch.to(self.device)
                    for _ in range(mh_steps):
                        buf_batch = self.mh_kernel.step(buf_batch, stats=mh_stats)

                    # --- loss
                    with self._autocast():
                        v_pos = self.model(data_batch).float()
                        v_neg = self.model(buf_batch).float()
                        L_contrast = coupling_pcd_loss(v_pos, v_neg)
                        L_reg = l2_reg * (v_pos.pow(2).mean() + v_neg.pow(2).mean())
                        loss = L_contrast + L_reg

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.optimizer.step()

                    # --- write back evolved chains into buffer, with refresh
                    self.pcd_buffer.update_from_batch(buf_batch, slot_idxs, data_pool=self.data_pool)

                    loss_val = loss.item()
                    loss_ema = loss_val if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_val
                    self.step += 1

                    if self.step % log_every == 0:
                        self._log({
                            "phase": "coupling",
                            "epoch": epoch,
                            "loss": loss_val,
                            "loss_contrast": L_contrast.item(),
                            "loss_reg": L_reg.item(),
                            "v_pos_mean": v_pos.mean().item(),
                            "v_neg_mean": v_neg.mean().item(),
                            "mh_accept_rate": mh_stats.accept_rate,
                            "lr": self.optimizer.param_groups[0]["lr"],
                            "loss_ema": loss_ema,
                            "samples_per_sec": (self.step * batch_size) / max(time.time() - t0, 1e-6),
                        })
                    if self.step % ckpt_every == 0:
                        self._save_ckpt("coupling_last")
                    if max_steps is not None and self.step >= max_steps:
                        return

        # Seed a dedicated RNG so data-pool sampling is independent of PCDBuffer's.
        import random as _rnd
        self._coupling_rng = _rnd.Random(self.cfg["run"].get("seed", 0) + 1)
        loop()
        self._save_ckpt("coupling_final")
        return {"step": self.step, "loss_ema": loss_ema, "mh_accept_rate": mh_stats.accept_rate}

    def attach_pcd(self, buffer: PCDBuffer, data_pool, mh_kernel: FragmentNodeFlipMH) -> None:
        """Register the PCD dependencies before calling ``fit()`` with phase=pretrain_coupling."""
        self.pcd_buffer = buffer
        self.data_pool = data_pool
        self.mh_kernel = mh_kernel

    # -- Phase 3: conditional joint (QM frozen; coupling + mu trained) --

    def _fit_joint_conditional(self, max_steps: int | None = None) -> dict:
        """Phase 3 conditional: train CouplingPotential + ChemicalPotentialHead jointly.

        QM head is treated as frozen / absent in this phase (MILESTONES risk
        register: "freeze QM head entirely" is the recommended mitigation).
        Data comes from the conditional LMDB (phi pre-cached per molecule).

        ``self.model`` must expose ``.coupling`` and ``.mu`` submodules. A
        wrapper ``CouplingMuModule`` is provided by the scripts/train.py caller.

        Requires on ``self``:
          * ``self.pcd_buffer`` PCDBuffer seeded from conditional data pool
          * ``self.data_pool``  Sequence[Data] exposing ``.phi`` per sample
          * ``self.mh_kernel``  FragmentNodeFlipMH (coupling-only acceptance)
          * ``self.phi_mean``   [K] tensor used to standardize phi
          * ``self.phi_std``    [K] tensor

        Loss = L_couple + lam_reg * (V^2) + lam_mu * L_mu,
        L_mu = MSE(mu(y), beta * phi_std(m_data)) where
        y = phi_std(m_data) + sigma * N(0, I).
        """
        import random as _rnd
        from torch_geometric.data import Batch

        t = self.cfg["training"]
        batch_size = int(t["batch_size"])
        log_every = int(t["log_every"])
        ckpt_every = int(t["ckpt_every"])
        epochs = int(t["epochs"])
        grad_clip = float(t.get("grad_clip", 5.0))
        lw = t["loss_weights"]
        lam_couple = float(lw.get("couple", 1.0))
        lam_mu = float(lw.get("mu", 0.1))
        lam_reg = float(lw.get("couple_l2", 1e-3))
        mh_steps = int(t.get("mh_steps", 3))
        beta_mu = float(t.get("mu_beta", 1.0))
        y_noise = float(t.get("mu_y_noise", 0.3))

        self.model.train()
        t0 = time.time()
        loss_ema = None
        mh_stats = FragmentMHStats()

        phi_mean = self.phi_mean.to(self.device)
        phi_std = self.phi_std.to(self.device)

        self._joint_rng = _rnd.Random(self.cfg["run"].get("seed", 0) + 2)

        def loop():
            nonlocal loss_ema
            steps_per_epoch = max(1, len(self.data_pool) // batch_size)
            for epoch in range(epochs):
                for _ in range(steps_per_epoch):
                    idxs = [self._joint_rng.randrange(len(self.data_pool)) for _ in range(batch_size)]
                    data_list = [self.data_pool[i] for i in idxs]
                    data_batch = Batch.from_data_list(data_list).to(self.device)
                    # Per-molecule raw phi, standardized.
                    phi_raw = torch.stack([d.phi for d in data_list], dim=0).to(self.device)  # [B, K]
                    phi_z = (phi_raw - phi_mean) / phi_std  # [B, K]
                    # y = phi_z + noise; this is the conditioning target the sampler sees.
                    y = phi_z + y_noise * torch.randn_like(phi_z)

                    buf_batch, slot_idxs = self.pcd_buffer.sample_batch(batch_size)
                    buf_batch = buf_batch.to(self.device)
                    for _ in range(mh_steps):
                        buf_batch = self.mh_kernel.step(buf_batch, stats=mh_stats)

                    with self._autocast():
                        v_pos = self.model.coupling(data_batch).float()
                        v_neg = self.model.coupling(buf_batch).float()
                        mu_pred = self.model.mu(y).float()  # [B, K]
                        mu_target = beta_mu * phi_z.float()  # [B, K]

                        L_contrast = coupling_pcd_loss(v_pos, v_neg)
                        L_reg = lam_reg * (v_pos.pow(2).mean() + v_neg.pow(2).mean())
                        L_mu_val = mu_calibration_loss(mu_pred, mu_target)
                        loss = lam_couple * L_contrast + L_reg + lam_mu * L_mu_val

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.optimizer.step()

                    self.pcd_buffer.update_from_batch(buf_batch, slot_idxs, data_pool=self.data_pool)

                    loss_val = loss.item()
                    loss_ema = loss_val if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_val
                    self.step += 1

                    if self.step % log_every == 0:
                        # mu(y=0) vector: the baseline "no-condition" chemical potential
                        with torch.no_grad():
                            mu_at_zero = self.model.mu(torch.zeros(1, y.shape[1], device=self.device))[0]
                        self._log({
                            "phase": "joint_conditional",
                            "epoch": epoch,
                            "loss": loss_val,
                            "L_couple": L_contrast.item(),
                            "L_mu": L_mu_val.item(),
                            "L_reg": L_reg.item(),
                            "v_pos_mean": v_pos.mean().item(),
                            "v_neg_mean": v_neg.mean().item(),
                            "mu_at_zero_norm": float(mu_at_zero.norm().item()),
                            "mh_accept_rate": mh_stats.accept_rate,
                            "loss_ema": loss_ema,
                            "lr": self.optimizer.param_groups[0]["lr"],
                            "samples_per_sec": (self.step * batch_size) / max(time.time() - t0, 1e-6),
                        })
                    if self.step % ckpt_every == 0:
                        self._save_ckpt("joint_last")
                    if max_steps is not None and self.step >= max_steps:
                        return

        loop()
        self._save_ckpt("joint_final")
        # Fit Laplace posterior on the mu head using the data-distribution y's.
        # Draw a bounded number of batches worth of phi to fit the diagonal.
        try:
            n_y = min(4096, len(self.data_pool))
            pool_idxs = self._joint_rng.sample(range(len(self.data_pool)), k=n_y)
            phi_stack = torch.stack([self.data_pool[i].phi for i in pool_idxs], dim=0).to(self.device)
            phi_z_stack = (phi_stack - phi_mean) / phi_std
            from thermofrag.potentials.external_field import make_laplace_y_iter
            self.model.mu.fit_laplace(make_laplace_y_iter(phi_z_stack, batch_size=256))
            self._save_ckpt("joint_final")  # overwrite with Laplace-fitted buffer
            self._log({"phase": "joint_conditional", "laplace_fitted": True})
        except Exception as e:  # pragma: no cover - non-fatal
            self._log({"phase": "joint_conditional", "laplace_fit_error": str(e)})

        return {"step": self.step, "loss_ema": loss_ema, "mh_accept_rate": mh_stats.accept_rate}

    def attach_joint_conditional(
        self,
        buffer: PCDBuffer,
        data_pool,
        mh_kernel: FragmentNodeFlipMH,
        phi_mean: torch.Tensor,
        phi_std: torch.Tensor,
    ) -> None:
        self.pcd_buffer = buffer
        self.data_pool = data_pool
        self.mh_kernel = mh_kernel
        self.phi_mean = phi_mean
        self.phi_std = phi_std

    # -- Joint Hamiltonian fine-tune -------------------------------------

    def _fit_joint(self, max_steps: int | None = None) -> dict:
        """Phase 3: jointly fine-tune the full Hamiltonian H = E^QM + V^couple - μ(y)·φ.

        Requires on ``self``:
          * ``self.model``        Hamiltonian instance (wraps qm, coupling, mu).
          * ``self.train_loader`` iterable yielding conditional batches. Each batch
            exposes attributes: atomic (Batch), fragment (Batch), y [B, K],
            phi [B, K], energy_norm [B], forces [N, 3].
          * ``self.pcd_buffer``   PCD buffer of fragment graphs (for coupling negatives).
          * ``self.mh_kernel``    Fragment MH kernel scoring V(m).
          * ``self.data_pool``    Sequence[Data] fragment pool (for refresh).

        Losses = L_QM + λ1 L_couple + λ2 L_μ + λ3 L_reg(V)
        """
        t = self.cfg["training"]
        epochs = int(t["epochs"])
        log_every = int(t["log_every"])
        ckpt_every = int(t["ckpt_every"])
        grad_clip = float(t.get("grad_clip", 5.0))
        alpha_force = float(t["loss_weights"].get("qm_force", 0.5))
        lam_couple = float(t["loss_weights"].get("couple", 0.1))
        lam_mu = float(t["loss_weights"].get("mu", 0.1))
        lam_reg = float(t["loss_weights"].get("couple_l2", 1e-3))
        mh_steps = int(t.get("mh_steps", 3))
        beta_sample = float(self.cfg["sampling"]["beta_schedule"].get("beta_T", 1.0))

        self.model.train()
        t0 = time.time()
        loss_ema = None
        mh_stats = FragmentMHStats()

        def loop():
            nonlocal loss_ema
            for epoch in range(epochs):
                for batch in self.train_loader:
                    atomic = batch.atomic.to(self.device)
                    fragment = batch.fragment.to(self.device)
                    y = batch.y.to(self.device)
                    phi = batch.phi.to(self.device)
                    E_true = batch.energy_norm.to(self.device)
                    F_true = batch.forces.to(self.device)

                    # --- evolve PCD buffer
                    buf_batch, slot_idxs = self.pcd_buffer.sample_batch(y.shape[0])
                    buf_batch = buf_batch.to(self.device)
                    for _ in range(mh_steps):
                        buf_batch = self.mh_kernel.step(buf_batch, stats=mh_stats)

                    # --- forward
                    atomic.pos = atomic.pos.detach().clone().requires_grad_(True)
                    with self._autocast():
                        # QM energy + forces (per-atom E summed per molecule)
                        scalar, _ = self.model.qm.backbone(atomic)
                        atom_e = self.model.qm.energy_mlp(scalar).squeeze(-1) + self.model.qm.atom_ref(atomic.z).squeeze(-1)
                        E_pred = scatter_add(atom_e, atomic.batch, dim=0)

                    (grad_pos,) = torch.autograd.grad(E_pred.sum(), atomic.pos, create_graph=True)
                    F_pred = -grad_pos

                    with self._autocast():
                        # Coupling: on the molecule's fragment graph (pos) and on buffer (neg)
                        v_pos = self.model.coupling(fragment).float()
                        v_neg = self.model.coupling(buf_batch).float()
                        # μ calibration via finite-difference thermodynamic integration
                        fe_grad = free_energy_gradient(phi, beta=beta_sample)  # [K]
                        mu_pred = self.model.mu(y)  # [B, K]
                        mu_target = fe_grad.unsqueeze(0).expand_as(mu_pred)

                        L_qm = qm_loss(E_pred.float(), E_true.float(), F_pred.float(), F_true.float(), alpha_force=alpha_force)
                        L_couple = coupling_pcd_loss(v_pos, v_neg)
                        L_mu = mu_calibration_loss(mu_pred.float(), mu_target.float())
                        L_reg = lam_reg * (v_pos.pow(2).mean() + v_neg.pow(2).mean())
                        loss = L_qm + lam_couple * L_couple + lam_mu * L_mu + L_reg

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.optimizer.step()

                    self.pcd_buffer.update_from_batch(buf_batch, slot_idxs, data_pool=self.data_pool)

                    loss_val = loss.item()
                    loss_ema = loss_val if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_val
                    self.step += 1

                    if self.step % log_every == 0:
                        self._log({
                            "phase": "joint",
                            "epoch": epoch,
                            "loss": loss_val,
                            "L_qm": L_qm.item(),
                            "L_couple": L_couple.item(),
                            "L_mu": L_mu.item(),
                            "L_reg": L_reg.item(),
                            "v_pos_mean": v_pos.mean().item(),
                            "v_neg_mean": v_neg.mean().item(),
                            "mh_accept_rate": mh_stats.accept_rate,
                            "loss_ema": loss_ema,
                            "lr": self.optimizer.param_groups[0]["lr"],
                            "samples_per_sec": (self.step * y.shape[0]) / max(time.time() - t0, 1e-6),
                        })
                    if self.step % ckpt_every == 0:
                        self._save_ckpt("joint_last")
                    if max_steps is not None and self.step >= max_steps:
                        return

        loop()
        self._save_ckpt("joint_final")
        return {"step": self.step, "loss_ema": loss_ema, "mh_accept_rate": mh_stats.accept_rate}

    # -- io --------------------------------------------------------------

    def _save_ckpt(self, tag: str) -> Path:
        suffix = str(self.cfg["run"].get("ckpt_suffix", "") or "")
        name = f"{tag}_{suffix}.pt" if suffix else f"{tag}.pt"
        p = self.ckpt_dir / name
        torch.save({"step": self.step, "state_dict": self.model.state_dict(), "cfg": self.cfg}, p)
        return p

    def _log(self, metrics: dict) -> None:
        with self.metrics_path.open("a") as f:
            f.write(json.dumps({"step": self.step, **metrics}) + "\n")


def build_spice_loaders(cfg: dict) -> tuple[DataLoader, DataLoader | None]:
    """Build train / optional val DataLoaders from preprocessed SPICE shards.

    ``cfg.data.qm_train`` may point at either a flat shard directory or a
    preprocess root containing ``train/`` + ``val/`` subdirs. In the latter
    case, and when ``cfg.data.qm_val`` is not set, this auto-discovers the
    sibling val split. Energy + force normalization stats come from the
    train-split manifest and are applied to both splits so predictions are
    directly comparable.
    """
    data_cfg = cfg["data"]
    train_root = Path(data_cfg["qm_train"])
    if (train_root / "train").is_dir() and any((train_root / "train").glob("shard_*.npz")):
        train_dir = train_root / "train"
        implicit_val = train_root / "val"
    else:
        train_dir = train_root
        implicit_val = None

    stats = _read_energy_stats(train_dir)
    shard_kwargs = dict(
        energy_mean=stats["mean"], energy_std=stats["std"], force_std=stats.get("force_std", 1.0)
    )
    train_ds = SPICEShard(train_dir, **shard_kwargs)

    val_loader = None
    val_path = data_cfg.get("qm_val")
    val_dir = Path(val_path) if val_path else implicit_val
    if val_dir and val_dir.is_dir() and any(val_dir.glob("shard_*.npz")):
        val_ds = SPICEShard(val_dir, **shard_kwargs)
        val_loader = DataLoader(
            val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
            num_workers=int(cfg["training"].get("num_workers", 0)),
            persistent_workers=cfg["training"].get("num_workers", 0) > 0,
        )

    nw = int(cfg["training"].get("num_workers", 4))
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=nw,
        persistent_workers=nw > 0,
        pin_memory=cfg["run"].get("device", "cuda") == "cuda",
    )
    return train_loader, val_loader


def _read_energy_stats(shard_dir: Path) -> dict:
    mf = Path(shard_dir) / "manifest.json"
    if mf.is_file():
        meta = json.loads(mf.read_text())
        es = meta.get("energy_stats", {"mean": 0.0, "std": 1.0, "force_std": 1.0})
        return {
            "mean": float(es.get("mean", 0.0)),
            "std": float(es.get("std", 1.0)),
            "force_std": float(es.get("force_std", 1.0)),
        }
    return {"mean": 0.0, "std": 1.0, "force_std": 1.0}
