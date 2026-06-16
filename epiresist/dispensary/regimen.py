from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from epiresist.apparatus import atomic_save, get_logger, resolve_device, seed_all
from epiresist.balances import LossWeights, total_loss
from epiresist.formulary import EpiResistPINN, FormularyConfig
from epiresist.materia import CohortBatch, collate

Cohort = Dataset[dict[str, torch.Tensor]]

LOGGER = get_logger("epiresist.regimen")


class Regimen:
    def __init__(self, config: FormularyConfig) -> None:
        self.config = config
        self.device = resolve_device(config.regimen.device)
        self.weights = LossWeights(
            config.balance.lambda_physics,
            config.balance.lambda_sparse,
            config.balance.lambda_temporal,
        )

    def _loader(self, cohort: Cohort, shuffle: bool) -> DataLoader:
        return DataLoader(
            cohort,
            batch_size=self.config.regimen.batch_size,
            shuffle=shuffle,
            collate_fn=collate,
        )

    def _step(
        self, model: EpiResistPINN, batch: CohortBatch, optimizer: torch.optim.Optimizer
    ) -> dict[str, float]:
        batch = batch.to(self.device)
        output = model(batch)
        loss, components = total_loss(
            output.regression,
            batch.target,
            output.logits,
            batch.label,
            output.physics,
            output.attention_mass,
            output.trajectory,
            self.weights,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.regimen.grad_clip)
        optimizer.step()
        return components

    def fit_steps(self, model: EpiResistPINN, cohort: Cohort, n_steps: int) -> list[float]:
        seed_all(self.config.regimen.seed)
        model.to(self.device).train()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.regimen.lr,
            weight_decay=self.config.regimen.weight_decay,
        )
        losses: list[float] = []
        loader = self._loader(cohort, shuffle=True)
        iterator = iter(loader)
        for _ in range(n_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            components = self._step(model, batch, optimizer)
            losses.append(components["total"])
        return losses

    def fit(
        self,
        model: EpiResistPINN,
        train_cohort: Cohort,
        val_cohort: Cohort,
        checkpoint: str | Path | None = None,
    ) -> dict[str, list[float]]:
        seed_all(self.config.regimen.seed)
        model.to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.regimen.lr,
            weight_decay=self.config.regimen.weight_decay,
        )
        ema = copy.deepcopy(model.state_dict())
        history: dict[str, list[float]] = {"train": [], "val": []}
        best = float("inf")
        stale = 0
        train_loader = self._loader(train_cohort, shuffle=True)
        for epoch in range(self.config.regimen.epochs):
            model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                components = self._step(model, batch, optimizer)
                epoch_loss += components["total"]
                self._update_ema(model, ema)
            epoch_loss /= max(len(train_loader), 1)
            val_loss = self.evaluate_loss(model, val_cohort)
            history["train"].append(epoch_loss)
            history["val"].append(val_loss)
            LOGGER.info("epoch %d train %.4f val %.4f", epoch, epoch_loss, val_loss)
            if val_loss < best - 1e-5:
                best = val_loss
                stale = 0
                if checkpoint is not None:
                    atomic_save(
                        {
                            "model": model.state_dict(),
                            "ema": ema,
                            "seed": self.config.regimen.seed,
                            "epoch": epoch,
                        },
                        checkpoint,
                    )
            else:
                stale += 1
                if stale >= self.config.regimen.patience:
                    break
        return history

    def _update_ema(self, model: EpiResistPINN, ema: dict[str, torch.Tensor]) -> None:
        decay = self.config.regimen.ema_decay
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                ema[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema[key] = value.detach().clone()

    @torch.no_grad()
    def evaluate_loss(self, model: EpiResistPINN, cohort: Cohort) -> float:
        model.eval()
        loader = self._loader(cohort, shuffle=False)
        total = 0.0
        count = 0
        for batch in loader:
            batch = batch.to(self.device)
            output = model(batch)
            loss, _ = total_loss(
                output.regression,
                batch.target,
                output.logits,
                batch.label,
                output.physics,
                output.attention_mass,
                output.trajectory,
                self.weights,
            )
            total += float(loss)
            count += 1
        return total / max(count, 1)
