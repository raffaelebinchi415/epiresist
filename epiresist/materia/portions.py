from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CohortBatch:
    expr: torch.Tensor
    methyl: torch.Tensor
    drug: torch.Tensor
    dose: torch.Tensor
    target: torch.Tensor
    label: torch.Tensor

    def to(self, device: torch.device) -> CohortBatch:
        return CohortBatch(
            expr=self.expr.to(device),
            methyl=self.methyl.to(device),
            drug=self.drug.to(device),
            dose=self.dose.to(device),
            target=self.target.to(device),
            label=self.label.to(device),
        )

    def __len__(self) -> int:
        return int(self.expr.shape[0])


def collate(samples: list[dict[str, torch.Tensor]]) -> CohortBatch:
    return CohortBatch(
        expr=torch.stack([s["expr"] for s in samples]),
        methyl=torch.stack([s["methyl"] for s in samples]),
        drug=torch.stack([s["drug"] for s in samples]),
        dose=torch.stack([s["dose"] for s in samples]),
        target=torch.stack([s["target"] for s in samples]),
        label=torch.stack([s["label"] for s in samples]),
    )
