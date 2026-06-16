from __future__ import annotations

import torch
from torch import nn


class PhysicsGate(nn.Module):
    def __init__(self, d_model: int, n_modalities: int = 3, n_state: int = 3) -> None:
        super().__init__()
        self.n_modalities = n_modalities
        self.n_state = n_state
        self.gate = nn.Linear(n_state + d_model, d_model)
        self.sparsity_weight: torch.Tensor

    def forward(
        self, attended: torch.Tensor, ode_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = attended.shape[0]
        state = ode_state.unsqueeze(1).expand(-1, self.n_modalities, -1)
        gate_input = torch.cat([state, attended], dim=-1)
        weights = torch.sigmoid(self.gate(gate_input))
        fused = (weights * attended).sum(dim=1)
        attention_mass = weights.reshape(batch, -1)
        return fused, attention_mass

    def zero_state(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, self.n_state, device=device)
