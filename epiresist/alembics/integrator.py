from __future__ import annotations

from collections.abc import Callable

import torch
from torchdiffeq import odeint, odeint_adjoint

RhsFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def integrate(
    func: RhsFn,
    y0: torch.Tensor,
    times: torch.Tensor,
    method: str = "dopri5",
    rtol: float = 1e-5,
    atol: float = 1e-7,
    adjoint: bool = True,
) -> torch.Tensor:
    solver = odeint_adjoint if (adjoint and method == "dopri5") else odeint
    options = None
    if method == "rk4":
        step = float((times[-1] - times[0]) / max(len(times) - 1, 1))
        options = {"step_size": max(step, 1e-3)}
    trajectory = solver(func, y0, times, method=method, rtol=rtol, atol=atol, options=options)
    return trajectory


def physics_residual(func: RhsFn, trajectory: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    if trajectory.shape[0] < 2:
        return trajectory.new_zeros(())
    deltas = trajectory[1:] - trajectory[:-1]
    dt = (times[1:] - times[:-1]).reshape(-1, 1, 1)
    empirical = deltas / (dt + 1e-9)
    midpoints = 0.5 * (trajectory[1:] + trajectory[:-1])
    mid_times = 0.5 * (times[1:] + times[:-1])
    analytic = torch.stack([func(mid_times[i], midpoints[i]) for i in range(midpoints.shape[0])])
    return ((empirical - analytic) ** 2).mean()
