from __future__ import annotations

import torch

from epiresist.alembics import STATE_M
from epiresist.formulary import build_model, load_formula


def test_trajectory_is_finite(model, batch):
    output = model(batch)
    assert torch.isfinite(output.trajectory).all()
    assert torch.isfinite(output.terminal_state).all()


def test_physics_residual_nonnegative_and_small(model, batch):
    output = model(batch)
    assert float(output.physics.detach()) >= 0.0
    assert float(output.physics.detach()) < 1.0


def test_initial_methylation_bounded(model, batch):
    output = model(batch)
    initial_methyl = output.trajectory[0, :, STATE_M]
    assert torch.all(initial_methyl >= 0.0)
    assert torch.all(initial_methyl <= 1.0)


def test_disabling_all_odes_bypasses_integration(batch):
    config = load_formula("ablation_no_all_odes")
    smoke = load_formula("_smoke")
    merged = config.__class__(
        name=config.name,
        enable_mechanism=smoke.enable_mechanism,
        materia=smoke.materia,
        preparation=smoke.preparation,
        alembic=config.alembic,
        balance=smoke.balance,
        regimen=smoke.regimen,
        assay=smoke.assay,
    )
    model = build_model(merged)
    output = model(batch)
    assert output.trajectory.shape[0] == 1
    assert float(output.physics.detach()) == 0.0
