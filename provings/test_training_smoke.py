from __future__ import annotations

from torch.utils.data import Subset

from epiresist.dispensary import Regimen


def test_two_step_training_runs(model, cohort, config):
    losses = Regimen(config).fit_steps(model, cohort, 2)
    assert len(losses) == 2
    assert all(loss == loss for loss in losses)


def test_short_run_reduces_loss(model, cohort, config):
    single = Subset(cohort, list(range(8)))
    losses = Regimen(config).fit_steps(model, single, 60)
    assert min(losses) < losses[0]


def test_full_fit_records_history(model, cohort, config):
    split = max(int(0.8 * len(cohort)), 1)
    train = Subset(cohort, list(range(split)))
    val = Subset(cohort, list(range(split, len(cohort))))
    history = Regimen(config).fit(model, train, val)
    assert len(history["train"]) >= 1
    assert len(history["val"]) == len(history["train"])
