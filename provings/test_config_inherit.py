from __future__ import annotations

from epiresist.formulary import load_formula


def test_smoke_overrides_dataset_shape():
    config = load_formula("_smoke")
    assert config.materia.n_genes == 64
    assert config.alembic.solver == "dopri5"
    assert config.preparation.d_model == 32


def test_ablation_inherits_main_and_overrides_flag():
    config = load_formula("ablation_no_coupling")
    assert config.alembic.use_coupling is False
    assert config.alembic.enable_epigenetic is True
    assert config.regimen.lr == 0.0005


def test_dataset_supplement_overrides_materia():
    config = load_formula("supplement_gdsc")
    assert config.materia.name == "gdsc_aml"
    assert config.materia.n_concentrations == 9
    assert config.balance.lambda_physics == 0.1


def test_modality_ablation_flags():
    config = load_formula("ablation_no_methylation")
    assert config.preparation.active_modalities == (True, False, True)
