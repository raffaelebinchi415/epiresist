from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from epiresist.apparatus import resolve_device
from epiresist.assay import (
    aggregate_enrichment,
    auroc,
    f1_score,
    gsea_prerank,
    parameter_attribution,
    pearson,
    rmse,
    spearman,
)
from epiresist.dispensary.regimen import Regimen
from epiresist.formulary import EpiResistPINN, FormularyConfig, build_model
from epiresist.materia import CohortSpec, SyntheticCohort, collate

Cohort = Dataset[dict[str, torch.Tensor]]


def _spec(config: FormularyConfig) -> CohortSpec:
    m = config.materia
    return CohortSpec(
        m.name,
        m.n_samples,
        m.n_genes,
        m.n_cpg,
        m.n_concentrations,
        m.n_replicates,
        m.n_gene_sets,
        m.seed,
    )


def _gene_sets(n_genes: int, n_sets: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    size = max(n_genes // max(n_sets, 1), 2)
    return [rng.choice(n_genes, size=size, replace=False) for _ in range(n_sets)]


@torch.no_grad()
def gather_predictions(
    model: EpiResistPINN, cohort: Cohort, device: torch.device, batch_size: int
) -> dict[str, torch.Tensor]:
    model.eval().to(device)
    loader = DataLoader(cohort, batch_size=batch_size, shuffle=False, collate_fn=collate)
    regression, target, probability, label = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        regression.append(output.regression.cpu())
        target.append(batch.target.cpu())
        probability.append(torch.sigmoid(output.logits).cpu())
        label.append(batch.label.cpu())
    return {
        "regression": torch.cat(regression),
        "target": torch.cat(target),
        "probability": torch.cat(probability),
        "label": torch.cat(label),
    }


def pathway_enrichment(
    model: EpiResistPINN, cohort: Cohort, device: torch.device, config: FormularyConfig
) -> float:
    loader = DataLoader(cohort, batch_size=64, shuffle=False, collate_fn=collate)
    batch = next(iter(loader)).to(device)
    attribution = parameter_attribution(model, batch, "mu")
    gene_sets = _gene_sets(config.materia.n_genes, config.assay.top_k, config.materia.seed)
    nes = gsea_prerank(attribution, gene_sets, config.assay.n_perm, config.materia.seed)
    return aggregate_enrichment(nes, config.assay.top_k)


def evaluate(model: EpiResistPINN, cohort: Cohort, config: FormularyConfig) -> dict[str, float]:
    device = resolve_device(config.regimen.device)
    preds = gather_predictions(model, cohort, device, config.regimen.batch_size)
    metrics = {
        "pcc": pearson(preds["regression"], preds["target"]),
        "spearman": spearman(preds["regression"], preds["target"]),
        "rmse": rmse(preds["regression"], preds["target"]),
        "auroc": auroc(preds["probability"], preds["label"]),
        "f1": f1_score(preds["probability"], preds["label"]),
    }
    if config.enable_mechanism:
        metrics["pathway_es"] = pathway_enrichment(model, cohort, device, config)
    return metrics


def kfold(config: FormularyConfig) -> dict[str, float]:
    cohort = SyntheticCohort(_spec(config))
    indices = np.arange(len(cohort))
    folds = np.array_split(indices, config.regimen.folds)
    scores: dict[str, list[float]] = {}
    for i in range(len(folds)):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
        model = build_model(config)
        regimen = Regimen(config)
        regimen.fit(
            model,
            Subset(cohort, train_idx.tolist()),
            Subset(cohort, val_idx.tolist()),
        )
        fold_metrics = evaluate(model, Subset(cohort, val_idx.tolist()), config)
        for key, value in fold_metrics.items():
            scores.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in scores.items()}


def cross_dataset_transfer(source: FormularyConfig, target: FormularyConfig) -> dict[str, float]:
    source_cohort = SyntheticCohort(_spec(source))
    target_cohort = SyntheticCohort(_spec(target))
    model = build_model(source)
    regimen = Regimen(source)
    split = int(0.8 * len(source_cohort))
    regimen.fit(
        model,
        Subset(source_cohort, list(range(split))),
        Subset(source_cohort, list(range(split, len(source_cohort)))),
    )
    return evaluate(model, target_cohort, target)
