from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import typer

from epiresist.apparatus import get_logger, load_checkpoint, resolve_device, seed_all
from epiresist.dispensary import Regimen, evaluate, gather_predictions
from epiresist.formulary import EncoderPathway, FormularyConfig, build_model, load_formula
from epiresist.materia import CohortSpec, SyntheticCohort

app = typer.Typer(add_completion=False, help="EpiResist-PINN dispensing counter")
LOGGER = get_logger("epiresist.counter")


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


def _load(formula: str, recipes: str) -> FormularyConfig:
    return load_formula(formula, recipes)


def _maybe_restore(model: torch.nn.Module, checkpoint: Path) -> None:
    if checkpoint.exists():
        payload = load_checkpoint(checkpoint)
        model.load_state_dict(payload["model"])
        LOGGER.info("restored weights from %s", checkpoint)


@app.command()
def triturate(
    formula: str = typer.Option("main"),
    recipes: str = typer.Option("recipes/formula"),
    out: str = typer.Option("data"),
) -> None:
    config = _load(formula, recipes)
    cohort = SyntheticCohort(_spec(config))
    fields: dict[str, list[np.ndarray]] = {
        "expression": [],
        "methylation": [],
        "doseresponse": [],
        "dose": [],
        "target": [],
        "label": [],
    }
    name_map = {
        "expression": "expr",
        "methylation": "methyl",
        "doseresponse": "drug",
        "dose": "dose",
        "target": "target",
        "label": "label",
    }
    for i in range(len(cohort)):
        sample = cohort[i]
        for file_key, tensor_key in name_map.items():
            fields[file_key].append(sample[tensor_key].numpy())
    root = Path(out) / config.materia.name
    root.mkdir(parents=True, exist_ok=True)
    for key, values in fields.items():
        np.save(root / f"{key}.npy", np.stack(values))
    LOGGER.info("triturated %d specimens to %s", len(cohort), root)


@app.command()
def compound(
    formula: str = typer.Option("main"),
    recipes: str = typer.Option("recipes/formula"),
    runs: str = typer.Option("runs"),
) -> None:
    config = _load(formula, recipes)
    seed_all(config.regimen.seed)
    cohort = SyntheticCohort(_spec(config))
    split = max(int(0.8 * len(cohort)), 1)
    train = torch.utils.data.Subset(cohort, list(range(split)))
    val = torch.utils.data.Subset(cohort, list(range(split, len(cohort))))
    model = build_model(config)
    checkpoint = Path(runs) / config.name / "best.pt"
    history = Regimen(config).fit(model, train, val, checkpoint)
    LOGGER.info("final train %.4f val %.4f", history["train"][-1], history["val"][-1])


@app.command()
def assay(
    formula: str = typer.Option("main"),
    recipes: str = typer.Option("recipes/formula"),
    runs: str = typer.Option("runs"),
) -> None:
    config = _load(formula, recipes)
    model = build_model(config)
    _maybe_restore(model, Path(runs) / config.name / "best.pt")
    cohort = SyntheticCohort(_spec(config))
    metrics = evaluate(model, cohort, config)
    typer.echo(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=2))


@app.command()
def dispense(
    formula: str = typer.Option("main"),
    recipes: str = typer.Option("recipes/formula"),
    runs: str = typer.Option("runs"),
    count: int = typer.Option(5),
) -> None:
    config = _load(formula, recipes)
    model = build_model(config)
    _maybe_restore(model, Path(runs) / config.name / "best.pt")
    cohort = SyntheticCohort(_spec(config))
    device = resolve_device(config.regimen.device)
    preds = gather_predictions(model, cohort, device, config.regimen.batch_size)
    for i in range(min(count, preds["regression"].shape[0])):
        typer.echo(
            f"specimen {i}: response={float(preds['regression'][i]):.4f} "
            f"resistance={float(preds['probability'][i]):.4f}"
        )


@app.command()
def decant(
    formula: str = typer.Option("main"),
    recipes: str = typer.Option("recipes/formula"),
    runs: str = typer.Option("runs"),
) -> None:
    config = _load(formula, recipes)
    model = build_model(config)
    _maybe_restore(model, Path(runs) / config.name / "best.pt")
    model.eval()
    pathway = EncoderPathway(model)
    m = config.materia
    dummy = (
        torch.randn(1, m.n_genes),
        torch.rand(1, m.n_cpg),
        torch.rand(1, m.n_replicates, m.n_concentrations),
        torch.rand(1),
    )
    destination = Path(runs) / config.name / "epiresist.onnx"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            pathway,
            dummy,
            str(destination),
            input_names=["expression", "methylation", "doseresponse", "dose"],
            output_names=["response", "resistance"],
            opset_version=17,
        )
        LOGGER.info("exported encoder pathway to %s", destination)
    except (RuntimeError, ValueError) as error:
        LOGGER.error("onnx export failed: %s", error)
        raise typer.Exit(code=1) from error


if __name__ == "__main__":
    app()
