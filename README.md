# EpiResist-PINN — Pharmacopoeia Monograph

```
MONOGRAPH ........ epiresist
CLASS ............ multimodal physics-constrained neural-ODE model
INDICATION ....... HMA drug-resistance prediction and mechanism discovery in AML
PREPARED FOR ..... npj Digital Medicine
```

## Definition

`epiresist` dispenses the EpiResist-PINN architecture: three modality-specific encoders
(gene-expression Transformer with gene-set attention, DNA-methylation graph-attention network,
dose-response convolutional network) are fused by cross-modal attention under a physics-aware gate,
and the fused representation parameterizes three coupled ordinary differential equation systems —
epigenetic methylation kinetics, pharmacokinetic/pharmacodynamic response, and Lotka-Volterra
sensitive-to-resistant population dynamics. The dual output heads dispense a drug-response prediction
and a resistance-mechanism score; the differentiable ODE parameters yield interpretable pathway
signatures.

## Category and action

| Active principle | Where it acts | Governing relation |
|---|---|---|
| Epigenetic kinetics | `epiresist/alembics/epigenetic.py` | Eq. 1 (methylation / demethylation) |
| PK/PD response | `epiresist/alembics/pharmacology.py` | Eq. 2-3 (first-order clearance + Hill) |
| Population dynamics | `epiresist/alembics/population.py` | Eq. 4-5 (Lotka-Volterra, transition rate mu) |
| Coupling | `epiresist/alembics/coupling.py` | shared C, E, R/(S+R) |
| Physics-aware gating | `epiresist/compounding/gating.py` | Eq. 6 |
| Training objective | `epiresist/balances/objective.py` | Eq. 7 (lambda 0.1 / 0.01 / 0.05) |

## Preparation

pip:

```bash
pip install -e .
```

conda:

```bash
conda env create -f environment.yml
conda activate epiresist
pip install -e .
```

Docker:

```bash
docker build -t epiresist:0.1.0 .
docker run --rm epiresist:0.1.0 --help
```

## Formulary

| Compartment | Contents |
|---|---|
| `epiresist/materia` | cohort intake, preprocessing (`grinding`), batch container (`portions`) |
| `epiresist/preparations` | the three modality encoders |
| `epiresist/compounding` | cross-modal attention and physics-aware gate |
| `epiresist/alembics` | the three ODE rate laws, coupling, torchdiffeq integrator |
| `epiresist/balances` | the four-term training objective |
| `epiresist/assay` | metrics, GSEA enrichment, parameter attribution, statistics |
| `epiresist/dispensary` | trainer (`regimen`), evaluation and transfer (`rounds`) |
| `epiresist/counter` | the dispensing command line |
| `recipes/` | composable YAML formulae (an `inherit` chain) |
| `provings/` | the test suite |

## Method of compounding

```bash
python -m epiresist.counter triturate --formula main     # materialize a cohort
python -m epiresist.counter compound  --formula main     # train on the main formula
python -m epiresist.counter assay     --formula main     # evaluate
python -m epiresist.counter dispense  --formula main     # per-specimen predictions
python -m epiresist.counter decant    --formula main     # export the encoder pathway to ONNX
```

Any field can be overridden by editing a formula in `recipes/formula/` or by adding a new file with an
`inherit` line. Reported configurations ship as named formulae: `main`, the `ablation_*` series
(Table 2), and the `supplement_*` series (Tables S3, S6; GDSC and CCLE transfer).

## Assay

Each reported endpoint maps to one formula. Values are the manuscript means over 20 seeds with the
stated tolerance; the synthetic cohort that ships here exercises the pipeline rather than the cohorts
that produce these numbers.

| Reading | Formula | Manuscript value |
|---|---|---|
| Beat AML PCC | `assay --formula main` | 0.711 +/- 0.018 |
| Beat AML AUROC | `assay --formula main` | 0.832 +/- 0.015 |
| Beat AML pathway ES | `assay --formula main` | 0.743 +/- 0.028 |
| GDSC-AML PCC | `assay --formula supplement_gdsc` | 0.924 +/- 0.008 |
| CCLE-AML PCC | `assay --formula supplement_ccle` | 0.839 +/- 0.014 |
| w/o all ODEs pathway ES | `assay --formula ablation_no_all_odes` | 0.487 +/- 0.035 |
| Cross-dataset avg degradation | `rounds.cross_dataset_transfer` | 6.1% |

## Standard stock

Dataset links were checked before listing; an unreachable link is dropped (see `docs/deviations.md`).

| Source | Scope | Access |
|---|---|---|
| Beat AML | 805 patients, 409 with coupled genomic + drug data; 17,198 genes, 12,543 CpG, 122 drugs | https://www.vizome.org/ |
| TCGA-LAML | 200 adult de novo AML cases | https://portal.gdc.cancer.gov/projects/TCGA-LAML |
| CCLE / DepMap | ~1,000 cell lines, multi-omics + drug sensitivity | https://depmap.org/portal/ |
| GDSC (GDSC2) | 198 compounds x 969 cell lines, 32 myeloid | https://cog.sanger.ac.uk/cancerrxgene/ |
| TARGET-AML | pediatric AML, RNA-seq + methylation | https://portal.gdc.cancer.gov/projects/TARGET-AML |

The GDSC project site `cancerrxgene.org` now returns HTTP 410; the GDSC2 release files remain on the
Sanger object store shown above.

## Storage and equipment

| Quantity | Reported value |
|---|---|
| Accelerator | 1x NVIDIA A100 (80 GB) |
| Training | 4.2 GPU-hours per seed |
| Peak memory | 38 GB (full configuration) |
| Inference | 18.3 ms per specimen |
| Parameters | 8.7M total (1.2M in ODE components) |
| Solver | DOPRI5, rtol 1e-5, atol 1e-7, adjoint backpropagation |

## Identification tests

```bash
make test     # pytest: shapes, ODE invariants, single-batch overfit, metric correctness,
              # enrichment, gradient flow, determinism, config inheritance, ablation variants,
              # style guard, training smoke
make lint     # ruff + black + isort
make type     # mypy
make smoke    # triturate + compound + assay on the smoke formula
```

## Caution

All evaluation is computational and retrospective on public cohorts; the model is a research artifact,
not a clinical decision tool. The shipped cohort is synthetic and exists to drive the pipeline and the
tests. The three ODE systems are deterministic low-dimensional models of processes that involve
additional stochastic and spatial structure, and the equilibrium reading of cross-sectional data
follows the manuscript's stated assumption. Predicted resistance pathways are hypothesis-generating
and require experimental validation before any translational use.
