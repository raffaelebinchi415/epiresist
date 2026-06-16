# Project Context — EpiResist-PINN

    project_name       : epiresist                                   [HIGH]
    domain             : computational oncology — HMA drug-resistance
                         prediction & mechanism discovery in AML      [HIGH]
    framework          : PyTorch 2.x + torchdiffeq (DOPRI5, adjoint)  [HIGH]
    venue              : npj Digital Medicine                         [MEDIUM]
    primary_datasets   : 5 datasets (see §6)                          [HIGH]
    compute_target     : 1x NVIDIA A100 80GB, 4.2 GPU-h/seed          [HIGH]
    hparams_reference  : Methods §4.8 + Eq.7 weights + Table S3        [HIGH]
    supp_path          : embedded Supplementary Information (pp. 23-33 of the manuscript PDF)
    extra_signals      : 47 shared ODE parameters; no released checkpoints;
                         3 algorithm-bearing ODE blocks (Eq.1-5); code-availability =
                         "upon reasonable request"; computational-only validation.

    NEEDS_USER_DECISION: 0

## 1. project_name
`epiresist` — taken directly from the model name EpiResist-PINN (Abstract; Methods §4.2). HIGH.

## 2. supp_path
Supplementary Information is bound into the same manuscript PDF (Supplementary Tables S1-S8,
Supplementary Figure S1). No separate sibling file was found. HIGH.

## 3. domain
Computational oncology. Specifically multimodal drug-response regression + resistance classification
and mechanistic resistance-pathway discovery for hypomethylating-agent (HMA) treated acute myeloid
leukemia. Source: Abstract; Introduction §1; Methods §4.1. HIGH.

## 4. framework
PyTorch 2.x with torchdiffeq for the coupled neural-ODE integration. Methods §4.8 states the coupled
systems are integrated with a Dormand-Prince (DOPRI5) adaptive Runge-Kutta solver implemented in
torchdiffeq, with adjoint backpropagation, absolute tolerance 1e-7 and relative tolerance 1e-5. HIGH.

## 5. venue
npj Digital Medicine. Inferred from the manuscript filename prefix and the structured-abstract +
Data/Code Availability layout typical of the journal. MEDIUM (no explicit "submitted to" line).

## 6. primary_datasets
Each dataset link was verified before inclusion; unreachable links are dropped.

| Dataset | Scope | Access (verified) | License / status |
|---|---|---|---|
| Beat AML | 805 patients / 942 specimens; 409 with coupled genomic + drug-response data; p=17,198 genes, q=12,543 CpG sites, 122 drugs, 7-point series | https://www.vizome.org/ (HTTP 200) | Open via Genomic Data Commons / Vizome; primary evaluation cohort |
| TCGA-LAML | 200 adult de novo AML cases; RNA-seq 179, methylation 194 | https://portal.gdc.cancer.gov/projects/TCGA-LAML (GDC API confirmed) | Open access tier; GDC Data Portal |
| CCLE / DepMap | ~1,000 cell lines, multi-omics + drug sensitivity; AML-relevant subset | https://depmap.org/portal/ (HTTP 200) | Open; Broad Institute DepMap |
| GDSC (GDSC2) | 198 compounds x 969 cell lines; 32 AML / myeloid lines | https://cog.sanger.ac.uk/cancerrxgene/ (HTTP 200) — Sanger object store | Open; **the manuscript URL https://www.cancerrxgene.org/ returns HTTP 410 Gone and is dropped** (see deviations D1) |
| TARGET-AML | pediatric AML, RNA-seq + methylation + WGS/WXS | https://portal.gdc.cancer.gov/projects/TARGET-AML (GDC API confirmed) | Open access tier; GDC Data Portal |

## 7. compute_target
Single NVIDIA A100 (80 GB). Training 4.2 GPU-hours per seed on the largest configuration; peak memory
38 GB for the full model; inference 18.3 ms per sample; 8.7M parameters (1.2M in ODE components).
Source: Table 5; Methods §4.8. HIGH.

## 8. hparams_reference
Methods §4.8 (Implementation details): Adam, lr 5e-4, weight decay 1e-5, max 200 epochs, early stopping
patience 20, batch size 32, gradient clipping max-norm 1.0, encoder hidden dim d=256, DOPRI5 rtol 1e-5 /
atol 1e-7, 5-fold cross-validation, 20 seeds (42, 137, 256, 314, 512, 628, 789, 1024, 1337, 1729, 2048,
2718, 3141, 4096, 5000, 6174, 7890, 8192, 9001, 9999). Loss weights (Methods §4.6 / Eq.7): lambda1 0.1
(physics), lambda2 0.01 (sparsity), lambda3 0.05 (temporal). Sensitivity grids in Table S3. HIGH.

## 9. extra_signals
- 47 shared ODE parameters (ODE-1: 8; ODE-2: 6; ODE-3: 7; coupling + gating: 26); 3 patient-specific
  (k_e, EC50, mu). Source: Methods §4.3.
- No released checkpoints; weights "upon reasonable request".
- Three coupled-ODE blocks carry the governing equations (Eq.1-5); gating Eq.6; loss Eq.7.
- SI-only experiments: per-seed table (S1), hyperparameter sensitivity (S3), learned-parameter subgroup
  distributions (S4), baseline reproduction (S5), solver sensitivity (S6), drug-specific performance
  (S7), ODE-specification robustness (S8).
- Verbatim code-availability statement (kept here, not in the README per project rule):
  "Code for EpiResist-PINN is available upon reasonable request to the corresponding author."
- All validation is computational/retrospective on public datasets; no prospective clinical testing.
