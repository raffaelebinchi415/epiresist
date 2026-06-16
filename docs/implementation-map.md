# Implementation Map — EpiResist-PINN

All paper provenance lives in this file (source code carries no comments and no docstrings).
Columns: paper section / equation-figure-table / file path / module / notes.

## Governing equations

| Paper | Eq/Fig/Tab | File | Module | Notes |
|---|---|---|---|---|
| §4.3 ODE-1 epigenetic | Eq.1 | epiresist/alembics/epigenetic.py | EpigeneticState | dMi/dt = k_methyl(1-Mi) f(DNMT3A,DNMT3B) - k_demethyl Mi g(TET2,C); Mi bounded [0,1] via the (1-Mi) factor; f sigmoid over DNMT3A/3B, g over TET2 modulated by drug conc C from ODE-2 |
| §4.3 ODE-2 PK/PD | Eq.2-3 | epiresist/alembics/pharmacology.py | Pharmacology | dC/dt = -k_e C + k_a D(t); dE/dt = k_eff C^n/(EC50^n + C^n) - k_off E; Hill PD; k_e patient-specific |
| §4.3 ODE-3 population | Eq.4-5 | epiresist/alembics/population.py | Population | Lotka-Volterra S/R competition; delta_S(E), delta_R(E)=delta_R,max E alpha; transition rate mu (mechanism axis) |
| §4.3 coupling | Eq.4-5, prose | epiresist/alembics/coupling.py | CoupledSystem | ODE-1->2 via C; ODE-2->3 via E; ODE-3->1 via R/(S+R); single torchdiffeq state vector |
| §4.8 solver | — | epiresist/alembics/integrator.py | integrate | torchdiffeq odeint_adjoint DOPRI5 rtol 1e-5 atol 1e-7; fixed-step RK4 fallback for tests |
| §4.4 gene-expr encoder | Fig.2 | epiresist/preparations/transcript.py | TranscriptEncoder | Transformer, gene-set attention (intra/inter-set), 4 heads, 2 layers, d=256 |
| §4.4 methylation encoder | Fig.2 | epiresist/preparations/methylation.py | MethylationEncoder | GAT over CpG-gene bipartite graph (networkx), 2 layers, 4 heads, d=256 |
| §4.4 dose-response encoder | Fig.2 | epiresist/preparations/doseresponse.py | DoseResponseEncoder | 1D-CNN, kernels 3/5/7, filters 64/128/256, global average pool |
| §4.4 cross-modal fusion | Eq.6 | epiresist/compounding/crossmodal.py | CrossModalAttention | multi-head cross-attention, each modality alternately query |
| §4.4 physics-aware gating | Eq.6 | epiresist/compounding/gating.py | PhysicsGate | sigma(W_g [s_ODE; h_m]) elementwise-mult; s_ODE = [M_avg, E, S/(S+R)] |
| §4.2/§4.6 assembled model | Fig.2, Eq.7 | epiresist/formulary.py | EpiResistPINN, FormularyConfig | encoders -> fusion+gating -> coupled ODE -> dual heads (regression + resistance) + mechanism scores |

## Losses, metrics, mechanism discovery

| Paper | Eq/Fig/Tab | File | Module | Notes |
|---|---|---|---|---|
| §4.6 total loss | Eq.7 | epiresist/balances/objective.py | total_loss | L_pred (MSE + BCE) + 0.1 L_physics + 0.01 L_sparse + 0.05 L_temporal |
| §4.6 physics residual | Eq.7 | epiresist/balances/objective.py | physics_residual | sum_k ||r_k||^2 of the three ODE residuals at collocation points |
| §2.1 metrics | Table 1 | epiresist/assay/measures.py | pearson, spearman, rmse, auroc, f1 | regression + resistance-classification metrics |
| §4.5 enrichment | Table 2/4, §4.5 | epiresist/assay/enrichment.py | enrichment_score, gsea_prerank | GSEA NES; aggregate ES = mean top-K NES, K=10 |
| §4.5 attribution | Table 4, Fig.1 | epiresist/assay/attribution.py | parameter_attribution, resistance_signature | d theta / d X via autograd through the ODE solver for theta in {k_methyl, k_e, EC50, mu} |
| §2.1/§2.2 statistics | Table 1-3, S2 | epiresist/assay/statistics.py | bootstrap_ci, delong_test, welch_t | 95% bootstrap CI (1000 iters); DeLong AUC test; Welch t |

## Data, training, evaluation, CLI

| Paper | Eq/Fig/Tab | File | Module | Notes |
|---|---|---|---|---|
| §4.1/§4.7 data | Fig.1 | epiresist/materia/intake.py | CohortSpec, SyntheticCohort, load_cohort | deterministic synthetic cohort matching paper shapes with recoverable signal; manifest adapter for real data |
| §4.8 preprocessing | — | epiresist/materia/grinding.py | log2_quantile, beta_values, minmax_doseresponse | log2 + quantile norm (expr), beta extraction (methylation), min-max (dose-response) |
| — | — | epiresist/materia/portions.py | CohortBatch, collate | typed batch container + collate |
| §4.8 training | — | epiresist/dispensary/regimen.py | Regimen | Adam lr 5e-4 wd 1e-5, early stopping patience 20, grad clip 1.0, AMP optional, DDP-ready, EMA, atomic ckpt, seed restore |
| §2.1/§2.3 evaluation | Table 1/3, S1 | epiresist/dispensary/rounds.py | evaluate, cross_dataset_transfer, kfold | 5-fold CV within-dataset; held-out cross-dataset transfer |
| §4.8 reproducibility | — | epiresist/apparatus.py | seed_all, atomic_save, get_logger, resolve_device | single seed utility stored in every checkpoint |
| — | — | epiresist/counter/__main__.py | app | Typer: triturate / compound / assay / dispense / decant |

## Reported numbers reproduced (README Assay table)

| Endpoint | Paper value | Source | Command |
|---|---|---|---|
| Beat AML PCC | 0.711 +/- 0.018 | Table 1a | assay --formula main |
| Beat AML AUROC | 0.832 +/- 0.015 | Table 1a | assay --formula main |
| Beat AML pathway ES | 0.743 +/- 0.028 | Table 2 | assay --formula main |
| GDSC-AML PCC | 0.924 +/- 0.008 | Table 1b | assay --formula main data=gdsc |
| CCLE-AML PCC | 0.839 +/- 0.014 | Table 1b | assay --formula main data=ccle |
| Cross-dataset avg degradation | 6.1% | Table 3 | assay --formula supplement_cross_dataset |

## Verbatim code-availability statement (not placed in README per project rule)
"Code for EpiResist-PINN is available upon reasonable request to the corresponding author."
