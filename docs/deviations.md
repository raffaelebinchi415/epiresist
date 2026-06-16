# Deviations

Intentional departures from the manuscript, each with a paper-section anchor and justification.

## D1 — GDSC access URL substituted
- Anchor: §4.7 Datasets / Data Availability; manuscript URL `https://www.cancerrxgene.org/`.
- Observed: the entire `cancerrxgene.org` domain returns HTTP 410 Gone (verified on `/`,
  `/downloads`, `/about` with a browser user-agent; sibling links vizome.org and depmap.org/portal
  return 200 under the same agent, so this is a real server response, not a bot block).
- Action: the dead URL is dropped. GDSC2 release data remain reachable on the Sanger object store
  `https://cog.sanger.ac.uk/cancerrxgene/` (HTTP 200), which is used as the documented access point.
- Effect: documentation only; no change to model behavior.

## D2 — Synthetic cohort shipped in place of restricted/large raw data
- Anchor: §4.1, §4.7.
- Observed: Beat AML / TCGA-LAML / CCLE-DepMap / GDSC / TARGET-AML are large and partly access-gated,
  and the manuscript performs computational-only evaluation on cross-sectional snapshots.
- Action: `epiresist/materia/intake.py` ships a deterministic synthetic cohort that matches the paper
  feature shapes (gene-expression, CpG methylation, multi-concentration dose-response, resistance label)
  and embeds recoverable signal so the overfit and metric tests are meaningful. A manifest adapter loads
  real arrays when present at the verified URLs above.
- Effect: enables a self-contained run; real-data runs use the manifest adapter unchanged.

## D3 — Equilibrium (steady-state) ODE integration target
- Anchor: §4.3 "Equilibrium interpretation of ODE dynamics from cross-sectional data".
- Observed: training data are cross-sectional snapshots, not time series.
- Action: the coupled system is integrated from biologically motivated initial conditions to a terminal
  state, and the loss matches that terminal state to observations, as the manuscript specifies. This is
  a faithful implementation of the stated equilibrium interpretation, recorded here for clarity.
- Effect: none beyond the manuscript's own design.
