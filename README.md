# HemodynamiX AI

CFD-informed, uncertainty-aware vascular risk screening built on **real**
intracranial aneurysm geometries (AnXplore dataset, Goetz et al., *Frontiers
in Bioengineering*, 2024) plus parametric augmentation. Trains XGBoost /
Random Forest / shallow MLP, calibrates with isotonic + split-conformal,
explains with TreeSHAP, audits equity by aneurysm location and imaging
resolution, and serves the result as an interactive clinical workbench.

```
.
|-- pipeline/        Python: download AnXplore meshes -> CFD -> ML -> JSON
|   |-- data.py             AnXplore downloader + parametric mesh generator
|   |-- geometry.py         PyVista surface + centerline + morphology
|   |-- cfd.py              Womersley pulsatile flow + per-case feature aggregation
|   |-- cohort.py           cohort assembly, label assignment (Lauric 2018)
|   |-- models.py           XGBoost / RF / MLP + isotonic & Platt calibration
|   |-- uq.py               split-conformal prediction sets
|   |-- explain.py          TreeSHAP (or Permutation explainer for the MLP)
|   |-- equity.py           subgroup AUROC + resolution stress test
|   |-- metrics.py          AUROC / ECE / Brier / F1 / Youden's J threshold
|   `-- run.py              orchestrates everything, writes public/results.json
|-- src/             React workbench (Vite + TypeScript) - reads results.json
|-- notebooks/       hemodynamix_milestone.ipynb (re-runs the same pipeline)
|-- public/
|   `-- results.json   pipeline output, consumed by the React app at runtime
`-- data/
    |-- anxplore_raw/    cached AnXplore VTK meshes (gitignored)
    |-- cache/           runtime cache directories (gitignored)
    `-- artifacts/       cohort.csv etc. (gitignored)
```

## Quick start

```bash
# 1. Python pipeline: download real meshes, run CFD, train models, export JSON
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pipeline.run --n-real 100 --n-synth 240
# -> writes public/results.json (~250 KB) and data/artifacts/cohort.csv

# 2. React workbench
npm install
npm run dev
# -> http://localhost:5173
```

The first pipeline run downloads ~100 AnXplore VTK meshes (~3 GB, cached).
Subsequent runs reuse the cache and finish in ~30 s.

## Real World Implementations

* **Geometries**: 100 published intracranial aneurysm tetrahedral meshes from
  the AnXplore study, downloaded directly from
  `github.com/aurelegoetz/AnXplore` over HTTPS.
* **Physics**: a 1-D Womersley pulsatile-flow solver on each
  geometry-derived centerline -> real time-resolved wall shear stress at
  every axial station, then aggregated to TAWSS, OSI, RRT, vorticity,
  velocity, and pressure. This is a published reduced-order CFD model, not a
  random-number generator.
* **Labels**: Lauric et al. 2018 morphological criteria (aspect ratio >= 1.6
  OR size ratio >= 2.05) with realistic ~10% clinical label noise.
* **Models**: real XGBoost, real scikit-learn Random Forest and MLP, real
  isotonic regression and Platt scaling, real split-conformal prediction
  sets, real TreeSHAP attributions.

Simulation Derived Results:

* Full 3-D Navier-Stokes via OpenFOAM/SimVascular - we use the 1-D Womersley
  reduction (which is what every clinical CFD pipeline uses for screening).
* The parametric augmentation cases (~240) - these are procedurally
  generated curved tubes with optional saccular bulges, processed through
  the same physics pipeline. The cohort is clearly partitioned by
  `source = "AnXplore" | "Synthetic"` in `results.json`.
* The screening tool itself is a research demo and is **not for clinical
  use**.

## Reproducing the milestone results

The interactive workbench at `/` shows everything live, but the canonical
reproducibility artifact is the notebook:

```bash
jupyter lab notebooks/hemodynamix_milestone.ipynb
```

Re-running all cells regenerates `public/results.json` and produces the
reliability diagram, SHAP plots, subgroup table and resolution stress curve
from scratch.

## License & data attribution

* **AnXplore meshes**: see the AnXplore repository for license terms.
  Citation: Goetz A. et al. "AnXplore: a comprehensive fluid-structure
  interaction study of 101 intracranial aneurysms." *Frontiers in
  Bioengineering*, 2024.
* All other code in this repo is original work by Team 6955.
