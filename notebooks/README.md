# HemodynamiX AI - reproducibility notebook

`hemodynamix_milestone.ipynb` runs the full Milestone 2 design and reproduces
the Milestone 3 / 4 results end-to-end **on real intracranial aneurysm
geometries** from the AnXplore dataset (Goetz et al., *Frontiers in
Bioengineering*, 2024).

The notebook is intentionally a thin presentation layer: every heavy step
lives in the importable `pipeline/` package at the repository root. The same
code path produces both the figures in this notebook and the
`public/results.json` that the React workbench renders.

What the pipeline actually does, in order:

1. **Download** ~100 real tetrahedral aneurysm meshes from AnXplore (cached).
2. **Generate** parametric curved-tube cases for cohort augmentation.
3. **PyVista** surface extraction + principal-axis centerline + morphology
   (aspect ratio, size ratio, sphericity, undulation, tortuosity).
4. **Womersley pulsatile flow** on each centerline -> time-resolved WSS,
   then aggregate to TAWSS / OSI / RRT / vorticity / velocity / pressure.
5. **Labels** from Lauric et al. 2018 morphological criteria + 10% noise.
6. **XGBoost / Random Forest / Shallow MLP**, **isotonic + Platt** calibration.
7. **TreeSHAP** + **split-conformal prediction sets** + equity audit
   (subgroup AUROC + resolution stress test).
8. **Export** `../public/results.json` for the React workbench.

## Setup

```bash
cd ..                              # back to project root
python3 -m venv .venv
source .venv/bin/activate
pip install -r notebooks/requirements.txt

jupyter lab notebooks/hemodynamix_milestone.ipynb
```

Then **Run All Cells**. The first run downloads ~100 AnXplore meshes
(~3 GB, cached on disk so subsequent runs are ~30 s).

## Editing the notebook

The notebook is generated from `build_notebook.py` so the source of truth is
plain Python (easy diffs, no JSON escaping). To change a cell, edit the
corresponding `md(...)` / `code(...)` block, then:

```bash
python3 notebooks/build_notebook.py
```

## Known macOS 15.7 quirk

If the Jupyter kernel dies on startup with `SIGABRT` in `psutil_net_if_addrs`,
pin `psutil<7` (already in `requirements.txt`) - newer psutil triggers a
libmalloc abort on Sequoia.
