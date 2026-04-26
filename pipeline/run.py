"""End-to-end pipeline runner.

Usage:
    python -m pipeline.run [--no-real]

Outputs:
    public/results.json     consumed by the React workbench
    data/artifacts/*.csv    feature/label tables for the notebook
    data/artifacts/*.png    diagnostic plots (reliability, SHAP, equity)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .cohort import build_cohort, feature_matrix, labels_array
from .config import FEATURE_KEYS, FEATURE_LABELS, LOCATIONS, PipelineConfig, ensure_dirs
from .equity import by_location, resolution_stress
from .explain import shap_global, shap_per_case
from .fem3d_pass import upgrade_with_fem3d
from .metrics import (
    auroc,
    best_threshold,
    brier,
    expected_calibration_error,
    reliability_bins,
)
from .models import fit_calibrator, fit_mlp, fit_rf, fit_xgb, TrainedModel
from .uq import calibrate, evaluate


def _stratified_train_cal_test(y, cfg: PipelineConfig, seed: int):
    n = len(y)
    idx = np.arange(n)
    idx_train_cal, idx_test = train_test_split(
        idx, test_size=cfg.cohort_test_frac, random_state=seed, stratify=y
    )
    y_tc = y[idx_train_cal]
    idx_train, idx_cal = train_test_split(
        idx_train_cal,
        test_size=cfg.cohort_cal_frac,
        random_state=seed + 1,
        stratify=y_tc,
    )
    return idx_train, idx_cal, idx_test


def run(
    cfg: PipelineConfig,
    *,
    no_real: bool = False,
    solver: str = "womersley",
    n_fem: int = 0,
) -> dict:
    ensure_dirs(cfg)
    if no_real:
        cfg.n_real_cases = 0

    t0 = time.time()
    cohort = build_cohort(cfg)
    print(f"[run] cohort built in {time.time()-t0:.1f}s")

    # Optional FEM3D upgrade pass: replace the Womersley features on the top
    # K AnXplore cases with steady Navier-Stokes results, and render the
    # headline 4K image for each. Everything downstream (ML, calibration,
    # SHAP, equity, JSON shape) is unchanged.
    n_fem_target = n_fem if solver in ("fem3d", "both") else 0
    if n_fem_target > 0:
        t_fem = time.time()
        upgraded = upgrade_with_fem3d(cohort, cfg, n_fem_target)
        print(
            f"[run] FEM3D pass: upgraded {len(upgraded)} cases in "
            f"{time.time() - t_fem:.0f}s ({(time.time()-t_fem)/max(1,len(upgraded)):.1f}s avg)"
        )

    X = feature_matrix(cohort)
    y = labels_array(cohort)
    locations = [c.location for c in cohort]

    if len(np.unique(y)) < 2:
        raise RuntimeError(
            f"Degenerate cohort: only one class present (n={len(y)}, sum={y.sum()})."
        )

    idx_train, idx_cal, idx_test = _stratified_train_cal_test(y, cfg, cfg.seed)
    Xtr, ytr = X[idx_train], y[idx_train]
    Xcal, ycal = X[idx_cal], y[idx_cal]
    Xte, yte = X[idx_test], y[idx_test]
    locs_test = [locations[i] for i in idx_test]

    print(f"[run] train={len(idx_train)} cal={len(idx_cal)} test={len(idx_test)}")

    print("[run] training models...")
    models: list[TrainedModel] = [
        fit_xgb(Xtr, ytr, cfg.seed),
        fit_rf(Xtr, ytr, cfg.seed),
        fit_mlp(Xtr, ytr, cfg.seed),
    ]
    for m in models:
        fit_calibrator(m, Xcal, ycal)

    evals: list[dict] = []
    for m in models:
        p = m.predict_proba(Xte)
        thr = best_threshold(p, yte)
        evals.append(
            {
                "name": m.name,
                "auroc": auroc(p, yte),
                "f1Pos": thr.f1,
                "sensitivity": thr.sensitivity,
                "specificity": thr.specificity,
                "accuracy": thr.accuracy,
                "ece": expected_calibration_error(p, yte),
                "brier": brier(p, yte),
                "predictions": [float(v) for v in p],
                "trueLabels": [int(v) for v in yte],
            }
        )

    evals.sort(key=lambda e: -e["auroc"])
    best_name = evals[0]["name"]
    best = next(m for m in models if m.name == best_name)
    p_test = best.predict_proba(Xte)
    p_cal = best.predict_proba(Xcal)
    print(f"[run] best model = {best_name} (AUROC={evals[0]['auroc']:.3f})")

    # Conformal
    q = calibrate(p_cal, ycal, cfg.conformal_alpha)
    conformal = evaluate(p_test, yte, q, cfg.conformal_alpha)

    # Reliability
    reliability = reliability_bins(p_test, yte, n_bins=10)

    # SHAP
    print("[run] explaining...")
    importance = shap_global(best, Xtr, Xte)
    # Top 4 highest-risk cases for the per-case attribution panel
    top_idx = np.argsort(-p_test)[: min(4, len(p_test))]
    case_ids = [cohort[idx_test[i]].case_id for i in top_idx]
    base_p = float(np.mean(ytr))
    attributions = shap_per_case(best, Xtr, Xte[top_idx], case_ids, baseline_p=base_p)

    # Equity + resolution
    by_loc = by_location(locs_test, p_test, yte)
    by_res = resolution_stress(best.predict_proba, Xte, yte, seed=cfg.seed + 5)

    # Persist tabular artifacts (so the notebook can read them)
    import pandas as pd

    df = pd.DataFrame(
        {
            "case_id": [c.case_id for c in cohort],
            "source": [c.source for c in cohort],
            "location": [c.location for c in cohort],
            **{k: [c.features[k] for c in cohort] for k in FEATURE_KEYS},
            "aspect_ratio": [c.morphology.aspect_ratio for c in cohort],
            "size_ratio": [c.morphology.size_ratio for c in cohort],
            "sphericity": [c.morphology.sphericity for c in cohort],
            "bulge_amplitude": [c.morphology.bulge_amplitude for c in cohort],
            "label_clean": [c.label_clean for c in cohort],
            "label": [c.label for c in cohort],
        }
    )
    df.to_csv(cfg.artifacts_dir / "cohort.csv", index=False)
    print(f"[run] wrote {cfg.artifacts_dir / 'cohort.csv'}")

    # ---------------------------------------------------------------- export JSON
    cohort_payload = []
    for c in cohort:
        entry = {
            "id": c.case_id,
            "source": c.source,
            "location": c.location,
            "label": c.label,
            "labelClean": c.label_clean,
            "solver": c.solver,
            "features": {k: float(c.features[k]) for k in FEATURE_KEYS},
            "morphology": {
                "aspectRatio": float(c.morphology.aspect_ratio),
                "sizeRatio": float(c.morphology.size_ratio),
                "sphericity": float(c.morphology.sphericity),
                "undulationIndex": float(c.morphology.undulation_index),
                "bulgeAmplitude": float(c.morphology.bulge_amplitude),
                "tortuosity": float(c.morphology.tortuosity),
                "centerlineLength": float(c.morphology.centerline_length),
                "surfaceArea": float(c.morphology.surface_area),
                "volume": float(c.morphology.volume),
            },
        }
        if c.cfd3d_image:
            entry["cfd3dImage"] = c.cfd3d_image
        if c.cfd3d_summary:
            entry["cfd3dSummary"] = c.cfd3d_summary
        cohort_payload.append(entry)

    # Prefer a FEM3D-upgraded case as the headline focus when available, so
    # the React inspector shows the real 3-D rendering. Fall back to the
    # highest-risk case with a stored Womersley grid/centerline.
    fem3d_cases = [c for c in cohort if c.solver == "fem3d" and c.cfd3d_image]
    if fem3d_cases:
        focus = max(fem3d_cases, key=lambda c: c.morphology.bulge_amplitude)
    else:
        cases_with_grid = [c for c in cohort if c.grid]
        if cases_with_grid:
            focus = max(cases_with_grid, key=lambda c: c.morphology.bulge_amplitude)
        else:
            focus = cohort[0]

    out = {
        "version": "1.0.0",
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "nRealCases": cfg.n_real_cases,
            "nSyntheticCases": cfg.n_synthetic_cases,
            "labelNoiseRate": cfg.label_noise_rate,
            "conformalAlpha": cfg.conformal_alpha,
            "seed": cfg.seed,
        },
        "featureKeys": list(FEATURE_KEYS),
        "featureLabels": FEATURE_LABELS,
        "cohort": cohort_payload,
        "split": {
            "train": [int(i) for i in idx_train],
            "cal": [int(i) for i in idx_cal],
            "test": [int(i) for i in idx_test],
        },
        "models": evals,
        "bestModel": best_name,
        "conformal": {
            "alpha": conformal.alpha,
            "q": conformal.q,
            "empiricalCoverage": conformal.empirical_coverage,
            "intervalWidthMean": conformal.interval_width_mean,
            "abstainRate": conformal.abstain_rate,
        },
        "reliability": reliability,
        "importance": [
            {"feature": fi.feature, "delta": float(fi.delta)} for fi in importance
        ],
        "caseAttributions": [
            {
                "caseId": a.case_id,
                "baseline": a.baseline,
                "prediction": a.prediction,
                "contributions": a.contributions,
            }
            for a in attributions
        ],
        "byLocation": [
            {
                "subgroup": s.subgroup,
                "n": s.n,
                "auroc": s.auroc,
                "positives": s.positives,
            }
            for s in by_loc
        ],
        "byResolution": [
            {"noiseLevel": r.noise_level, "auroc": r.auroc} for r in by_res
        ],
        "focusCase": {
            "id": focus.case_id,
            "source": focus.source,
            "location": focus.location,
            "label": focus.label,
            "solver": focus.solver,
            "morphology": cohort_payload[[c.case_id for c in cohort].index(focus.case_id)]["morphology"],
            "narrativeBullets": _build_narrative(focus),
            "uncertaintySummary": _build_uncertainty_summary(focus, conformal),
            "modalityNote": _modality_note(focus),
            "grid": focus.grid,
            "centerline": focus.centerline,
            "radiusProfile": focus.radius_profile,
            "cfd3dImage": focus.cfd3d_image,
            "cfd3dSummary": focus.cfd3d_summary,
        },
    }

    out_path = cfg.public_dir / "results.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"[run] wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

    return out


def _build_narrative(focus) -> list[str]:
    f = focus.features
    m = focus.morphology
    bullets = [
        f"Case {focus.case_id} ({focus.source}, {focus.location}) - aspect ratio {m.aspect_ratio:.2f}, size ratio {m.size_ratio:.2f}.",
        f"Wall TAWSS {f['tawss']:.2f} Pa, OSI {f['osi']:.2f}, RRT {f['rrt']:.2f} 1/Pa.",
        f"Peak vorticity {f['vorticity']:.0f} 1/s, peak velocity {f['velocity']:.2f} m/s.",
        f"Sphericity {m.sphericity:.2f}; bulge amplitude {m.bulge_amplitude:.2f} mm.",
    ]
    if focus.label == 1:
        bullets.append("Morphological criteria flag this case as elevated rupture risk.")
    return bullets


def _build_uncertainty_summary(focus, conformal) -> str:
    return (
        f"Split-conformal calibration at alpha={conformal.alpha:.2f}: "
        f"empirical coverage on the held-out cohort = {conformal.empirical_coverage:.2f}, "
        f"abstain rate = {conformal.abstain_rate:.2f}. Heatmap colour encodes mean wall risk; "
        f"opacity encodes posterior standard deviation (epistemic uncertainty)."
    )


def _modality_note(focus) -> str:
    if focus.solver == "fem3d":
        return (
            "Steady 3-D incompressible Navier-Stokes (Taylor-Hood P2/P1, Picard) on the real "
            "AnXplore tetrahedral fluid domain. Wall colored by computed WSS; streamlines seeded "
            "at the inlet plane. Demonstration only - not for clinical use."
        )
    return (
        "Reduced-order Womersley pulsatile CFD on real AnXplore mesh + parametric augmentation. "
        "Demonstration only - not for clinical use."
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-real", action="store_true", help="skip AnXplore download")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--n-real", type=int, default=90)
    p.add_argument("--n-synth", type=int, default=220)
    p.add_argument(
        "--solver",
        choices=("womersley", "fem3d", "both"),
        default="womersley",
        help="womersley = fast 1-D analytic CFD on every case; "
             "fem3d/both = additionally run steady 3-D Navier-Stokes + render "
             "the headline image on the top --n-fem AnXplore cases.",
    )
    p.add_argument(
        "--n-fem",
        type=int,
        default=0,
        help="Number of AnXplore cases to upgrade with the steady 3-D NS solver "
             "(only used when --solver=fem3d or both).",
    )
    args = p.parse_args(argv)

    cfg = PipelineConfig(
        seed=args.seed,
        n_real_cases=args.n_real,
        n_synthetic_cases=args.n_synth,
    )
    run(cfg, no_real=args.no_real, solver=args.solver, n_fem=args.n_fem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
