"""Upgrade the headline AnXplore subset to true 3-D Navier-Stokes physics.

The default (Womersley) cohort solves a fast 1-D analytic pulsatile model on
~340 cases for ML training. This pass picks the K most-clinically-interesting
real meshes, re-runs them with the steady incompressible Navier-Stokes solver
in `pipeline.cfd_3d` (Taylor-Hood, Picard), renders a 4K WSS+streamlines
PNG to `public/cases/{case_id}.png`, and overwrites the case's six-feature
vector with the FEM3D-derived numbers.

Nothing else in the pipeline changes shape: the React workbench still
reads the same `results.json` with the same six features per case, just
better physics on the cases we showcase in the inspector.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from .cfd_3d import solve_case
from .cohort import CohortCase
from .config import FEATURE_KEYS, PipelineConfig
from .render import render_case

LOG = logging.getLogger("hxai.fem3d_pass")


def _candidate_score(case: CohortCase) -> float:
    """Higher = more interesting for the FEM3D pass.

    We want real (AnXplore) meshes, prefer cases with clear bulges (where
    the 3-D physics matters most for the demo image), and break ties by
    aspect ratio so we still cover near-spherical and elongated sacs.
    """
    if case.source != "AnXplore" or case.volume_path is None:
        return -np.inf
    return (
        2.0 * float(case.morphology.bulge_amplitude)
        + 1.0 * float(case.morphology.aspect_ratio)
        + 0.5 * float(case.morphology.size_ratio)
    )


def select_subset(cohort: List[CohortCase], n: int) -> List[CohortCase]:
    """Pick up to N AnXplore cases by descending interest score.

    We sort by the score above and require an on-disk volume_path, so every
    returned case is guaranteed to have a real .vtk available for tetgen.
    """
    eligible = [c for c in cohort if c.source == "AnXplore" and c.volume_path is not None]
    if not eligible:
        LOG.warning("FEM3D pass requested but no AnXplore cases available")
        return []
    eligible.sort(key=_candidate_score, reverse=True)
    return eligible[: max(0, int(n))]


def _morph_payload(case: CohortCase) -> dict:
    m = case.morphology
    return {
        "aspectRatio": float(m.aspect_ratio),
        "sizeRatio": float(m.size_ratio),
        "sphericity": float(m.sphericity),
        "bulgeAmplitude": float(m.bulge_amplitude),
    }


def upgrade_with_fem3d(
    cohort: List[CohortCase],
    cfg: PipelineConfig,
    n: int,
    *,
    image_subdir: str = "cases",
    headless: bool = True,
) -> List[CohortCase]:
    """Run the steady 3-D NS solver + renderer on the top-N AnXplore cases.

    Mutates the matching `CohortCase` objects in-place: replaces `features`
    with FEM3D-derived numbers, sets `solver = "fem3d"`, stores the relative
    URL of the rendered PNG in `cfd3d_image`, and records a small summary
    dict for the inspector card.

    Returns the list of cases that were upgraded.
    """
    if n <= 0:
        return []

    if headless and "PYVISTA_OFF_SCREEN" not in os.environ:
        os.environ["PYVISTA_OFF_SCREEN"] = "true"

    image_dir = cfg.public_dir / image_subdir
    image_dir.mkdir(parents=True, exist_ok=True)

    chosen = select_subset(cohort, n)
    if not chosen:
        return []

    LOG.info("FEM3D pass: %d candidates selected", len(chosen))
    upgraded: List[CohortCase] = []

    for k, case in enumerate(chosen, 1):
        t0 = time.time()
        png_path = image_dir / f"{case.case_id}.png"
        try:
            res = solve_case(case.case_id, case.volume_path, _morph_payload(case))
            render_case(res, png_path)
        except Exception as exc:
            LOG.exception("FEM3D failed on %s (%d/%d): %s", case.case_id, k, len(chosen), exc)
            continue

        # Overwrite the canonical 6 features with FEM3D-derived numbers
        for fk in FEATURE_KEYS:
            if fk in res.features:
                case.features[fk] = float(res.features[fk])

        case.solver = "fem3d"
        case.cfd3d_image = f"{image_subdir}/{png_path.name}"
        wss = res.wss_per_facet
        wall = res.case_mesh.wall_facets
        u_max = float(np.linalg.norm(res.velocity_coef.reshape(-1, 3), axis=1).max())
        case.cfd3d_summary = {
            "solver": "Steady 3-D Navier-Stokes (Taylor-Hood, Picard)",
            "elements": int(res.case_mesh.skfem_mesh.t.shape[1]),
            "velocityDOFs": int(res.Vh.N),
            "wallFacets": int(wall.size),
            "tawssPa": float(np.mean(wss)) if wss.size else 0.0,
            "wssP95Pa": float(np.percentile(wss, 95)) if wss.size else 0.0,
            "uMaxMs": u_max,
            "inletRadiusMm": float(res.case_mesh.inlet_radius * 1e3),
            "renderPath": case.cfd3d_image,
            "secondsPerCase": round(time.time() - t0, 1),
        }
        upgraded.append(case)
        LOG.info(
            "FEM3D %d/%d  %s  TAWSS=%.2f Pa  |u|max=%.2f m/s  -> %s  (%.1fs)",
            k, len(chosen), case.case_id,
            case.cfd3d_summary["tawssPa"], u_max, case.cfd3d_image,
            case.cfd3d_summary["secondsPerCase"],
        )

    LOG.info("FEM3D pass: upgraded %d/%d cases", len(upgraded), len(chosen))
    return upgraded
