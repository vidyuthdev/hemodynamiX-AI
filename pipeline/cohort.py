"""Cohort assembly: real + parametric meshes -> features + labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

from .cfd import HemoFields, aggregate_features, solve_womersley, wall_risk_grid, centerline_trace
from .config import FEATURE_KEYS, LOCATIONS, PipelineConfig
from .data import RawMesh, download_anxplore, generate_parametric, load_real_meshes
from .geometry import Morphology, morphology, downsample_radius_profile

# Lauric et al. 2018 morphological thresholds for high rupture risk.
ASPECT_RATIO_THRESHOLD = 1.6
SIZE_RATIO_THRESHOLD = 2.05


@dataclass
class CohortCase:
    case_id: str
    source: str
    location: str
    morphology: Morphology
    fields: HemoFields
    features: dict[str, float]
    label: int
    label_clean: int
    grid: List[List[dict]] = field(default_factory=list)
    centerline: List[dict] = field(default_factory=list)
    radius_profile: List[float] = field(default_factory=list)


def _assign_location(case_id: str, rng: np.random.Generator) -> str:
    # Idealised cases distributed across the three commonest aneurysm locations
    # we report on in the equity audit. Real AnXplore cases cycle deterministically
    # so we get balanced subgroups for the test set.
    h = (sum(ord(c) for c in case_id) + rng.integers(0, 3)) % len(LOCATIONS)
    return LOCATIONS[h]


def _label_from_morphology(m: Morphology) -> int:
    return int(
        (m.aspect_ratio >= ASPECT_RATIO_THRESHOLD)
        or (m.size_ratio >= SIZE_RATIO_THRESHOLD)
        or (m.bulge_amplitude >= 1.0)
    )


def build_cohort(
    cfg: PipelineConfig,
    *,
    keep_grids_for: int = 4,
) -> List[CohortCase]:
    """Assemble cohort from real downloads + parametric augmentation."""
    rng = np.random.default_rng(cfg.seed)

    # Download + load real AnXplore meshes
    real_meshes: List[RawMesh] = []
    if cfg.n_real_cases > 0:
        try:
            real_paths = download_anxplore(cfg, cfg.n_real_cases)
            real_meshes = load_real_meshes(real_paths)
        except Exception as e:
            print(f"[cohort] real mesh load failed, falling back to synthetic only: {e}")

    print(f"[cohort] real meshes loaded: {len(real_meshes)}")

    # Generate parametric meshes to round out the cohort
    syn_meshes: List[RawMesh] = []
    if cfg.n_synthetic_cases > 0:
        syn_meshes = generate_parametric(cfg, cfg.n_synthetic_cases)
    print(f"[cohort] synthetic meshes generated: {len(syn_meshes)}")

    all_meshes = real_meshes + syn_meshes
    if not all_meshes:
        raise RuntimeError("No meshes available - aborting cohort build")

    cohort: List[CohortCase] = []

    # Sort so we keep heatmap grids for the most-likely-positive cases
    bulge_estimates = []
    for rm in all_meshes:
        try:
            morph = morphology(rm.surface)
        except Exception as e:
            print(f"  ! morphology failed for {rm.case_id}: {e}")
            continue
        bulge_estimates.append((rm, morph))

    bulge_estimates.sort(key=lambda x: x[1].bulge_amplitude, reverse=True)
    grid_quota = keep_grids_for

    for rm, morph in bulge_estimates:
        try:
            fields = solve_womersley(morph.centerline)
            feats = aggregate_features(
                morph.centerline,
                fields,
                morph_aspect_ratio=morph.aspect_ratio,
                morph_bulge_amplitude_mm=morph.bulge_amplitude,
            )
        except Exception as e:
            print(f"  ! cfd failed for {rm.case_id}: {e}")
            continue

        loc = _assign_location(rm.case_id, rng)
        clean_label = _label_from_morphology(morph)
        # realistic clinical label noise
        label = clean_label
        if rng.random() < cfg.label_noise_rate:
            label = 1 - clean_label

        case = CohortCase(
            case_id=rm.case_id,
            source=rm.source,
            location=loc,
            morphology=morph,
            fields=fields,
            features=feats,
            label=int(label),
            label_clean=int(clean_label),
        )

        if grid_quota > 0:
            case.grid = wall_risk_grid(fields, n_circ=22, n_axial=18, rng=rng)
            case.centerline = centerline_trace(fields, morph.centerline, n=32)
            case.radius_profile = downsample_radius_profile(morph.centerline, n=24)
            grid_quota -= 1

        cohort.append(case)

    print(f"[cohort] final cohort size: {len(cohort)}")
    pos = sum(c.label for c in cohort)
    print(f"[cohort] prevalence (noisy): {pos}/{len(cohort)} = {pos/len(cohort):.2f}")
    return cohort


def feature_matrix(cohort: List[CohortCase]) -> np.ndarray:
    return np.array([[c.features[k] for k in FEATURE_KEYS] for c in cohort], dtype=np.float64)


def labels_array(cohort: List[CohortCase]) -> np.ndarray:
    return np.array([c.label for c in cohort], dtype=np.int64)
