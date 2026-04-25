"""Download real intracranial aneurysm meshes from AnXplore + parametric fallback."""

from __future__ import annotations

import concurrent.futures as cf
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pyvista as pv
import requests

from .config import (
    ANXPLORE_BASE,
    ANXPLORE_NAMED_BASE,
    NAMED_CASES,
    PipelineConfig,
)


@dataclass
class RawMesh:
    """A real or synthetic vascular mesh, ready for feature extraction."""

    case_id: str
    source: str            # "AnXplore" | "Synthetic"
    surface: pv.PolyData   # cleaned, triangulated surface mesh
    raw_volume: float      # mesh volume in mm^3
    raw_area: float        # surface area in mm^2


# -- AnXplore downloader -----------------------------------------------------


def _download_one(idx: int | str, cfg: PipelineConfig) -> Optional[Path]:
    """Download a single AnXplore VTK if not already cached."""
    if isinstance(idx, str):
        url = ANXPLORE_NAMED_BASE.format(name=idx)
        out = cfg.raw_dir / f"Fluid_case{idx}.vtk"
    else:
        url = ANXPLORE_BASE.format(idx=idx)
        out = cfg.raw_dir / f"Fluid_{idx}.vtk"

    if out.exists() and out.stat().st_size > 1_000_000:
        return out

    try:
        with requests.get(url, stream=True, timeout=180) as r:
            if r.status_code != 200:
                return None
            tmp = out.with_suffix(".vtk.part")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)
            os.replace(tmp, out)
        return out
    except Exception as e:
        print(f"  ! download failed for {url}: {e}")
        return None


def download_anxplore(cfg: PipelineConfig, n: int) -> List[Path]:
    """Download up to n AnXplore real-mesh cases in parallel.

    Always pulls the four hand-curated cases (A, B, C, R) first because they
    are referenced in the original FSI paper, then numeric cases 0..N.
    """
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    targets: list[int | str] = list(NAMED_CASES) + list(range(0, max(0, n - len(NAMED_CASES))))

    results: list[Path] = []
    print(f"[data] Downloading {len(targets)} AnXplore meshes (cached if present)...")
    with cf.ThreadPoolExecutor(max_workers=cfg.download_workers) as ex:
        futs = {ex.submit(_download_one, t, cfg): t for t in targets}
        for fut in cf.as_completed(futs):
            p = fut.result()
            if p is not None:
                results.append(p)
    results.sort()
    print(f"[data] Got {len(results)} real meshes on disk.")
    return results


# -- Mesh loader -------------------------------------------------------------


def _surface_from_vtk(path: Path) -> pv.PolyData:
    grid = pv.read(str(path))
    if isinstance(grid, pv.UnstructuredGrid):
        surf = grid.extract_surface(algorithm="dataset_surface")
    else:
        surf = grid.extract_surface() if hasattr(grid, "extract_surface") else grid
    surf = surf.triangulate().clean()
    return surf


def load_real_meshes(paths: Iterable[Path]) -> List[RawMesh]:
    out: list[RawMesh] = []
    for p in paths:
        try:
            surf = _surface_from_vtk(p)
            stem = p.stem.replace("Fluid_", "")
            try:
                vol = float(surf.volume)
            except Exception:
                vol = float(np.nan)
            out.append(
                RawMesh(
                    case_id=f"AX-{stem}",
                    source="AnXplore",
                    surface=surf,
                    raw_volume=vol,
                    raw_area=float(surf.area),
                )
            )
        except Exception as e:
            print(f"  ! failed to load {p.name}: {e}")
    return out


# -- Parametric mesh generator ------------------------------------------------


def _curved_tube_with_bulge(
    n_axial: int,
    n_circ: int,
    length: float,
    base_radius: float,
    bulge_position: float,
    bulge_factor: float,
    bulge_extent: float,
    curve_amp: float,
    rng: np.random.Generator,
) -> pv.PolyData:
    """Generate a curved cylindrical tube with a Gaussian saccular bulge.

    Returns a triangulated surface (PolyData).
    """
    s = np.linspace(0.0, 1.0, n_axial)
    theta = np.linspace(0.0, 2.0 * np.pi, n_circ, endpoint=False)
    SS, TT = np.meshgrid(s, theta, indexing="ij")

    # Centerline: arc in xy-plane (gentle bend) + slight z perturbation
    cx = curve_amp * np.sin(np.pi * s)
    cy = length * s
    cz = 0.2 * curve_amp * np.sin(2.0 * np.pi * s)

    # Local radius: base + Gaussian bulge
    sigma = max(0.04, bulge_extent)
    bulge = (bulge_factor - 1.0) * np.exp(
        -0.5 * ((s - bulge_position) / sigma) ** 2
    )
    R = base_radius * (1.0 + bulge)
    R = R + 0.02 * base_radius * rng.standard_normal(R.shape)

    # Frenet-like local frame using finite differences
    dC_ds = np.gradient(np.column_stack([cx, cy, cz]), axis=0)
    tangent = dC_ds / (np.linalg.norm(dC_ds, axis=1, keepdims=True) + 1e-9)
    up = np.tile(np.array([0.0, 0.0, 1.0]), (n_axial, 1))
    bnorm = np.cross(tangent, up)
    bnorm = bnorm / (np.linalg.norm(bnorm, axis=1, keepdims=True) + 1e-9)
    nrml = np.cross(bnorm, tangent)

    pts = np.zeros((n_axial * n_circ, 3))
    for i in range(n_axial):
        ring = (
            R[i]
            * (
                np.outer(np.cos(theta), nrml[i])
                + np.outer(np.sin(theta), bnorm[i])
            )
        )
        pts[i * n_circ : (i + 1) * n_circ, 0] = cx[i] + ring[:, 0]
        pts[i * n_circ : (i + 1) * n_circ, 1] = cy[i] + ring[:, 1]
        pts[i * n_circ : (i + 1) * n_circ, 2] = cz[i] + ring[:, 2]

    # Build quad faces, then triangulate
    faces = []
    for i in range(n_axial - 1):
        for j in range(n_circ):
            j1 = (j + 1) % n_circ
            a = i * n_circ + j
            b = i * n_circ + j1
            c = (i + 1) * n_circ + j1
            d = (i + 1) * n_circ + j
            faces.extend([4, a, b, c, d])
    faces = np.array(faces, dtype=np.int64)

    surf = pv.PolyData(pts, faces).triangulate().clean()
    return surf


def generate_parametric(cfg: PipelineConfig, n: int) -> List[RawMesh]:
    """Procedurally generate n parametric vascular geometries.

    Half are healthy curved tubes, half have varying saccular aneurysm bulges.
    All produce realistic surface meshes the same downstream pipeline can consume.
    """
    rng = np.random.default_rng(cfg.seed + 7)
    out: list[RawMesh] = []
    for i in range(n):
        is_diseased = rng.random() < 0.5
        bulge_factor = (
            1.0 + rng.uniform(0.0, 0.18)
            if not is_diseased
            else rng.uniform(1.4, 2.6)
        )
        bulge_pos = rng.uniform(0.35, 0.65)
        bulge_extent = rng.uniform(0.04, 0.13)
        base_r = rng.uniform(1.4, 2.4)              # mm
        length = rng.uniform(28.0, 48.0)            # mm
        curve_amp = rng.uniform(2.0, 7.0)           # mm

        try:
            surf = _curved_tube_with_bulge(
                n_axial=64,
                n_circ=28,
                length=length,
                base_radius=base_r,
                bulge_position=bulge_pos,
                bulge_factor=bulge_factor,
                bulge_extent=bulge_extent,
                curve_amp=curve_amp,
                rng=rng,
            )
            try:
                vol = float(surf.volume)
            except Exception:
                vol = float(np.nan)
            out.append(
                RawMesh(
                    case_id=f"SY-{i + 1:03d}",
                    source="Synthetic",
                    surface=surf,
                    raw_volume=vol,
                    raw_area=float(surf.area),
                )
            )
        except Exception as e:
            print(f"  ! parametric mesh {i} failed: {e}")
    return out
