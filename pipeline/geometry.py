"""Geometric / morphological feature extraction from vascular surface meshes.

We approximate a centerline by:
  1. Performing PCA on the surface vertices to find the principal vessel axis.
  2. Binning vertices along that axis to estimate centroid + cross-section
     radius profiles.

This is not as accurate as VMTK's geodesic centerline but is robust on
arbitrary meshes (including the parametric ones), runs in O(n) time, and
produces the same morphology / diameter signal we need.

Morphological features (per Lauric et al. 2018, Dhar et al. 2008):
  - aspect_ratio_morph  = max diameter / median (parent) diameter along axis
  - size_ratio_morph    = max diameter / parent vessel radius * 2
  - sphericity          = pi^(1/3) * (6V)^(2/3) / A   (range 0..1, sphere=1)
  - undulation_index    = 1 - V / V_convex_hull
  - bulge_amplitude     = max - median radius (mm)
  - tortuosity          = arclength(centerline) / euclidean(endpoints)
  - centerline_length   = total arclength
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pyvista as pv


@dataclass
class CenterlineProfile:
    """Discretized vessel centerline."""

    s: np.ndarray            # arclength parameter (n,)
    points: np.ndarray       # centerline xyz (n, 3)
    radius: np.ndarray       # local cross-section equivalent radius (n,)
    tangent: np.ndarray      # unit tangents (n, 3)


@dataclass
class Morphology:
    """Aggregated morphological descriptors of one case."""

    centerline: CenterlineProfile
    aspect_ratio: float
    size_ratio: float
    sphericity: float
    undulation_index: float
    bulge_amplitude: float
    tortuosity: float
    centerline_length: float
    surface_area: float
    volume: float
    bbox_extent: tuple[float, float, float]


def _principal_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)
    w, v = np.linalg.eigh(cov)
    # largest eigenvalue is principal axis
    axis = v[:, -1]
    if axis[1] < 0:  # consistent orientation (vessels run along +y in AnXplore)
        axis = -axis
    return centroid, axis


def estimate_centerline(surface: pv.PolyData, n_bins: int = 48) -> CenterlineProfile:
    """Approximate centerline by axial binning along the principal axis."""
    pts = np.asarray(surface.points)
    centroid, axis = _principal_axis(pts)

    # Project all points onto principal axis
    rel = pts - centroid
    proj = rel @ axis                                # scalar coordinate
    p_min, p_max = float(np.percentile(proj, 1)), float(np.percentile(proj, 99))
    edges = np.linspace(p_min, p_max, n_bins + 1)
    centers_proj = 0.5 * (edges[:-1] + edges[1:])

    bin_idx = np.clip(np.digitize(proj, edges) - 1, 0, n_bins - 1)

    # For each bin, compute centroid (xyz) and equivalent radius from cross-section.
    cl_pts = np.zeros((n_bins, 3))
    cl_r = np.zeros(n_bins)
    cl_count = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        mask = bin_idx == b
        if not np.any(mask):
            cl_pts[b] = centroid + axis * centers_proj[b]
            cl_r[b] = np.nan
            continue
        section = pts[mask]
        c = section.mean(axis=0)
        # equivalent radius = mean perpendicular distance from centroid in the
        # plane orthogonal to the principal axis
        section_rel = section - c
        perp = section_rel - np.outer(section_rel @ axis, axis)
        d = np.linalg.norm(perp, axis=1)
        if d.size:
            cl_r[b] = float(np.median(d))
        else:
            cl_r[b] = np.nan
        cl_pts[b] = c
        cl_count[b] = int(mask.sum())

    # interpolate any NaN bins linearly
    nan_mask = np.isnan(cl_r)
    if nan_mask.any() and (~nan_mask).sum() >= 2:
        idx = np.arange(n_bins)
        cl_r[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], cl_r[~nan_mask])

    # arclength along the discrete centerline
    seg = np.diff(cl_pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])

    # tangent vectors via central differences
    tang = np.zeros_like(cl_pts)
    if n_bins >= 3:
        tang[1:-1] = cl_pts[2:] - cl_pts[:-2]
        tang[0] = cl_pts[1] - cl_pts[0]
        tang[-1] = cl_pts[-1] - cl_pts[-2]
    norms = np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    tang = tang / norms

    return CenterlineProfile(s=s, points=cl_pts, radius=cl_r, tangent=tang)


def morphology(surface: pv.PolyData) -> Morphology:
    """Compute Morphology descriptors on a cleaned triangulated surface."""
    cl = estimate_centerline(surface)

    r = cl.radius.copy()
    r = r[~np.isnan(r)]
    if r.size == 0:
        r = np.array([1.0])

    # parent vessel proxy: median of the lower-radius half (away from any bulge)
    parent_r = float(np.median(np.sort(r)[: max(1, r.size // 2)]))
    max_r = float(np.max(r))
    min_r = float(np.min(r))

    aspect_ratio = max_r / max(min_r, 1e-3)
    size_ratio = max_r / max(parent_r, 1e-3)
    bulge_amp = max(0.0, max_r - parent_r)

    try:
        vol = float(surface.volume)
    except Exception:
        vol = 0.0
    area = float(surface.area)

    sphericity = 0.0
    if vol > 0 and area > 0:
        sphericity = float(np.cbrt(np.pi) * (6.0 * vol) ** (2.0 / 3.0) / area)
        sphericity = float(np.clip(sphericity, 0.0, 1.5))

    # undulation = 1 - V / V_convex_hull
    undulation = 0.0
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(np.asarray(surface.points))
        vh = float(hull.volume)
        if vh > vol > 0:
            undulation = float(1.0 - vol / vh)
            undulation = float(np.clip(undulation, 0.0, 0.95))
    except Exception:
        undulation = 0.0

    arc = float(cl.s[-1])
    end_dist = float(np.linalg.norm(cl.points[-1] - cl.points[0]))
    tortuosity = arc / max(end_dist, 1e-3)

    bb = surface.bounds
    extent = (
        float(bb[1] - bb[0]),
        float(bb[3] - bb[2]),
        float(bb[5] - bb[4]),
    )

    return Morphology(
        centerline=cl,
        aspect_ratio=aspect_ratio,
        size_ratio=size_ratio,
        sphericity=sphericity,
        undulation_index=undulation,
        bulge_amplitude=bulge_amp,
        tortuosity=tortuosity,
        centerline_length=arc,
        surface_area=area,
        volume=vol,
        bbox_extent=extent,
    )


def downsample_radius_profile(cl: CenterlineProfile, n: int = 24) -> List[float]:
    """Resample radius profile to fixed length for transport into the UI."""
    if cl.s[-1] <= 0:
        return [float(cl.radius.mean())] * n
    target = np.linspace(0.0, cl.s[-1], n)
    return list(np.interp(target, cl.s, cl.radius))
