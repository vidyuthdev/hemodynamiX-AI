"""Off-screen PyVista renderer for 3-D CFD results.

Produces the headline image: aneurysm wall coloured by WSS, with seeded
streamlines threading the lumen and inlet velocity glyphs. 1920x1440 PNG
suitable for posters, papers, and the React UI inspector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
import skfem

from .cfd_3d import CFD3DResult, CaseMesh

LOG = logging.getLogger("hxai.render")

# Off-screen rendering must be enabled before any Plotter is created.
pv.set_plot_theme("document")


def _wall_polydata(case: CaseMesh, wss: np.ndarray) -> pv.PolyData:
    """Build a triangular surface mesh of the no-slip wall, painted with
    per-facet WSS in Pa. Vertex coordinates are in millimetres so we don't
    fight the camera at the metres scale."""
    skf = case.skfem_mesh
    pts_mm = skf.p.T * 1e3
    facets = skf.facets[:, case.wall_facets].T   # (n_wall, 3)

    pd_faces = np.full((facets.shape[0], 4), 3, dtype=np.int64)
    pd_faces[:, 1:] = facets
    surf = pv.PolyData(pts_mm, pd_faces.ravel())
    surf.cell_data["WSS_Pa"] = wss.astype(np.float32)
    return surf


def _velocity_field(case: CaseMesh, u_coef: np.ndarray) -> pv.UnstructuredGrid:
    """Volume mesh with point-wise velocity in mm-coords. We keep velocity
    components in m/s so streamlines/glyphs can be quantitatively compared.
    """
    skf = case.skfem_mesh
    pts_mm = skf.p.T * 1e3
    n_pts = pts_mm.shape[0]
    # Vh has interleaved (x,y,z,...) DOFs at vertices+edge-midpoints.
    # The first n_pts triplets correspond to vertex DOFs (P2 numbers vertex
    # DOFs first, then edge DOFs). Take those.
    u_vert = u_coef[: 3 * n_pts].reshape(n_pts, 3)

    cells = skf.t.T            # (n_cells, 4)
    grid_cells = np.hstack([np.full((cells.shape[0], 1), 4, dtype=np.int64), cells]).ravel()
    cell_types = np.full(cells.shape[0], pv.CellType.TETRA, dtype=np.uint8)
    ug = pv.UnstructuredGrid(grid_cells, cell_types, pts_mm)
    ug.point_data["velocity_m_s"] = u_vert.astype(np.float32)
    ug.point_data["speed_m_s"] = np.linalg.norm(u_vert, axis=1).astype(np.float32)
    return ug


def _seed_inlet_points(
    inlet_centroid_mm: np.ndarray,
    inlet_normal: np.ndarray,
    inlet_radius_mm: float,
    n_seeds: int = 50,
    rng_seed: int = 42,
) -> np.ndarray:
    """Concentric-ring seed pattern on the inlet plane (slightly inside the
    cap so the streamline integrator picks up the velocity field cleanly)."""
    n = inlet_normal / (np.linalg.norm(inlet_normal) + 1e-12)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, helper); e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(n, e1);     e2 /= np.linalg.norm(e2) + 1e-12

    rng = np.random.default_rng(rng_seed)
    pts: list = []
    radii = np.linspace(0.15 * inlet_radius_mm, 0.85 * inlet_radius_mm, 5)
    per_ring = max(1, n_seeds // len(radii))
    for r in radii:
        for k in range(per_ring):
            theta = 2.0 * np.pi * (k + rng.uniform(0.0, 1.0)) / per_ring
            pts.append(inlet_centroid_mm + r * (np.cos(theta) * e1 + np.sin(theta) * e2))
    # Step a small distance INTO the domain (inlet_normal already points
    # INWARD per build_case_mesh -> _flip_inward).
    seed_offset = n * 0.10 * inlet_radius_mm
    return np.asarray(pts) + seed_offset


def render_case(
    res: CFD3DResult,
    out_path: Path,
    width: int = 1920,
    height: int = 1440,
) -> Path:
    """Save a 4K-quality PNG of the WSS-coloured wall + streamlines."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wall_surf = _wall_polydata(res.case_mesh, res.wss_per_facet)
    volume = _velocity_field(res.case_mesh, res.velocity_coef)

    inlet_centroid_mm = res.case_mesh.inlet_centroid * 1e3
    inlet_radius_mm = res.case_mesh.inlet_radius * 1e3
    diag_mm = float(np.linalg.norm(np.asarray(wall_surf.bounds[1::2]) - np.asarray(wall_surf.bounds[::2])))

    # Push the seed disc well INTO the domain (one inlet diameter inward) so
    # we are clear of the boundary's zero-velocity layer; tighten the source
    # radius so we sit in the parabolic core where speed is highest.
    seed_offset_mm = 1.0 * inlet_radius_mm
    seed_center = tuple(
        inlet_centroid_mm + seed_offset_mm * res.case_mesh.inlet_normal
    )
    try:
        streamlines = volume.streamlines(
            "velocity_m_s",
            source_radius=0.55 * inlet_radius_mm,
            source_center=seed_center,
            n_points=80,
            integration_direction="both",
            max_steps=20000,
            initial_step_length=0.20,
            terminal_speed=1e-6,
            compute_vorticity=False,
        )
    except Exception as exc:                              # pragma: no cover - PyVista variability
        LOG.warning("streamlines failed (%s); rendering wall only", exc)
        streamlines = pv.PolyData()

    # Inlet seed glyphs: a small disc at the inlet centroid for visual cue
    seed_disc = _seed_inlet_points(
        inlet_centroid_mm, res.case_mesh.inlet_normal, inlet_radius_mm, n_seeds=24
    )
    seed_pd = pv.PolyData(seed_disc)

    # Wall colour range: clip to 5-95 percentile so the variation is visible
    # (a few stagnation cells can squish the linear scale otherwise).
    wss = res.wss_per_facet
    lo, hi = float(np.percentile(wss, 5)), float(np.percentile(wss, 95))
    if hi - lo < 0.05:
        lo, hi = float(wss.min()), float(wss.max())

    plotter = pv.Plotter(off_screen=True, window_size=(width, height))
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    plotter.enable_depth_peeling(number_of_peels=8)

    # Streamlines first so they sit "inside" and are blended through the wall
    if hasattr(streamlines, "n_points") and streamlines.n_points > 0:
        tube_radius = max(0.05, 0.0035 * diag_mm)
        plotter.add_mesh(
            streamlines.tube(radius=tube_radius),
            scalars="speed_m_s",
            cmap="cool",
            scalar_bar_args={
                "title": "|u|  [m/s]",
                "title_font_size": 24,
                "label_font_size": 20,
                "n_labels": 4,
                "color": "black",
                "fmt": "%.2f",
                "vertical": True,
                "position_x": 0.04,
                "position_y": 0.10,
                "width": 0.03,
                "height": 0.45,
            },
        )
        LOG.info("streamlines: %d points across %d lines", streamlines.n_points, streamlines.n_cells)
    else:
        LOG.warning("no streamlines generated for %s", res.case_id)

    plotter.add_mesh(
        wall_surf,
        scalars="WSS_Pa",
        cmap="inferno",
        clim=(lo, hi),
        opacity=0.45,
        smooth_shading=True,
        ambient=0.35,
        diffuse=0.80,
        specular=0.30,
        scalar_bar_args={
            "title": "WSS  [Pa]",
            "title_font_size": 30,
            "label_font_size": 24,
            "n_labels": 5,
            "color": "black",
            "fmt": "%.2f",
            "vertical": True,
            "position_x": 0.92,
            "position_y": 0.10,
            "width": 0.03,
            "height": 0.80,
        },
    )

    plotter.add_mesh(
        seed_pd.glyph(orient=False, scale=False, geom=pv.Sphere(radius=max(0.08, 0.0035 * diag_mm))),
        color="#00d4ff",
        show_scalar_bar=False,
    )

    # Camera: oblique view, zoomed slightly past the bounding box
    bounds = wall_surf.bounds
    cx = 0.5 * (bounds[0] + bounds[1])
    cy = 0.5 * (bounds[2] + bounds[3])
    cz = 0.5 * (bounds[4] + bounds[5])
    diag = np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
    plotter.camera_position = [
        (cx + 1.4 * diag, cy + 0.5 * diag, cz + 0.8 * diag),
        (cx, cy, cz),
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.zoom(1.10)

    u_max = float(np.linalg.norm(res.velocity_coef.reshape(-1, 3), axis=1).max())
    plotter.add_text(
        f"{res.case_id}   TAWSS = {res.features['tawss']:.2f} Pa    |u|max = {u_max:.2f} m/s",
        position="upper_left",
        font_size=18,
        color="black",
    )

    plotter.screenshot(str(out_path), transparent_background=False)
    plotter.close()
    LOG.info("rendered %s -> %s (%.0f KB)", res.case_id, out_path, out_path.stat().st_size / 1024)
    return out_path
