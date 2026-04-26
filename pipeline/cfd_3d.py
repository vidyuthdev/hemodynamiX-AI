"""Steady incompressible Navier-Stokes on real AnXplore aneurysm anatomy.

Pipeline per case:
  1. Read AnXplore tetrahedral fluid volume (~944k tets).
  2. Extract surface, decimate to ~3-5k triangles, re-tetrahedralize with
     TetGen -> ~30k tets, ~7k vertices.
  3. Detect inlet / outlet / wall facets by region-growing the surface on
     normal-similarity (caps = large connected regions with near-uniform
     normal; the rest is wall).
  4. Solve steady Navier-Stokes with Taylor-Hood (P2 velocity / P1 pressure)
     in scikit-fem, Picard linearisation of the convective term.
  5. Compute wall shear stress vector field on the wall facets and
     aggregate to the canonical 6 hemodynamic features used everywhere
     else in the pipeline.

Memory budget on a 16 GB M4:
  ~33k tets -> ~190k DOFs -> ~3 GB peak during sparse LU factorisation,
  4-7 minutes per case end-to-end.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import skfem
import tetgen
from skfem.helpers import ddot, dot, grad, sym_grad
from skfem.assembly import BilinearForm, LinearForm

LOG = logging.getLogger("hxai.cfd_3d")

# Blood properties (CGS-friendly SI). Same as the Womersley path so the
# two solvers feed the ML stage with comparable numbers.
RHO = 1056.0          # kg / m^3
MU = 0.0035           # Pa.s

# Mean-cycle ICA flow rate (Holdsworth et al. 1999): ~ 2.0 mL/s = 2.0e-6
# m^3/s. We solve for the time-averaged steady state, so this is the right
# representative flow for clinical hemodynamic indices. Using peak-systolic
# (~4.5 mL/s) at Re ~ 800 pushes plain Picard out of its convergence basin
# without proper SUPG/GLS stabilisation.
Q_INLET = 2.0e-6      # m^3/s
DECIMATION_FRACTION = 0.95   # -> ~30-40k tets after re-meshing
PICARD_ITERS = 8
PICARD_TOL = 5e-3
PICARD_RELAX = 0.5           # under-relaxation factor on the velocity update
PSEUDO_DT = 0.005            # pseudo-timestep for rho/dt * (u - u_old) damping
ADV_LAMBDA_MAX = 0.85        # cap the advection ramp; full advection at this
                              # Pe_h is unstable on a 1mm mesh without SUPG.
SHARP_DIHEDRAL_DEG = 50.0    # edges with >= this angle separate caps from the wall
MIN_CAP_TRIS = 8             # ignore tiny splinters
CAP_PLANARITY_TOL = 0.06     # max (plane-residual / cap_radius)
MIN_CAP_RADIUS_MM = 0.4      # below this is geometric noise, not a real opening


# --------------------------------------------------------------------------
# 1) Mesh handling: decimate + tetgen + cap detection
# --------------------------------------------------------------------------


@dataclass
class CaseMesh:
    """Everything downstream needs to solve and post-process one case.

    All facet / node indices refer to the *metric* skfem mesh (mm has been
    rescaled to metres) so the solver can use them directly with no
    re-classification.
    """

    skfem_mesh: skfem.MeshTet           # in metres
    inlet_facets: np.ndarray
    outlet_facets: np.ndarray
    wall_facets: np.ndarray
    inlet_nodes: np.ndarray
    outlet_nodes: np.ndarray
    wall_nodes: np.ndarray
    inlet_normal: np.ndarray
    inlet_centroid: np.ndarray          # metres
    inlet_radius: float                 # metres
    decimated_surface: pv.PolyData      # in mm, cached for rendering


def _extract_decimated_surface(volume_path: Path) -> pv.PolyData:
    raw = pv.read(str(volume_path))
    surf = raw.extract_surface(algorithm="dataset_surface").triangulate().clean()
    dec = surf.decimate(DECIMATION_FRACTION).clean().triangulate()
    return dec


def _tetrahedralize(surface: pv.PolyData) -> Tuple[np.ndarray, np.ndarray]:
    tg = tetgen.TetGen(surface)
    tg.tetrahedralize(mindihedral=18, minratio=1.4, quality=True)
    return np.asarray(tg.node), np.asarray(tg.elem)


def _surface_with_normals(surface: pv.PolyData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = surface.compute_normals(
        point_normals=False, cell_normals=True, auto_orient_normals=True, consistent_normals=True
    )
    normals = np.asarray(s.cell_data["Normals"], dtype=float)
    tri = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
    centroids = surface.points[tri].mean(1)
    return tri, normals, centroids


def _build_tri_adjacency_smooth(
    tri: np.ndarray, normals: np.ndarray, cos_dihedral_cut: float,
) -> List[List[int]]:
    """Triangle adjacency that DROPS edges where the dihedral angle is sharper
    than the cut (default 50 deg). The remaining graph splits the surface
    into smooth pieces -- the wall becomes one big component, and each
    discrete cap becomes its own small component.
    """
    edge_to_tri: Dict[Tuple[int, int], List[int]] = {}
    for ti, t in enumerate(tri):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            key = (a, b) if a < b else (b, a)
            edge_to_tri.setdefault(key, []).append(ti)
    adj: List[List[int]] = [[] for _ in range(len(tri))]
    for ts in edge_to_tri.values():
        if len(ts) != 2:
            continue
        n1, n2 = normals[ts[0]], normals[ts[1]]
        if (n1 @ n2) >= cos_dihedral_cut:
            adj[ts[0]].append(ts[1])
            adj[ts[1]].append(ts[0])
    return adj


def _detect_caps(surface: pv.PolyData) -> List[Dict]:
    """Split the surface at sharp creases (~90 deg cap-to-wall edges) and
    return the connected components that are (a) NOT the wall and (b)
    geometrically planar discs.
    """
    tri, normals, _ = _surface_with_normals(surface)
    pts = surface.points
    n_tri = tri.shape[0]
    cos_dihedral = float(np.cos(np.deg2rad(SHARP_DIHEDRAL_DEG)))
    adj = _build_tri_adjacency_smooth(tri, normals, cos_dihedral)

    seen = np.zeros(n_tri, dtype=bool)
    components: List[List[int]] = []
    for seed in range(n_tri):
        if seen[seed]:
            continue
        stack = [seed]
        members: List[int] = []
        while stack:
            t = stack.pop()
            if seen[t]:
                continue
            seen[t] = True
            members.append(t)
            for nbr in adj[t]:
                if not seen[nbr]:
                    stack.append(nbr)
        components.append(members)
    components.sort(key=len, reverse=True)

    # Largest component is the curved vessel wall; everything else is a candidate cap.
    LOG.info(
        "surface components (after dihedral cut): top sizes = %s",
        [len(c) for c in components[:8]],
    )
    candidates = components[1:]

    caps: List[Dict] = []
    for members in candidates:
        if len(members) < MIN_CAP_TRIS:
            continue
        members_arr = np.asarray(members)
        verts = np.unique(tri[members_arr].ravel())
        cap_pts = pts[verts]
        cap_center = cap_pts.mean(0)
        cap_normal = normals[members_arr].mean(0)
        cap_normal /= np.linalg.norm(cap_normal) + 1e-12

        rel = cap_pts - cap_center
        plane_residual = float(np.std(rel @ cap_normal))
        radial = rel - (rel @ cap_normal)[:, None] * cap_normal
        radius = float(np.linalg.norm(radial, axis=1).max())
        if radius < MIN_CAP_RADIUS_MM:
            continue
        if plane_residual / radius > CAP_PLANARITY_TOL:
            continue

        caps.append(
            {
                "tri_idx": members_arr,
                "vert_idx": verts,
                "centroid": cap_center,
                "normal": cap_normal,
                "radius": radius,
                "n_tri": len(members),
                "planarity": plane_residual / radius,
            }
        )

    if not caps:
        raise RuntimeError("No caps detected on the decimated surface")
    return caps


def _classify_inlet_outlet(caps: List[Dict]) -> Tuple[Dict, List[Dict]]:
    """Pick the cap with the largest cross-section as the inlet.

    Inlet flow direction == cap normal pointing INTO the fluid domain. We
    flip the normal to ensure the dot product with (centroid_domain -
    centroid_cap) is positive.
    """
    if len(caps) < 2:
        raise RuntimeError(f"Need >= 2 caps, found {len(caps)}")
    # Largest radius = parent artery in idealized cerebral geometries
    inlet = max(caps, key=lambda c: c["radius"])
    outlets = [c for c in caps if c is not inlet]
    return inlet, outlets


def _flip_inward(cap: Dict, all_pts: np.ndarray) -> np.ndarray:
    """Flip cap normal so it points into the bulk of the fluid domain."""
    bulk_dir = all_pts.mean(0) - cap["centroid"]
    n = cap["normal"]
    if n @ bulk_dir < 0:
        n = -n
    return n / (np.linalg.norm(n) + 1e-12)


def _classify_facets_geometric(
    skf: skfem.MeshTet, caps_m: List[Dict],
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Bucket every boundary facet into a cap or the wall by geometric distance
    to the cap planes. Robust to Steiner points inserted by tetgen.

    A facet belongs to cap k if its centroid is within (1.10 * cap_radius)
    of the cap centroid AND its perpendicular distance to the cap plane is
    less than (0.20 * cap_radius). caps_m must be in metres.
    """
    bnd_facets = skf.boundary_facets()
    # facet centroids in metres
    fc = skf.p[:, skf.facets[:, bnd_facets]].mean(axis=1).T   # (n_bnd, 3)

    assigned = -np.ones(bnd_facets.size, dtype=int)            # -1 = wall
    for k, c in enumerate(caps_m):
        rel = fc - c["centroid"]
        d_plane = np.abs(rel @ c["normal"])
        d_radial = np.linalg.norm(
            rel - (rel @ c["normal"])[:, None] * c["normal"], axis=1
        )
        mask = (d_radial <= 1.10 * c["radius"]) & (d_plane <= 0.20 * c["radius"])
        # Don't overwrite a facet already claimed by an earlier (larger) cap
        new = mask & (assigned == -1)
        assigned[new] = k

    cap_facets = [bnd_facets[assigned == k] for k in range(len(caps_m))]
    wall_facets = bnd_facets[assigned == -1]
    return wall_facets, cap_facets, bnd_facets


def build_case_mesh(volume_path: Path) -> CaseMesh:
    """Decimate, tetrahedralise, build skfem mesh, classify boundaries."""
    t0 = time.time()
    surf = _extract_decimated_surface(volume_path)
    LOG.info("decimated surface: %d tris (%.2fs)", surf.n_cells, time.time() - t0)

    t0 = time.time()
    nodes_mm, elems = _tetrahedralize(surf)
    LOG.info("tetgen: %d tets (%.2fs)", len(elems), time.time() - t0)

    # Detect caps on the SURFACE (mm)
    caps = _detect_caps(surf)
    inlet, outlets = _classify_inlet_outlet(caps)
    inlet_normal = _flip_inward(inlet, nodes_mm)
    LOG.info(
        "caps detected=%d -> inlet r=%.2fmm (n_tri=%d), outlets=%s",
        len(caps), inlet["radius"], inlet["n_tri"],
        [(round(o["radius"], 2), o["n_tri"]) for o in outlets],
    )

    # Build skfem mesh in METRES (so derivatives have SI units).
    nodes_m = nodes_mm * 1e-3
    skf = skfem.MeshTet(nodes_m.T.copy(), elems.T.copy())

    # Move cap descriptors into metres
    caps_m = [
        {**c, "centroid": c["centroid"] * 1e-3, "radius": c["radius"] * 1e-3, "normal": c["normal"]}
        for c in caps
    ]
    # caps.index() compares with == which broadcasts over numpy arrays inside
    # the cap dicts; use object identity instead.
    inlet_idx = next(i for i, c in enumerate(caps) if c is inlet)
    outlet_idx = [next(i for i, c in enumerate(caps) if c is o) for o in outlets]

    wall_facets, cap_facets_list, bnd_facets = _classify_facets_geometric(skf, caps_m)
    inlet_facets = cap_facets_list[inlet_idx]
    outlet_facets = (
        np.concatenate([cap_facets_list[i] for i in outlet_idx])
        if outlet_idx else np.array([], dtype=np.int64)
    )

    if inlet_facets.size == 0:
        raise RuntimeError("Inlet cap matched zero boundary facets (geometry too coarse?)")

    # Build node sets for downstream BCs and pressure pin
    inlet_nodes = np.unique(skf.facets[:, inlet_facets].ravel())
    outlet_nodes = np.unique(skf.facets[:, outlet_facets].ravel()) if outlet_facets.size else np.array([], dtype=np.int64)
    cap_node_set = set(inlet_nodes.tolist()) | set(outlet_nodes.tolist())
    wall_nodes = np.unique(skf.facets[:, wall_facets].ravel())
    wall_nodes = np.array([n for n in wall_nodes if n not in cap_node_set], dtype=np.int64)

    LOG.info(
        "BCs (geometric mapping): inlet_facets=%d outlet_facets=%d wall_facets=%d "
        "/ nodes inlet=%d outlet=%d wall=%d",
        inlet_facets.size, outlet_facets.size, wall_facets.size,
        inlet_nodes.size, outlet_nodes.size, wall_nodes.size,
    )

    return CaseMesh(
        skfem_mesh=skf,
        inlet_facets=inlet_facets,
        outlet_facets=outlet_facets,
        wall_facets=wall_facets,
        inlet_nodes=inlet_nodes,
        outlet_nodes=outlet_nodes,
        wall_nodes=wall_nodes,
        inlet_normal=inlet_normal,
        inlet_centroid=inlet["centroid"] * 1e-3,
        inlet_radius=inlet["radius"] * 1e-3,
        decimated_surface=surf,
    )


# --------------------------------------------------------------------------
# 2) Steady Navier-Stokes (Taylor-Hood, Picard)
# --------------------------------------------------------------------------


def _parabolic_profile(
    nodes_xyz: np.ndarray, centroid: np.ndarray, normal: np.ndarray, radius_m: float, q: float
) -> np.ndarray:
    """Hagen-Poiseuille parabolic inlet velocity (m/s).

    nodes_xyz is in metres, centroid in metres, radius_m in metres.
    Output points INTO the domain along (-normal).
    """
    u_mean = q / (np.pi * radius_m * radius_m)
    u_max = 2.0 * u_mean
    rel = nodes_xyz - centroid
    rel_perp = rel - (rel @ normal)[:, None] * normal
    r2 = np.sum(rel_perp ** 2, axis=1)
    r2_norm = np.clip(1.0 - r2 / max(radius_m * radius_m, 1e-12), 0.0, 1.0)
    speed = u_max * r2_norm
    return -speed[:, None] * normal[None, :]


def _solve_stokes_picard(
    case: CaseMesh, q_inlet: float = Q_INLET, with_convection: bool = True,
) -> Tuple[np.ndarray, np.ndarray, skfem.MeshTet, skfem.Basis, skfem.Basis]:
    """Solve steady incompressible Navier-Stokes. Stokes-only when with_convection=False."""
    mesh_m = case.skfem_mesh

    Vh = skfem.Basis(mesh_m, skfem.ElementVector(skfem.ElementTetP2()))
    Qh = Vh.with_element(skfem.ElementTetP1())
    LOG.info("system size: V=%d  Q=%d  total=%d", Vh.N, Qh.N, Vh.N + Qh.N)

    @BilinearForm
    def a_diff(u, v, w):
        return MU * ddot(grad(u), grad(v))

    @BilinearForm
    def b_div(u, q, w):
        gu = grad(u)
        return -q * (gu[0, 0] + gu[1, 1] + gu[2, 2])

    @BilinearForm
    def a_mass(u, v, w):
        # vector mass matrix: rho * u . v (used for pseudo-timestepping)
        return RHO * (u[0] * v[0] + u[1] * v[1] + u[2] * v[2])

    @BilinearForm
    def a_conv(u, v, w):
        u_old = w["u_old"]
        gu = grad(u)
        cu0 = u_old[0] * gu[0, 0] + u_old[1] * gu[0, 1] + u_old[2] * gu[0, 2]
        cu1 = u_old[0] * gu[1, 0] + u_old[1] * gu[1, 1] + u_old[2] * gu[1, 2]
        cu2 = u_old[0] * gu[2, 0] + u_old[1] * gu[2, 1] + u_old[2] * gu[2, 2]
        return RHO * (cu0 * v[0] + cu1 * v[1] + cu2 * v[2])

    @LinearForm
    def f_mass_old(v, w):
        u_old = w["u_old"]
        return (RHO / PSEUDO_DT) * (u_old[0] * v[0] + u_old[1] * v[1] + u_old[2] * v[2])

    A_diff = skfem.asm(a_diff, Vh, Vh)
    A_mass = skfem.asm(a_mass, Vh, Vh)
    B = skfem.asm(b_div, Vh, Qh)

    # Inlet + wall Dirichlet DOFs
    inlet_dofs = Vh.get_dofs(facets=case.inlet_facets).all()
    wall_dofs = Vh.get_dofs(facets=case.wall_facets).all()
    if inlet_dofs.size == 0:
        raise RuntimeError("Inlet DOF set is empty (cap detection failure?)")

    # Parabolic inlet velocity. ElementVector(ElementTetP2) DOFs are
    # interleaved per node (x,y,z,x,y,z,...) so component = dof_idx % 3.
    v_dir = np.zeros(Vh.N)
    inlet_xyz = Vh.doflocs[:, inlet_dofs].T
    u_at_dof = _parabolic_profile(
        inlet_xyz, case.inlet_centroid, case.inlet_normal, case.inlet_radius, q_inlet
    )
    comps = inlet_dofs % 3
    v_dir[inlet_dofs] = u_at_dof[np.arange(inlet_dofs.size), comps]
    u_max = float(np.linalg.norm(u_at_dof, axis=1).max())
    re_d = RHO * u_max * (2.0 * case.inlet_radius) / MU
    LOG.info("inlet u_max=%.3f m/s, Re_diameter=%.0f", u_max, re_d)

    # Pressure pin at one outlet node to fix the constant pressure nullspace
    p_pin = int(case.outlet_nodes[0])

    # Solver loop:
    #   it = 0      : Stokes -> initial guess
    #   it = 1..N   : pseudo-time-stepped Picard with Reynolds continuation
    #                 (advection ramped from 0 to 1 over the first half of
    #                 iterations to keep the problem inside Picard's basin).
    # Solver loop:
    #   it = 0       : Stokes -> initial guess
    #   it = 1..N    : pseudo-time-stepped Picard with Reynolds continuation,
    #                  advection ramped from 0 -> ADV_LAMBDA_MAX over ~half
    #                  the iterations. Final iterate = best stable iterate.
    u_coef = np.zeros(Vh.N)
    p_coef = np.zeros(Qh.N)
    best_u = None
    best_p = None
    best_resid = np.inf

    iters = PICARD_ITERS if with_convection else 1
    ramp_iters = max(3, iters // 2)

    for it in range(iters):
        if it == 0:
            K = sp.bmat([[A_diff, B.T], [B, None]], format="csr")
            f = np.zeros(Vh.N + Qh.N)
            lam = 0.0
        else:
            lam = min(ADV_LAMBDA_MAX, it / ramp_iters * ADV_LAMBDA_MAX)
            A_conv = skfem.asm(a_conv, Vh, Vh, u_old=Vh.interpolate(u_coef))
            A_total = A_diff + lam * A_conv + (1.0 / PSEUDO_DT) * A_mass
            K = sp.bmat([[A_total, B.T], [B, None]], format="csr")
            f = np.zeros(Vh.N + Qh.N)
            f[: Vh.N] = skfem.asm(f_mass_old, Vh, u_old=Vh.interpolate(u_coef))

        D_vel = np.concatenate([inlet_dofs, wall_dofs])
        D_full = np.concatenate([D_vel, np.array([Vh.N + p_pin])])
        x = np.zeros(K.shape[0])
        x[D_vel] = v_dir[D_vel]

        x_new = skfem.solve(*skfem.condense(K, f, x=x, D=D_full))
        u_new = x_new[: Vh.N]
        p_new = x_new[Vh.N:]

        u_max_now = float(np.linalg.norm(u_new.reshape(-1, 3), axis=1).max())
        if it > 0:
            denom = float(np.linalg.norm(u_new) + 1e-12)
            r = float(np.linalg.norm(u_new - u_coef) / denom)
            LOG.info(
                "Picard %d (lambda=%.2f): rel diff = %.4f, u_max=%.3f m/s",
                it, lam, r, u_max_now,
            )

            # Sanity bound: if the new iterate's u_max is grossly out of the
            # physical envelope (10 x the inlet peak), reject and halve relax.
            if u_max_now > 10.0 * u_max:
                LOG.warning(
                    "Picard %d unstable (u_max=%.1f > 10x inlet); rejecting iterate", it, u_max_now,
                )
                # mild blend so we move slightly without diverging
                u_coef = 0.9 * u_coef + 0.1 * u_new
                p_coef = 0.9 * p_coef + 0.1 * p_new
                continue

            u_coef = (1.0 - PICARD_RELAX) * u_coef + PICARD_RELAX * u_new
            p_coef = (1.0 - PICARD_RELAX) * p_coef + PICARD_RELAX * p_new

            if lam >= ADV_LAMBDA_MAX - 1e-9 and r < best_resid:
                best_resid = r
                best_u = u_coef.copy()
                best_p = p_coef.copy()
            if lam >= ADV_LAMBDA_MAX - 1e-9 and r < PICARD_TOL:
                break
        else:
            u_coef, p_coef = u_new, p_new
            LOG.info("Stokes init: u_max=%.3f m/s", u_max_now)

    if best_u is not None:
        u_coef, p_coef = best_u, best_p
        LOG.info("returning best iterate (residual=%.4f)", best_resid)
    return u_coef, p_coef, mesh_m, Vh, Qh


# --------------------------------------------------------------------------
# 3) WSS + feature aggregation
# --------------------------------------------------------------------------


def compute_wss(
    u_coef: np.ndarray,
    case: CaseMesh,
    mesh_m: skfem.MeshTet,
    Vh: skfem.Basis,
) -> np.ndarray:
    """Magnitude of wall shear stress on each wall facet (Pa).

    tau_w = mu * (du_t / dn) where t is the wall-tangential direction.
    Computed as mu * |(eps . n) - (n . eps . n) n|, averaged over each
    facet's quadrature points.
    """
    wall = case.wall_facets
    if wall.size == 0:
        return np.zeros(0)

    fb = skfem.FacetBasis(mesh_m, Vh.elem, facets=wall)
    field = fb.interpolate(u_coef)
    grads = field.grad                                 # (3, 3, n_facets, n_q)
    eps = 0.5 * (grads + np.transpose(grads, (1, 0, 2, 3)))

    n = fb.normals if hasattr(fb, "normals") else fb.n  # (3, n_facets, n_q)
    if n.ndim == 2:                                     # (3, n_facets) -> add q axis
        n = n[:, :, None]

    en = np.einsum("ijFQ,jFQ->iFQ", eps, n)             # (3, F, Q)
    nen = np.einsum("iFQ,iFQ->FQ", n, en)               # (F, Q)
    tau_vec = en - n * nen[None, :, :]                  # (3, F, Q)
    tau_mag = MU * np.linalg.norm(tau_vec, axis=0)      # (F, Q)
    return tau_mag.mean(axis=1)                         # (F,)


def aggregate_features(
    u_coef: np.ndarray,
    p_coef: np.ndarray,
    wss_per_facet: np.ndarray,
    case: CaseMesh,
    mesh_m: skfem.MeshTet,
    Vh: skfem.Basis,
    Qh: skfem.Basis,
    morphology: Dict[str, float],
) -> Dict[str, float]:
    """Reduce 3-D fields to the canonical 6-feature vector used by the ML stage."""
    # Velocity field at all P2 nodes (not just vertices). Dim ordering: per-node triplets.
    u_node = u_coef.reshape(-1, 3)
    speed = np.linalg.norm(u_node, axis=1)
    # 90th percentile to match the Womersley convention
    velocity = float(np.percentile(speed, 90))

    # Pressure at vertex DOFs (mmHg)
    p_vertex = p_coef
    p_mmHg = p_vertex / 133.322
    # Reference: zero at the pinned outlet node; report mean dynamic pressure
    pressure = float(np.mean(np.abs(p_mmHg)))

    # WSS aggregates
    if wss_per_facet.size == 0:
        tawss = 0.0
    else:
        tawss = float(np.mean(wss_per_facet))
    # Steady solver -> OSI is undefined; estimate it from morphology only:
    # high size ratio + bulge amplitude correlate with high oscillation in time-resolved CFD.
    osi_proxy = float(
        np.clip(0.05 + 0.18 * np.tanh(morphology.get("bulgeAmplitude", 0.5) / 1.2), 0.0, 0.5)
    )
    rrt = float(min(25.0, 1.0 / max((1.0 - 2.0 * osi_proxy) * max(tawss, 1e-3), 1e-3)))

    # Vorticity from velocity curl at sample points
    # Sample interior cells, compute curl via FE basis
    cell_basis = skfem.CellBasis(mesh_m, Vh.elem)
    grads = cell_basis.interpolate(u_coef).grad   # (3,3,Q,T)
    curl_x = grads[2, 1] - grads[1, 2]
    curl_y = grads[0, 2] - grads[2, 0]
    curl_z = grads[1, 0] - grads[0, 1]
    curl_mag = np.sqrt(curl_x ** 2 + curl_y ** 2 + curl_z ** 2)   # (Q, T)
    vorticity = float(np.percentile(curl_mag.ravel(), 90))

    return {
        "tawss": tawss,
        "osi": osi_proxy,
        "rrt": rrt,
        "vorticity": vorticity,
        "velocity": velocity,
        "pressure": pressure,
    }


# --------------------------------------------------------------------------
# 4) Public per-case entry point
# --------------------------------------------------------------------------


@dataclass
class CFD3DResult:
    case_id: str
    features: Dict[str, float]
    wss_per_facet: np.ndarray
    case_mesh: CaseMesh
    velocity_coef: np.ndarray
    pressure_coef: np.ndarray
    mesh_m: skfem.MeshTet
    Vh: skfem.Basis
    Qh: skfem.Basis


def solve_case(case_id: str, volume_path: Path, morphology: Dict[str, float]) -> CFD3DResult:
    LOG.info("=== %s ===  %s", case_id, volume_path.name)
    t_total = time.time()
    case = build_case_mesh(volume_path)
    u, p, mesh_m, Vh, Qh = _solve_stokes_picard(case)
    wss = compute_wss(u, case, mesh_m, Vh)
    feats = aggregate_features(u, p, wss, case, mesh_m, Vh, Qh, morphology)
    LOG.info(
        "%s done in %.1fs  tawss=%.3f  vel=%.3f  vort=%.1f",
        case_id, time.time() - t_total, feats["tawss"], feats["velocity"], feats["vorticity"],
    )
    return CFD3DResult(
        case_id=case_id,
        features=feats,
        wss_per_facet=wss,
        case_mesh=case,
        velocity_coef=u,
        pressure_coef=p,
        mesh_m=mesh_m,
        Vh=Vh,
        Qh=Qh,
    )
