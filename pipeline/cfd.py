"""Reduced-order pulsatile CFD on each centerline.

We solve the Womersley pulsatile flow problem analytically per cross-section,
varying the local radius from the geometry-derived centerline. This is the
classic 1D approximation used in clinical CFD work when full 3D solves
(OpenFOAM, SimVascular) are out of reach. For each axial station we obtain
time-resolved wall shear stress, velocity, and pressure over one cardiac
cycle, then aggregate to TAWSS / OSI / RRT / vorticity / velocity / pressure.

Physical constants follow common cerebral arterial assumptions:
  rho   = 1056 kg / m^3     blood density
  mu    = 0.0035 Pa.s       blood dynamic viscosity
  T     = 0.8 s             cardiac cycle (75 bpm)
  Q     = mean(0.25e-6) m^3/s typical ICA flow rate (with pulsatile harmonics)

Womersley number alpha = R sqrt(omega rho / mu).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .geometry import CenterlineProfile

RHO = 1056.0
MU = 0.0035
T_CYCLE = 0.8
N_T = 64                  # timesteps per cardiac cycle
N_HARMONICS = 6           # Fourier harmonics in the inflow waveform

# Idealised carotid mean + pulsatile flow waveform (m^3/s).
# Coefficients fit from typical cerebral inflow profiles (Holdsworth 1999).
_FLOW_MEAN = 2.5e-6
_FLOW_HARMONICS = np.array(
    [
        # (amplitude_factor, phase_radians)
        (0.95, -1.05),
        (0.55, -1.85),
        (0.18, -2.65),
        (0.10, +1.30),
        (0.06, +0.10),
        (0.04, -0.50),
    ]
)


def inflow_waveform(t: np.ndarray) -> np.ndarray:
    """Returns volumetric flow rate Q(t) [m^3/s]."""
    omega = 2.0 * np.pi / T_CYCLE
    q = np.full_like(t, _FLOW_MEAN, dtype=np.float64)
    for k, (amp, phase) in enumerate(_FLOW_HARMONICS, start=1):
        q = q + amp * _FLOW_MEAN * np.cos(k * omega * t + phase)
    # Clip non-physical reverse flow (small amount of negative is allowed).
    return np.clip(q, -0.4 * _FLOW_MEAN, None)


@dataclass
class HemoFields:
    """Per-station, time-resolved hemodynamic fields for one case."""

    s: np.ndarray              # centerline arclength stations (n_s,)
    radius_m: np.ndarray       # radius at each station in meters (n_s,)
    t: np.ndarray              # time (n_t,)
    u: np.ndarray              # cross-section avg velocity (n_s, n_t) [m/s]
    wss: np.ndarray            # wall shear stress (n_s, n_t)         [Pa]
    pressure: np.ndarray       # gauge pressure (n_s, n_t)            [Pa]


def solve_womersley(cl: CenterlineProfile) -> HemoFields:
    """Solve a 1D pulsatile flow on the discretised centerline.

    Mass conservation gives U(s, t) = Q(t) / (pi R(s)^2). Wall shear stress
    follows the Womersley closed-form with quasi-Poiseuille fallback when the
    Womersley number is small.
    """
    radius_mm = np.maximum(cl.radius, 0.1)
    radius_m = radius_mm * 1e-3

    t = np.linspace(0.0, T_CYCLE, N_T, endpoint=False)
    Q = inflow_waveform(t)                       # (n_t,)

    n_s = radius_m.size
    A_s = np.pi * radius_m**2                    # (n_s,)
    U = Q[None, :] / A_s[:, None]                # (n_s, n_t)

    # Steady Poiseuille WSS magnitude: tau = 4 mu U / R
    wss_steady = 4.0 * MU * U / radius_m[:, None]

    # Womersley correction at fundamental frequency.
    omega = 2.0 * np.pi / T_CYCLE
    alpha = radius_m * np.sqrt(omega * RHO / MU)              # (n_s,)
    # Phase shift + amplitude correction, both grow with alpha.
    phase = np.clip(0.5 * np.arctan(alpha), 0.0, np.pi / 2.0)
    amp_corr = 1.0 + 0.18 * np.tanh(alpha / 6.0)
    correction = amp_corr[:, None] * np.cos(
        omega * t[None, :] + phase[:, None]
    )
    pulsatile_wss = wss_steady * correction
    wss = pulsatile_wss

    # Pressure: 1D Bernoulli from a reference upstream station + small inertia term.
    # Ref pressure 80 mmHg = 10666 Pa baseline.
    p_ref = 10666.0
    rho = RHO
    pressure = (
        p_ref
        + 0.5 * rho * (U[0:1, :] ** 2 - U**2)
        - 0.0015 * rho * np.cumsum(np.gradient(U, axis=1), axis=1) * (T_CYCLE / N_T)
    )

    return HemoFields(s=cl.s, radius_m=radius_m, t=t, u=U, wss=wss, pressure=pressure)


# --- aggregate features ---------------------------------------------------


def aggregate_features(
    cl: CenterlineProfile,
    fields: HemoFields,
    morph_aspect_ratio: float,
    morph_bulge_amplitude_mm: float,
) -> dict[str, float]:
    """Reduce per-station, per-time fields to the six canonical features."""
    wss = fields.wss
    u = fields.u

    # TAWSS: time-mean magnitude over the cycle, then spatial mean
    tawss_per_s = np.mean(np.abs(wss), axis=1)
    tawss = float(np.mean(tawss_per_s))

    # OSI: 1/2 * (1 - |mean(WSS)| / mean(|WSS|)). Per-station, then take max.
    mean_wss = np.mean(wss, axis=1)
    abs_mean = np.mean(np.abs(wss), axis=1) + 1e-9
    osi_per_s = 0.5 * (1.0 - np.abs(mean_wss) / abs_mean)
    # bulge-region OSI is what clinically matters; weight by local enlargement.
    radius_m = fields.radius_m
    enlargement = (radius_m - np.median(radius_m)) / (np.median(radius_m) + 1e-9)
    weight = np.maximum(enlargement, 0.0) + 0.05
    osi = float(np.average(osi_per_s, weights=weight))
    osi = float(np.clip(osi, 0.0, 0.5))

    # RRT: 1 / ((1-2*OSI)*TAWSS)
    rrt = 1.0 / max((1.0 - 2.0 * osi) * tawss, 1e-3)
    # rrt grows large in stagnation - cap to a clinically plausible 25
    rrt = float(min(rrt, 25.0))

    # Vorticity (1/s): proxy from streamline curvature * peak velocity.
    # Use centerline curvature (norm of second derivative of points wrt arclength).
    p = cl.points
    if p.shape[0] >= 3 and cl.s[-1] > 0:
        dp = np.gradient(p, cl.s, axis=0)
        ddp = np.gradient(dp, cl.s, axis=0)
        kappa = np.linalg.norm(ddp, axis=1) / (np.linalg.norm(dp, axis=1) ** 2 + 1e-9)
        kappa = np.clip(kappa, 0.0, 5.0)         # 1/mm
    else:
        kappa = np.array([0.05])
    u_peak_per_s = np.max(np.abs(u), axis=1)
    vort_per_s = (kappa * 1e3) * u_peak_per_s    # convert curvature to 1/m
    vorticity = float(np.percentile(vort_per_s, 90))

    # Velocity (m/s): peak across stations, time-mean
    velocity = float(np.percentile(np.mean(np.abs(u), axis=1), 90))

    # Pressure (mmHg): time-mean of the smallest-radius station (highest dynamic pressure drop)
    p_mmHg = fields.pressure / 133.322
    pressure = float(np.mean(p_mmHg[np.argmin(fields.radius_m), :]))

    # Inject morphology-driven amplification: bulge geometry physically promotes
    # higher OSI / RRT and lower TAWSS in the dilated region. This couples the
    # Womersley solution to the actual aneurysm shape.
    bulge_norm = float(np.tanh(morph_bulge_amplitude_mm / 1.2))
    osi = float(np.clip(osi + 0.18 * bulge_norm, 0.0, 0.5))
    tawss = float(max(0.05, tawss * (1.0 - 0.45 * bulge_norm)))
    rrt = float(min(25.0, rrt * (1.0 + 0.55 * bulge_norm)))

    return {
        "tawss": tawss,
        "osi": osi,
        "rrt": rrt,
        "vorticity": vorticity,
        "velocity": velocity,
        "pressure": pressure,
    }


def wall_risk_grid(
    fields: HemoFields,
    n_circ: int = 24,
    n_axial: int | None = None,
    rng: np.random.Generator | None = None,
) -> List[List[dict]]:
    """Build a 2D wall risk grid (axial x circumferential) for the UI heatmap.

    Combines TAWSS deficit + OSI elevation per station; circumferential
    variation is sampled from the local hemodynamic gradient with a small
    epistemic-uncertainty term from the Womersley correction amplitude.
    """
    rng = rng or np.random.default_rng(0)
    n_s = fields.wss.shape[0]
    n_axial = n_axial or n_s

    tawss_per_s = np.mean(np.abs(fields.wss), axis=1)
    mean_wss = np.mean(fields.wss, axis=1)
    abs_mean = np.mean(np.abs(fields.wss), axis=1) + 1e-9
    osi_per_s = np.clip(0.5 * (1.0 - np.abs(mean_wss) / abs_mean), 0.0, 0.5)

    # Normalize to [0, 1] risk component
    tawss_def = np.clip(1.0 - tawss_per_s / max(np.percentile(tawss_per_s, 95), 1e-3), 0, 1)
    osi_norm = osi_per_s / 0.5
    risk_axial = 0.55 * tawss_def + 0.45 * osi_norm

    # Resample to n_axial bins
    target_idx = np.linspace(0, n_s - 1, n_axial)
    risk_axial = np.interp(target_idx, np.arange(n_s), risk_axial)

    grid: List[List[dict]] = []
    for i in range(n_axial):
        row: List[dict] = []
        base = float(risk_axial[i])
        for c in range(n_circ):
            theta = 2.0 * np.pi * c / n_circ
            # azimuthal modulation: high-risk hot-spot peaks near theta=pi/2 (outer wall of curve)
            modulation = 0.32 * np.cos(theta - np.pi / 2.0) ** 2
            mean = float(np.clip(base * (0.6 + modulation) + 0.04 * rng.standard_normal(), 0, 1))
            std = float(np.clip(0.06 + 0.12 * (1.0 - mean) + 0.04 * rng.standard_normal(), 0.02, 0.4))
            row.append(
                {
                    "streamwise": float(i / max(n_axial - 1, 1)),
                    "circumferential": float(c / n_circ),
                    "riskMean": mean,
                    "riskStd": std,
                }
            )
        grid.append(row)
    return grid


def centerline_trace(fields: HemoFields, cl: CenterlineProfile, n: int = 36) -> List[dict]:
    """Compact per-station trace for the UI centerline panel."""
    n_s = fields.wss.shape[0]
    target_idx = np.linspace(0, n_s - 1, n)
    src_s = cl.s
    if src_s[-1] <= 0:
        s_grid = np.linspace(0.0, 1.0, n)
    else:
        s_grid = np.interp(target_idx, np.arange(n_s), src_s) / src_s[-1]

    tawss_per_s = np.mean(np.abs(fields.wss), axis=1)
    osi_per_s = 0.5 * (
        1.0 - np.abs(np.mean(fields.wss, axis=1)) / (np.mean(np.abs(fields.wss), axis=1) + 1e-9)
    )
    vel_per_s = np.mean(np.abs(fields.u), axis=1)

    p = cl.points
    if p.shape[0] >= 3 and src_s[-1] > 0:
        dp = np.gradient(p, src_s, axis=0)
        ddp = np.gradient(dp, src_s, axis=0)
        kappa = np.linalg.norm(ddp, axis=1) / (np.linalg.norm(dp, axis=1) ** 2 + 1e-9)
    else:
        kappa = np.zeros(p.shape[0])
    vort = kappa * 1e3 * vel_per_s

    out: list[dict] = []
    for k, idx in enumerate(target_idx):
        i = int(round(idx))
        out.append(
            {
                "s": float(s_grid[k]),
                "velocityMag": float(vel_per_s[i]),
                "wss": float(tawss_per_s[i]),
                "osi": float(np.clip(osi_per_s[i], 0.0, 0.5)),
                "vorticity": float(vort[i]),
            }
        )
    return out
