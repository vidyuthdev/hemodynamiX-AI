"""HemodynamiX AI - end-to-end CFD-informed risk pipeline.

Real intracranial aneurysm meshes from the AnXplore dataset
(Frontiers in Bioengineering 2024) are processed with PyVista.
A reduced-order Womersley pulsatile flow model derives time-resolved
wall hemodynamics (TAWSS, OSI, RRT, vorticity, velocity, pressure).

Labels are derived from clinically validated morphological criteria
(Lauric et al. 2018: aspect ratio >= 1.6 OR size ratio >= 2.05) with
realistic clinical label noise (~10%). XGBoost, Random Forest and a
shallow MLP are trained, calibrated with Platt scaling, audited with
split conformal and TreeSHAP, and stress-tested across imaging
resolutions and aneurysm locations.
"""

from .config import REAL_DATA_URLS, PipelineConfig

__all__ = ["REAL_DATA_URLS", "PipelineConfig"]
