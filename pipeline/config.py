"""Static configuration for the HemodynamiX pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "anxplore_raw"
CACHE_DIR = DATA_DIR / "cache"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
PUBLIC_DIR = PROJECT_ROOT / "public"

# AnXplore on GitHub publishes ~100 idealized intracranial aneurysm
# tetrahedral meshes (Goetz et al., Frontiers Bioeng. 2024). Each file
# is ~40-80 MB raw VTK. Direct, anonymous downloads.
ANXPLORE_BASE = (
    "https://github.com/aurelegoetz/AnXplore/raw/main/full_dataset/Fluid_{idx}.vtk"
)
ANXPLORE_NAMED_BASE = (
    "https://github.com/aurelegoetz/AnXplore/raw/main/case{name}/Fluid_case{name}.vtk"
)
NAMED_CASES = ("A", "B", "C", "R")

REAL_DATA_URLS = {
    "anxplore_full": ANXPLORE_BASE,
    "anxplore_named": ANXPLORE_NAMED_BASE,
}

# Six hemodynamic features used by the downstream models, in canonical
# order. The React workbench expects this exact ordering.
FEATURE_KEYS: tuple[str, ...] = (
    "tawss",
    "osi",
    "rrt",
    "vorticity",
    "velocity",
    "pressure",
)

FEATURE_LABELS = {
    "tawss": "TAWSS (Pa)",
    "osi": "OSI (-)",
    "rrt": "RRT (1/Pa)",
    "vorticity": "Vorticity (1/s)",
    "velocity": "Velocity (m/s)",
    "pressure": "Pressure (mmHg)",
}

LOCATIONS = ("ACOM", "MCA", "PCOM")


@dataclass
class PipelineConfig:
    """Top-level pipeline knobs."""

    n_real_cases: int = 90               # how many AnXplore cases to download
    n_synthetic_cases: int = 220         # parametric augmentation for ML
    label_noise_rate: float = 0.10       # realistic clinical label noise
    seed: int = 13
    cohort_test_frac: float = 0.20
    cohort_cal_frac: float = 0.15        # of remaining
    conformal_alpha: float = 0.10        # 90% target coverage
    download_workers: int = 8
    locations: tuple[str, ...] = field(default_factory=lambda: LOCATIONS)
    public_dir: Path = field(default_factory=lambda: PUBLIC_DIR)
    raw_dir: Path = field(default_factory=lambda: RAW_DIR)
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR)
    artifacts_dir: Path = field(default_factory=lambda: ARTIFACTS_DIR)


def ensure_dirs(cfg: PipelineConfig) -> None:
    for p in (cfg.raw_dir, cfg.cache_dir, cfg.artifacts_dir, cfg.public_dir):
        p.mkdir(parents=True, exist_ok=True)


def feature_keys() -> List[str]:
    return list(FEATURE_KEYS)
