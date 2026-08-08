"""CyberAdaptAI / AdaptEnsembleNet -- reimplementation from the paper text.

NOT the authors' code. No official implementation exists; see ../README.md ("Provenance").

Paper: Uddamari & Sammulal (2025), SSRG Int. J. of Electronics and Communication Engineering,
12(9), 11-31. DOI 10.14445/23488549/IJECE-V12I9P102
"""

from .config import Config, DiversityMeasure, ScalerFit, WeightRule
from .ensemble import AdaptEnsembleNet
from .runner import CyberAdaptAI
from .stream import Batch, BatchStream

__all__ = [
    "Config",
    "WeightRule",
    "DiversityMeasure",
    "ScalerFit",
    "AdaptEnsembleNet",
    "CyberAdaptAI",
    "BatchStream",
    "Batch",
]

__version__ = "0.1.0"
