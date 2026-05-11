"""
grn_confound_audit: Three-class confound audit for gene regulatory networks.

Classes:
    - Class 1 (Technical):   Batch/donor/method leakage detection via ASI
    - Class 2 (Genomic):     Chromosomal proximity enrichment analysis
    - Class 3 (Topological): Degree-preserving null calibration with
                             per-edge BH FDR
"""

__version__ = "0.2.0"

from .technical import TechnicalAudit
from .proximity import ProximityAudit
from .topological import TopologicalAudit, benjamini_hochberg
from .pipeline import ConfoundAuditPipeline
from .simulate import simulate, SimulationConfig, SimulationBundle

__all__ = [
    "TechnicalAudit",
    "ProximityAudit",
    "TopologicalAudit",
    "ConfoundAuditPipeline",
    "benjamini_hochberg",
    "simulate",
    "SimulationConfig",
    "SimulationBundle",
]
