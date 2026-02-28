"""
grn_confound_audit: Three-class confound audit for gene regulatory networks.

Implements the diagnostic pipeline from:
  "Three Classes of Confound in Gene-Regulatory-Network Inference:
   A Systematic Audit of Technical, Genomic, and Topological Biases"

Classes:
    - Class 1 (Technical): Batch/donor/method leakage detection via ASI
    - Class 2 (Genomic): Chromosomal proximity enrichment analysis
    - Class 3 (Topological): Degree-preserving null calibration
"""

__version__ = "0.1.0"

from .technical import TechnicalAudit
from .proximity import ProximityAudit
from .topological import TopologicalAudit
from .pipeline import ConfoundAuditPipeline
