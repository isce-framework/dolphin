"""Package for phase linking stacks of SLCs.

Currently implements the eigenvalue-based maximum likelihood (EMI) algorithm from
[@Ansari2018EfficientPhaseEstimation], as well as the EVD based approach from
[@Fornaro2015CAESARApproachBased] and [@Mirzaee2023NonlinearPhaseLinking]
"""

from ._compress import compress
from ._core import PhaseLinkRuntimeError, run_phase_linking
