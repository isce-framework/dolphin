"""Package for phase linking stacks of SLCs.

Currently implements the eigenvalue-based maximum likelihood (EMI)
algorithm from (Ansari, 2018).
"""
from ._compress import compress  # noqa: F401
from .mle import PhaseLinkRuntimeError, run_mle  # noqa: F401
