"""Package for phase linking stacks of SLCs.

Currently implements the eigenvalue-based maximum likelihood (EMI)
algorithm from (Ansari, 2018).
"""
# from .fringe import create_full_nmap_files, run_evd, run_nmap  # noqa: F401

from .mle import PhaseLinkRuntimeError, run_mle  # noqa: F401
