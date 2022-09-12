#!/usr/bin/env python
# from pathlib import Path

# from atlas import combine_ps_ds, phase_linking, ps  # , unwrap
from dolphin.log import get_log, log_runtime

logger = get_log()


@log_runtime
def run(cfg: dict):
    """Run the displacement workflow on one incremental SLC.

    Parameters
    ----------
    cfg : dict
        Loaded configuration from YAML workflow file.
    """
    print(cfg)
    pass
