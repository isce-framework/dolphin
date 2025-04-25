from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    _SubparserType = argparse._SubParsersAction[argparse.ArgumentParser]
else:
    _SubparserType = Any


def run(
    config_file: str,
    debug: bool = False,
) -> None:
    """Run the displacement workflow.

    Parameters
    ----------
    config_file : str
        YAML file containing the workflow options.
    debug : bool, optional
        Enable debug logging, by default False.

    """
    # rest of imports here so --help doesn't take forever

    from . import displacement
    from .config import DisplacementWorkflow

    cfg = DisplacementWorkflow.from_yaml(config_file)
    displacement.run(cfg, debug=debug)
