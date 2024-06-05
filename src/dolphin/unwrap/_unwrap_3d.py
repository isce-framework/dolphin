import logging
from pathlib import Path
from typing import Sequence

from dolphin._types import PathOrStr
from dolphin.workflows.config import UnwrapOptions

logger = logging.getLogger(__name__)


def unwrap_3d(
    ifg_filenames: Sequence[PathOrStr],
    cor_filenames: Sequence[PathOrStr] | None,
    unwrap_options: UnwrapOptions,
) -> tuple[list[Path], list[Path]]:
    """Perform 3D unwrapping using `spurt`."""
    if unwrap_options.run_goldstein:
        logger.warning("Goldstein filtering not implemented for 3D unwrapping")
    if unwrap_options.run_interpolation:
        logger.warning("Interpolation not implemented for 3D unwrapping")
    if cor_filenames is not None:
        assert len(ifg_filenames) == len(cor_filenames)
    raise NotImplementedError()
