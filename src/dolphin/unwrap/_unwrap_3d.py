import logging
from pathlib import Path
from typing import Sequence

from dolphin import io
from dolphin._types import PathOrStr
from dolphin.workflows.config import SpurtOptions

logger = logging.getLogger(__name__)

DEFAULT_OPTIONS = SpurtOptions()


def unwrap_spurt(
    ifg_filenames: Sequence[PathOrStr],
    cor_filenames: Sequence[PathOrStr] | None,
    mask_filename: PathOrStr | None = None,
    options: SpurtOptions = DEFAULT_OPTIONS,
) -> tuple[list[Path], list[Path]]:
    """Perform 3D unwrapping using `spurt`."""
    if cor_filenames is not None:
        assert len(ifg_filenames) == len(cor_filenames)
    if mask_filename is not None:
        _mask = io.load_gdal(mask_filename)
    assert options is not None  # Remove opon implementing
    raise NotImplementedError()
