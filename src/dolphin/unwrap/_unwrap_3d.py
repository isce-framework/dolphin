import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import rasterio as rio
from opera_utils import get_dates, sort_files_by_date
from rasterio.windows import Window
from scipy import ndimage, signal

from dolphin import io, utils
from dolphin._types import PathOrStr
from dolphin.workflows.config import SpurtOptions

from ._constants import CONNCOMP_SUFFIX, DEFAULT_CCL_NODATA, UNW_SUFFIX

logger = logging.getLogger(__name__)

DEFAULT_OPTIONS = SpurtOptions()

if TYPE_CHECKING:
    from spurt.io import Irreg3DInput


def unwrap_spurt(
    ifg_filenames: Sequence[PathOrStr],
    output_path: PathOrStr,
    temporal_coherence_file: PathOrStr,
    cor_filenames: Sequence[PathOrStr] | None = None,
    mask_filename: PathOrStr | None = None,
    options: SpurtOptions = DEFAULT_OPTIONS,
    scratchdir: PathOrStr | None = None,
) -> tuple[list[Path], list[Path]]:
    """Perform 3D unwrapping using `spurt`."""
    from spurt.graph import Hop3Graph
    from spurt.io import SLCStackReader
    from spurt.workflows.emcf import (
        GeneralSettings,
        MergerSettings,
        SolverSettings,
        TilerSettings,
        compute_phasediff_deciles,
        get_bulk_offsets,
        get_tiles,
        merge_tiles,
        unwrap_tiles,
    )

    if cor_filenames is not None:
        assert len(ifg_filenames) == len(cor_filenames)
    if mask_filename is not None:
        # TODO: Combine this with the temporal coherence to pass one 0/1 mask
        # This will still work for spurt, since it runs `> threshold`, which is
        # always true once we set our desired pixels to 1 and undesired to 0
        _mask = io.load_gdal(mask_filename)

    if scratchdir is None:
        scratchdir = Path(output_path) / "scratch"
    gen_settings = GeneralSettings(
        output_folder=output_path,
        intermediate_folder=scratchdir,
        **options.general_settings.model_dump(),
    )

    tile_settings = TilerSettings(**options.tiler_settings.model_dump())
    slv_settings = SolverSettings(**options.solver_settings.model_dump())
    mrg_settings = MergerSettings(**options.merger_settings.model_dump())

    # Using default Hop3Graph
    ifg_date_tuples = [get_dates(f) for f in ifg_filenames]
    all_dates = sorted(set(utils.flatten(ifg_date_tuples)))
    g_time = Hop3Graph(len(all_dates))
    logger.info(f"Using Hop3 Graph in time with { g_time.npoints } epochs.")

    if len(all_dates) == len(ifg_filenames) + 1:
        # Single reference interferograms: do they all have same first date?
        if len({d[0] for d in ifg_date_tuples}) > 1:
            errmsg = "interferograms for spurt must be single reference."
            raise ValueError(errmsg)
        logger.info(
            "Converting single-reference interferograms to Hop3 during unwrapping"
        )
        date_str_to_file = _map_date_str_to_file(ifg_filenames)
        stack = SLCStackReader(
            slc_files=date_str_to_file,
            temp_coh_file=temporal_coherence_file,
            temp_coh_threshold=options.temporal_coherence_threshold,
        )
    elif len(ifg_filenames) == len(g_time.links):
        logger.info("Using pre-formed nearest 3 interferograms")
        stack = IfgStackReader(
            ifg_filenames=sort_files_by_date(ifg_filenames)[0],
            quality_file=temporal_coherence_file,
            threshold=options.temporal_coherence_threshold,
        )
    else:
        raise ValueError("spurt requires nearest-3 interferograms, or single-reference")

    # Run the workflow
    # Generate tiles
    get_tiles(stack, gen_settings, tile_settings)

    # Unwrap tiles
    unwrap_tiles(stack, g_time, gen_settings, slv_settings)

    # Compute overlap stats
    compute_phasediff_deciles(gen_settings, mrg_settings)

    # Compute bulk offsets
    get_bulk_offsets(stack, gen_settings, mrg_settings)

    # Merge tiles and write output
    unw_filenames = merge_tiles(stack, g_time, gen_settings, mrg_settings)
    # TODO: What can we do for conncomps? Anything? Run snaphu?
    conncomp_filenames = _create_conncomps_from_mask(
        temporal_coherence_file,
        options.temporal_coherence_threshold,
        unw_filenames=unw_filenames,
    )

    return unw_filenames, conncomp_filenames


def _map_date_str_to_file(
    ifg_filenames: Sequence[PathOrStr], date_fmt: str = "%Y%m%d"
) -> dict[str, PathOrStr | None]:
    date_tuples = sorted([get_dates(f) for f in ifg_filenames])

    secondary_dates = [d[1] for d in date_tuples]
    first_date = date_tuples[0][0].strftime(date_fmt)
    date_strings = [utils.format_dates(d, fmt=date_fmt) for d in secondary_dates]

    # first date - set to None
    # None is special case for reference epoch
    date_str_to_file: dict[str, PathOrStr | None] = {first_date: None} | dict(
        zip(date_strings, ifg_filenames)
    )
    return date_str_to_file


def _create_conncomps_from_mask(
    temporal_coherence_file: PathOrStr,
    temporal_coherence_threshold: float,
    unw_filenames: Sequence[PathOrStr],
    dilate_by: int = 25,
) -> list[Path]:
    arr = io.load_gdal(temporal_coherence_file, masked=True)
    good_pixels = arr > temporal_coherence_threshold
    strel = np.ones((dilate_by, dilate_by))
    # "1" pixels will be spread out and have (approximately) 1.0 in surrounding pixels
    # Note: `ndimage.binary_dilation` scales ~quadratically with `dilate_by`,
    # whereas FFT-based convolution is roughly constant for any `dilate_by`.
    good_pixels_dilated = signal.fftconvolve(good_pixels, strel, mode="same") > 0.95
    # Label the contiguous areas based on the dilated version
    labels, nlabels = ndimage.label(good_pixels_dilated)
    logger.debug("Labeled %s connected components", nlabels)
    # Now these labels will extend to areas which are not "good" in the original.
    labels[~good_pixels] = 0
    # An make a masked version based on the original nodata, then fill with final output
    conncomp_arr = (
        np.ma.MaskedArray(data=labels, mask=arr.mask)
        .astype("uint16")
        .filled(DEFAULT_CCL_NODATA)
    )

    conncomp_files = [
        Path(str(outf).replace(UNW_SUFFIX, CONNCOMP_SUFFIX)) for outf in unw_filenames
    ]
    # Write the first one with geo-metadata
    io.write_arr(
        arr=conncomp_arr,
        like_filename=unw_filenames[0],
        output_name=conncomp_files[0],
        nodata=DEFAULT_CCL_NODATA,
    )
    for f in conncomp_files[1:]:
        shutil.copy(conncomp_files[0], f)
    return conncomp_files


class IfgStackReader:
    """Class to read slides of nearest 3- interferograms."""

    def __init__(
        self,
        ifg_filenames: Sequence[PathOrStr],
        quality_file: PathOrStr,
        threshold: float = 0.6,
    ):
        self.quality_file = quality_file
        self.threshold = threshold
        date_str_to_file = _map_date_str_to_file(ifg_filenames)
        # Keep naming as spurt currently does
        # TODO: change once reader interface is merged,
        # along with the "temp coh" references
        # Note that even though these "slc_files" are NOT really right,
        # spurt doesn't use this except for one place to get a `like_filename`.
        # This hack should also go away in the interface change
        self.slc_files = date_str_to_file
        # Extract dates by getting a list of slc_files keys
        self.dates = sorted(date_str_to_file.keys())
        self.ifg_filenames = ifg_filenames

        self._reader = io.RasterStackReader.from_file_list(ifg_filenames)

    @property
    def temp_coh_file(self) -> PathOrStr:
        return self.quality_file

    @property
    def temp_coh_threshold(self) -> float:
        return self.threshold

    def read_tile(self, space: tuple[slice, ...]) -> "Irreg3DInput":
        """Return a tile of 3D sparse data."""
        # First read the quality file to get dimensions
        from spurt.io import Irreg3DInput

        msk = self.read_mask(space=space)
        xy = np.column_stack(np.where(msk))

        # Assumed complex64 for now but can read one file and determine
        rows, cols = space
        arr = self._reader[:, rows, cols]

        return Irreg3DInput(arr[:, msk], xy)

    def read_quality(self, space: tuple[slice, ...] | None = None) -> np.ndarray:
        """Read in a slice from the quality raster."""
        if space is None:
            space = np.s_[:, :]
        row_slice, col_slice = space
        with rio.open(self.temp_coh_file) as src:
            window = Window.from_slices(
                row_slice, col_slice, width=src.width, height=src.height
            )
            return src.read(1, window=window)

    def read_temporal_coherence(
        self, space: tuple[slice, ...] | None = None
    ) -> np.ndarray:
        return self.read_quality(space=space)

    def read_mask(self, space: tuple[slice, ...] | None = None) -> np.ndarray:
        if space is None:
            space = np.s_[:, :]
        return self.read_temporal_coherence(space=space) > self.temp_coh_threshold
