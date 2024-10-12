import logging
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
from opera_utils import get_dates
from scipy import ndimage, signal

from dolphin import io, utils
from dolphin._types import PathOrStr
from dolphin.workflows.config import SpurtOptions

from ._constants import CONNCOMP_SUFFIX, DEFAULT_CCL_NODATA, UNW_SUFFIX

logger = logging.getLogger(__name__)

DEFAULT_OPTIONS = SpurtOptions()


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
    # TODO: this is a weird hack.. if there are 15 dates, there are 14 interferograms
    # the spurt cli expects one of the filenames to be all 0s? maybe?
    # But also still expects them to be date1_date2.int.tif?
    g_time = Hop3Graph(len(ifg_filenames) + 1)
    logger.info(f"Using Hop3 Graph in time with { g_time.npoints } epochs.")

    date_str_to_file = _map_date_str_to_file(ifg_filenames)
    stack = SLCStackReader(
        slc_files=date_str_to_file,
        temp_coh_file=temporal_coherence_file,
        temp_coh_threshold=options.temporal_coherence_threshold,
    )
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
    # Then list individual SLCs
    dates = [get_dates(f) for f in ifg_filenames]
    if len({d[0] for d in dates}) > 1:
        errmsg = "interferograms for spurt must be single reference."
        raise ValueError(errmsg)

    secondary_dates = [d[1] for d in dates]
    first_date = dates[0][0].strftime(date_fmt)
    date_strings = [utils.format_dates(d, fmt=date_fmt) for d in secondary_dates]

    date_str_to_file: dict[str, PathOrStr | None] = dict(
        zip(date_strings, ifg_filenames)
    )
    # first date - set to None
    # None is special case for reference epoch
    date_str_to_file[first_date] = None
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
