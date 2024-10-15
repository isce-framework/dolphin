import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import rasterio
from numpy.typing import NDArray
from opera_utils import get_dates
from scipy import ndimage, signal

from dolphin import io, utils
from dolphin._types import PathOrStr
from dolphin.workflows.config import SpurtOptions

from ._constants import CONNCOMP_SUFFIX, DEFAULT_CCL_NODATA, UNW_SUFFIX
from ._post_process import interpolate_masked_gaps

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

    if existing_unw_files := sorted(Path(output_path).glob(f"*{UNW_SUFFIX}")):
        logger.info(f"Found {len(existing_unw_files)} unwrapped files")
        existing_ccl_files = sorted(Path(output_path).glob(f"*{CONNCOMP_SUFFIX}"))
        return existing_unw_files, existing_ccl_files

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
    filled_masked_unw_regions(unw_filenames, ifg_filenames)

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


def filled_masked_unw_regions(
    unw_filenames: Sequence[PathOrStr],
    ifg_filenames: Sequence[PathOrStr],
    output_dir: Path | None = None,
) -> None:
    """Fill the nan gaps in `unw_filenames` using the wrapped `ifg_filenames`.

    This function iterates through the nearest-3 unwrapped filenames from spurt,
    calculates the wrapped phase difference from 2 `ifg_filenames` and interpolates
    the unwrapped ambiguity number to fill the gaps.

    Parameters
    ----------
    unw_filenames : Sequence[PathOrStr]
        List of the nearest-3 unwrapped filenames from spurt, containing nan gaps.
    ifg_filenames : Sequence[PathOrStr]
        Wrapped, single-reference interferogram filenames used as input to spurt.
    output_dir : Path, optional
        Separate folder to write output files after filling gaps.
        If None, overwrites the `unw_filenames`.

    """
    if output_dir is None:
        output_dir = Path(unw_filenames[0]).parent

    with rasterio.open(unw_filenames[0]) as src:
        profile = src.profile.copy()
    for unw_filename in unw_filenames:
        unw, wrapped_phase = _reform_wrapped_phase(unw_filename, ifg_filenames)
        interpolate_masked_gaps(unw, wrapped_phase)

        # Save the updated unwrapped phase
        kwargs = profile | {
            "count": 1,
            "height": unw.shape[0],
            "width": unw.shape[1],
            "dtype": "float32",
        }
        with rasterio.open(output_dir / Path(unw_filename).name, "w", **kwargs) as src:
            src.write(unw, 1)


def _reform_wrapped_phase(
    unw_filename: PathOrStr, ifg_filenames: Sequence[PathOrStr]
) -> tuple[NDArray[np.float64], NDArray[np.complex64]]:
    """Load unwrapped phase, and re-calculate the corresponding wrapped phase.

    Finds the matching ifg to `unw_filename`, or uses 2 to compute the correct
    wrapped phase. For example, if `unw_filename` is like (day4_day5), then we load
    the `ifg1 = (day1_day4)`, `ifg2 = (day1_day5)`, and compute `a * b.conj()`.
    """
    # Extract dates from unw_filename
    unw_dates = get_dates(Path(unw_filename))

    date1, date2 = unw_dates

    ifg_date_tuples = [get_dates(p) for p in ifg_filenames]
    if len({tup[0] for tup in ifg_date_tuples}) > 1:
        raise ValueError(
            "ifg_filenames must contain only single-reference interferograms"
        )

    # Find the required interferogram filenames
    ifg1_name = None
    ifg2_name = None
    for ifg in ifg_filenames:
        ifg_dates = get_dates(Path(ifg))
        if len(ifg_dates) != 2:
            continue
        if ifg_dates == unw_dates:
            ifg1_name = ifg
            break

        _ref, sec_date = ifg_dates
        if sec_date == date1:
            ifg1_name = ifg
        if sec_date == date2:
            ifg2_name = ifg

    if ifg1_name is None and ifg2_name is None:
        raise ValueError(f"Could not find required interferograms for {unw_filename}")

    logger.info(f"Interpolating nodata in {unw_filename} with {ifg1_name}, {ifg2_name}")
    # Load the files
    with rasterio.open(unw_filename) as src:
        unw = src.read(1)

    with rasterio.open(ifg1_name) as src:
        ifg1 = src.read(1)

    if ifg2_name is not None:
        with rasterio.open(ifg2_name) as src:
            ifg2 = src.read(1)
        # Calculate the wrapped phase difference
        wrapped_phase = np.angle(ifg1 * np.conj(ifg2))
    else:
        wrapped_phase = ifg1

    return unw, wrapped_phase
