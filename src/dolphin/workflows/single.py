"""Estimate wrapped phase for one ministack of SLCs.

References
----------
    .. [1] Mirzaee, Sara, Falk Amelung, and Heresh Fattahi. "Non-linear phase
    linking using joined distributed and persistent scatterers." Computers &
    Geosciences (2022): 105291.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import DTypeLike
from osgeo import gdal

import dolphin._dates
from dolphin import io, shp, utils
from dolphin._blocks import BlockManager
from dolphin._dates import DEFAULT_DATETIME_FORMAT
from dolphin._decorators import atomic_output
from dolphin._log import get_log
from dolphin._readers import VRTStack
from dolphin._types import Filename
from dolphin.phase_link import PhaseLinkRuntimeError, compress, run_mle

from .config import ShpMethod

logger = get_log(__name__)

__all__ = ["run_wrapped_phase_single"]


@dataclass
class OutputFile:
    filename: Path
    dtype: DTypeLike
    strides: Optional[dict[str, int]] = None


@atomic_output(output_arg="output_folder", is_dir=True)
def run_wrapped_phase_single(
    *,
    slc_vrt_file: Filename,
    # Ministack: MiniStack,
    output_folder: Filename,
    half_window: dict,
    strides: dict = {"x": 1, "y": 1},
    reference_idx: int = 0,
    beta: float = 0.01,
    use_evd: bool = False,
    mask_file: Optional[Filename] = None,
    ps_mask_file: Optional[Filename] = None,
    amp_mean_file: Optional[Filename] = None,
    amp_dispersion_file: Optional[Filename] = None,
    shp_method: ShpMethod = ShpMethod.NONE,
    shp_alpha: float = 0.05,
    shp_nslc: Optional[int] = None,
    block_shape: tuple[int, int] = (1024, 1024),
    n_workers: int = 1,
    gpu_enabled: bool = True,
    show_progress: bool = False,
):
    """Estimate wrapped phase for one ministack.

    Output files will all be placed in the provided `output_folder`.
    """
    # TODO: extract common stuff between here and sequential
    output_folder = Path(output_folder)
    vrt = VRTStack.from_vrt_file(slc_vrt_file)
    input_slc_files = vrt.file_list
    input_slc_dates = vrt.dates

    # If we are using a different number of SLCs for the amplitude data,
    # we should note that for the SHP finding algorithms
    if shp_nslc is None:
        shp_nslc = len(input_slc_files)

    logger.info(f"{vrt}: from {vrt.file_list[0]} to {vrt.file_list[-1]}")

    nrows, ncols = vrt.shape[-2:]

    nodata_mask = _get_nodata_mask(mask_file, nrows, ncols)
    ps_mask = _get_ps_mask(ps_mask_file, nrows, ncols)
    amp_mean, amp_variance = _get_amp_mean_variance(amp_mean_file, amp_dispersion_file)
    amp_stack: Optional[np.ndarray] = None

    xhalf, yhalf = half_window["x"], half_window["y"]

    # If we were passed any compressed SLCs in `input_slc_files`,
    # then we want that index for when we create new compressed SLCs.
    # We skip the old compressed SLCs to create new ones
    first_non_comp_idx = 0
    for filename in input_slc_files:
        if not Path(filename).name.startswith("compressed"):
            break
        first_non_comp_idx += 1

    # Make the output folder using the start/end dates
    d0 = input_slc_dates[first_non_comp_idx][0]
    d1 = input_slc_dates[-1][0]
    start_end = dolphin._dates._format_date_pair(d0, d1)

    msg = (
        f"Processing {len(input_slc_files) - first_non_comp_idx} SLCs +"
        f" {first_non_comp_idx} compressed SLCs. "
    )
    logger.info(msg)

    # Create the background writer for this ministack
    writer = io.Writer(debug=False)

    logger.info(
        f"{vrt}: from {Path(vrt.file_list[first_non_comp_idx]).name} to"
        f" {Path(vrt.file_list[-1]).name}"
    )
    logger.info(f"Total stack size (in pixels): {vrt.shape}")
    # Set up the output folder with empty files to write into
    phase_linked_slc_files = setup_output_folder(
        vrt,
        driver="GTiff",
        start_idx=first_non_comp_idx,
        strides=strides,
        output_folder=output_folder,
        nodata=0,
    )

    output_files: list[OutputFile] = [
        OutputFile(output_folder / f"compressed_{start_end}.tif", np.complex64),
        OutputFile(output_folder / f"tcorr_{start_end}.tif", np.float32, strides),
        OutputFile(output_folder / f"avg_coh_{start_end}.tif", np.uint16, strides),
        OutputFile(output_folder / f"shp_counts_{start_end}.tif", np.uint16, strides),
    ]
    for op in output_files:
        io.write_arr(
            arr=None,
            like_filename=vrt.outfile,
            output_name=op.filename,
            dtype=op.dtype,
            strides=op.strides,
            nbands=1,
            nodata=0,
        )

    # Iterate over the output grid
    block_manager = BlockManager(
        arr_shape=(nrows, ncols),
        block_shape=block_shape,
        strides=strides,
        half_window=half_window,
    )
    # block_gen = vrt.iter_blocks(
    #     block_shape=block_shape,
    #     overlaps=overlaps,
    #     skip_empty=True,
    #     nodata_mask=nodata_mask,
    #     show_progress=show_progress,
    # )
    # for cur_data, (rows, cols) in block_gen:
    blocks = list(block_manager.iter_blocks())
    logger.info(f"Iterating over {block_shape} blocks, {len(blocks)} total")
    for (
        (out_rows, out_cols),
        (trimmed_rows, trimmed_cols),
        (in_rows, in_cols),
        (in_no_pad_rows, in_no_pad_cols),
    ) in blocks:
        logger.debug(f"{out_rows = }, {out_cols = }, {in_rows = }, {in_no_pad_rows = }")
        cur_data = vrt.read_stack(rows=in_rows, cols=in_cols)
        if np.all(cur_data == 0):
            continue
        cur_data = cur_data.astype(np.complex64)

        if shp_method == "ks":
            # Only actually compute if we need this one
            amp_stack = np.abs(cur_data)

        # Compute the neighbor_arrays for this block
        neighbor_arrays = shp.estimate_neighbors(
            halfwin_rowcol=(yhalf, xhalf),
            alpha=shp_alpha,
            strides=strides,
            mean=amp_mean[in_rows, in_cols] if amp_mean is not None else None,
            var=amp_variance[in_rows, in_cols] if amp_variance is not None else None,
            nslc=shp_nslc,
            amp_stack=amp_stack,
            method=shp_method,
        )
        # Run the phase linking process on the current ministack
        reference_idx = max(0, first_non_comp_idx - 1)
        try:
            cur_mle_stack, tcorr, avg_coh = run_mle(
                cur_data,
                half_window=half_window,
                strides=strides,
                use_evd=use_evd,
                beta=beta,
                reference_idx=reference_idx,
                nodata_mask=nodata_mask[in_rows, in_cols],
                ps_mask=ps_mask[in_rows, in_cols],
                neighbor_arrays=neighbor_arrays,
                avg_mag=amp_mean[in_rows, in_cols] if amp_mean is not None else None,
                n_workers=n_workers,
                gpu_enabled=gpu_enabled,
            )
        except PhaseLinkRuntimeError as e:
            # note: this is a warning instead of info, since it should
            # get caught at the "skip_empty" step
            msg = f"At block {in_rows.start}, {in_cols.start}: {e}"
            if "are all NaNs" in e.args[0]:
                # Some SLCs in the ministack are all NaNs
                # This happens from a shifting burst window near the edges,
                # and seems to cause no issues
                logger.debug(msg)
            else:
                logger.warning(msg)
            continue

        # Fill in the nan values with 0
        np.nan_to_num(cur_mle_stack, copy=False)
        np.nan_to_num(tcorr, copy=False)

        # Save each of the MLE estimates (ignoring those corresponding to
        # compressed SLCs indexes)
        assert len(cur_mle_stack[first_non_comp_idx:]) == len(phase_linked_slc_files)

        for img, f in zip(
            cur_mle_stack[first_non_comp_idx:, trimmed_rows, trimmed_cols],
            phase_linked_slc_files,
        ):
            writer.queue_write(img, f, out_rows.start, out_cols.start)

        # Get the SHP counts for each pixel (if not using Rect window)
        shp_counts = np.sum(neighbor_arrays, axis=(-2, -1))

        # Get the inner portion of the full-res SLC data
        trim_full_col = slice(
            in_no_pad_cols.start - in_cols.start, in_no_pad_cols.stop - in_cols.stop
        )
        trim_full_row = slice(
            in_no_pad_rows.start - in_rows.start, in_no_pad_rows.stop - in_rows.stop
        )
        # Compress the ministack using only the non-compressed SLCs
        cur_comp_slc = compress(
            cur_data[first_non_comp_idx:, trim_full_row, trim_full_col],
            cur_mle_stack[first_non_comp_idx:, trimmed_rows, trimmed_cols],
        )

        # ### Save results ###

        # Save the compressed SLC block
        writer.queue_write(
            cur_comp_slc,
            output_files[0].filename,
            in_no_pad_rows.start,
            in_no_pad_cols.start,
        )

        # All other outputs are strided (smaller in size)
        out_datas = [tcorr, avg_coh, shp_counts]
        for data, output_file in zip(out_datas, output_files[1:]):
            if data is None:  # May choose to skip some outputs, e.g. "avg_coh"
                continue
            writer.queue_write(
                data[trimmed_rows, trimmed_cols],
                output_file.filename,
                out_rows.start,
                out_cols.start,
            )

    # Block until all the writers for this ministack have finished
    logger.info(f"Waiting to write {writer.num_queued} blocks of data.")
    writer.notify_finished()
    logger.info(f"Finished ministack of size {vrt.shape}.")

    comp_slc_file, tcorr_file = output_files[0:2]

    input_compressed_files = input_slc_files[:first_non_comp_idx]
    input_real_slc_files = input_slc_files[first_non_comp_idx:]
    _save_compressed_metadata(
        comp_slc_file.filename, input_real_slc_files, input_compressed_files
    )
    # TODO: what would make most sense to return here?
    # return phase_linked_slc_files, comp_slc_file, tcorr_file
    return phase_linked_slc_files, comp_slc_file.filename, tcorr_file.filename


def _save_compressed_metadata(
    comp_slc_file: Path,
    input_real_slc_files: list[Filename],
    input_compressed_files: list[Filename],
):
    """Save metadata about the compressed SLCs.

    The filenames will be stored in GDAL metadata as a list of strings, separated
    by commas, saved into the "DOLPHIN" domain.

    Parameters
    ----------
    comp_slc_file : Path
        Path to the compressed SLC file
    input_real_slc_files : list[Path]
        List of input SLC files that were used to create the compressed SLC
    input_compressed_files : list[Path]
        List of input compressed SLC files that were used in the ministack
        which produced the phase linking result used the compressed SLCs.
        Note that these were not included in the dot product, but were used
        by the phase linking optimization.
    """
    comp_ds = gdal.Open(str(comp_slc_file), gdal.GA_Update)
    real_names = [utils._get_path_from_gdal_str(p).name for p in input_real_slc_files]
    comp_ds.SetMetadataItem(
        "input_real_slc_files",
        ",".join(map(str, real_names)),
        "DOLPHIN",
    )
    comp_names = [utils._get_path_from_gdal_str(p).name for p in input_compressed_files]
    comp_ds.SetMetadataItem(
        "input_compressed_slc_files",
        ",".join(map(str, comp_names)),
        "DOLPHIN",
    )
    comp_ds.FlushCache()


def _get_nodata_mask(
    mask_file: Optional[Filename],
    nrows: int,
    ncols: int,
) -> np.ndarray:
    if mask_file is not None:
        # The mask file will by -2s at invalid data, 1s at good
        nodata_mask = io.load_gdal(mask_file, masked=True).astype(bool).filled(False)
        # invert the mask so -1s are the missing data pixels
        nodata_mask = ~nodata_mask
        # check middle pixel
        if nodata_mask[nrows // 0, ncols // 2]:
            logger.warning(f"{mask_file} is True at {nrows//0, ncols//2}")
            logger.warning("Proceeding without the nodata mask.")
            nodata_mask = np.zeros((nrows, ncols), dtype=bool)
    else:
        nodata_mask = np.zeros((nrows, ncols), dtype=bool)
    return nodata_mask


def _get_ps_mask(
    ps_mask_file: Optional[Filename], nrows: int, ncols: int
) -> np.ndarray:
    if ps_mask_file is not None:
        ps_mask = io.load_gdal(ps_mask_file, masked=True)
        # Fill the nodata values with false
        ps_mask = ps_mask.astype(bool).filled(False)
    else:
        ps_mask = np.zeros((nrows, ncols), dtype=bool)
    return ps_mask


def _get_amp_mean_variance(
    amp_mean_file: Optional[Filename],
    amp_dispersion_file: Optional[Filename],
) -> tuple[np.ndarray, np.ndarray]:
    if amp_mean_file is not None and amp_dispersion_file is not None:
        # Note: have to fill, since numba (as of 0.57) can't do masked arrays
        amp_mean = io.load_gdal(amp_mean_file, masked=True).filled(np.nan)
        amp_dispersion = io.load_gdal(amp_dispersion_file, masked=True).filled(np.nan)
        # convert back to variance from dispersion: amp_disp = std_dev / mean
        amp_variance = (amp_dispersion * amp_mean) ** 2
    else:
        amp_mean = amp_variance = None

    return amp_mean, amp_variance


def setup_output_folder(
    vrt_stack,
    driver: str = "GTiff",
    dtype="complex64",
    start_idx: int = 0,
    strides: dict[str, int] = {"y": 1, "x": 1},
    nodata: Optional[float] = 0,
    output_folder: Optional[Path] = None,
) -> list[Path]:
    """Create empty output files for each band after `start_idx` in `vrt_stack`.

    Also creates an empty file for the compressed SLC.
    Used to prepare output for block processing.

    Parameters
    ----------
    vrt_stack : VRTStack
        object containing the current stack of SLCs
    driver : str, optional
        Name of GDAL driver, by default "GTiff"
    dtype : str, optional
        Numpy datatype of output files, by default "complex64"
    start_idx : int, optional
        Index of vrt_stack to begin making output files.
        This should match the ministack index to avoid re-creating the
        past compressed SLCs.
    strides : dict[str, int], optional
        Strides to use when creating the empty files, by default {"y": 1, "x": 1}
        Larger strides will create smaller output files, computed using
        [dolphin.io.compute_out_shape][]
    nodata : float, optional
        Nodata value to use for the output files, by default 0.
    output_folder : Path, optional
        Path to output folder, by default None
        If None, will use the same folder as the first SLC in `vrt_stack`

    Returns
    -------
    list[Path]
        list of saved empty files for the outputs of phase linking
    """
    if output_folder is None:
        output_folder = vrt_stack.outfile.parent

    date_strs: list[str] = []
    for d in vrt_stack.dates[start_idx:]:
        if len(d) == 1:
            # normal SLC files will have a single date
            s = d[0].strftime(DEFAULT_DATETIME_FORMAT)
        else:
            # Compressed SLCs will have 2 dates in the name marking the start and end
            s = dolphin._dates._format_date_pair(d[0], d[1])
        date_strs.append(s)

    phase_linked_slc_files = []
    for filename in date_strs:
        slc_name = Path(filename).stem
        output_path = output_folder / f"{slc_name}.slc.tif"

        io.write_arr(
            arr=None,
            like_filename=vrt_stack.outfile,
            output_name=output_path,
            driver=driver,
            nbands=1,
            dtype=dtype,
            strides=strides,
            nodata=nodata,
        )

        phase_linked_slc_files.append(output_path)
    return phase_linked_slc_files
