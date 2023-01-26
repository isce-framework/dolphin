"""Estimate wrapped phase using batches of ministacks.

References
----------
    [1] Ansari, H., De Zan, F., & Bamler, R. (2017). Sequential estimator: Toward
    efficient InSAR time series analysis. IEEE Transactions on Geoscience and
    Remote Sensing, 55(10), 5637-5652.
"""
import warnings
from collections import defaultdict
from math import nan
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from osgeo_utils import gdal_calc
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dolphin import io
from dolphin._log import get_log
from dolphin._types import Filename
from dolphin.interferogram import VRTInterferogram
from dolphin.phase_link import PhaseLinkRuntimeError, run_mle
from dolphin.utils import upsample_nearest
from dolphin.vrt import VRTStack

logger = get_log()

__all__ = ["run_evd_sequential", "compress"]


def run_evd_sequential(
    *,
    slc_vrt_file: Filename,
    # weight_file: Filename,
    output_folder: Filename,
    half_window: dict,
    strides: dict = {"x": 1, "y": 1},
    ministack_size: int = 10,
    mask_file: Optional[Filename] = None,
    ps_mask_file: Optional[Filename] = None,
    beta: float = 0.1,
    max_bytes: float = 32e6,
    n_workers: int = 1,
    gpu_enabled: bool = True,
):
    """Estimate wrapped phase using batches of ministacks."""
    output_folder = Path(output_folder)
    v_all = VRTStack.from_vrt_file(slc_vrt_file)
    file_list_all = v_all.file_list
    date_list_all = v_all.dates

    logger.info(f"{v_all}: from {v_all.file_list[0]} to {v_all.file_list[-1]}")

    # Map of {ministack_index: [output_slc_files]}
    output_slc_files: Dict[int, List] = defaultdict(list)
    comp_slc_files: List[Path] = []
    tcorr_files: List[Path] = []

    nrows, ncols = v_all.shape[-2:]
    if mask_file is not None:
        nodata_mask = io.load_gdal(mask_file).astype(bool)
    else:
        nodata_mask = np.zeros((nrows, ncols), dtype=bool)

    if ps_mask_file is not None:
        ps_mask = io.load_gdal(ps_mask_file).astype(bool)
    else:
        ps_mask = np.zeros_like(nodata_mask)

    xhalf, yhalf = half_window["x"], half_window["y"]
    xs, ys = strides["x"], strides["y"]
    out_shape = io.compute_out_shape((nrows, ncols), strides)

    # Solve each ministack using the current chunk (and the previous compressed SLCs)
    ministack_starts = range(0, len(file_list_all), ministack_size)
    for mini_idx, full_stack_idx in enumerate(ministack_starts):
        cur_slice = slice(full_stack_idx, full_stack_idx + ministack_size)
        cur_files = file_list_all[cur_slice].copy()
        cur_dates = date_list_all[cur_slice].copy()

        # Make the current ministack output folder using the start/end dates
        d0, d1 = cur_dates[0], cur_dates[-1]
        start_end = io._format_date_pair(d0, d1)
        cur_output_folder = output_folder / start_end
        cur_output_folder.mkdir(parents=True, exist_ok=True)

        msg = f"Processing {len(cur_files)} files + {len(comp_slc_files)} compressed. "
        msg += f"Output folder: {cur_output_folder}"
        logger.info(msg)
        # Add the existing compressed SLC files to the start
        cur_files = comp_slc_files + cur_files
        cur_vrt = VRTStack(
            cur_files, outfile=cur_output_folder / f"{start_end}.vrt", sort_files=False
        )

        # mini_idx is first non-compressed SLC
        logger.info(
            f"{cur_vrt}: from {Path(cur_vrt.file_list[mini_idx]).name} to"
            f" {Path(cur_vrt.file_list[-1]).name}"
        )
        # Set up the output folder with empty files to write into
        cur_output_files = _setup_output_folder(
            cur_vrt, driver="GTiff", start_idx=mini_idx, strides=strides
        )
        # Save these for the final adjustment later
        # Keep the list of compressed SLCs to prepend to next VRTStack.file_list
        output_slc_files[mini_idx] = cur_output_files

        # Create the empty compressed SLC file
        cur_comp_slc_file = cur_output_folder / f"compressed_{start_end}.tif"
        io.write_arr(
            arr=None,
            like_filename=cur_vrt.outfile,
            output_name=cur_comp_slc_file,
            nbands=1,
            # shape=out_shape,
            # The compressed SLC is the same size as the original SLC
            shape=(nrows, ncols),
        )
        comp_slc_files.append(cur_comp_slc_file)

        # Create the empty compressed temporal coherence file
        tcorr_file = cur_output_folder / f"tcorr_{start_end}.tif"
        io.write_arr(
            arr=None,
            like_filename=cur_vrt.outfile,
            output_name=tcorr_file,
            nbands=1,
            dtype=np.float32,
            shape=out_shape,
        )
        tcorr_files.append(tcorr_file)

        # Iterate over the ministack in blocks
        # Note the overlap to redo the edge effects
        # TODO: adjust the writing to avoid the overlap

        # Note: dividing by len(stack) since cov is shape (rows, cols, nslc, nslc)
        # so we need to load less to not overflow memory
        stack_max_bytes = max_bytes / len(cur_vrt)
        overlaps = (yhalf, xhalf)
        num_blocks = cur_vrt._get_num_blocks(
            max_bytes=stack_max_bytes, overlaps=overlaps
        )
        block_gen = cur_vrt.iter_blocks(
            overlaps=overlaps,
            return_slices=True,
            max_bytes=stack_max_bytes,
            skip_empty=True,
            # TODO: get the nodata value from the vrt stack
            # this involves verifying that COMPASS correctly sets the nodata value
        )
        for cur_data, (rows, cols) in tqdm(block_gen, total=num_blocks):
            if np.all(cur_data == 0):
                continue
            cur_data = cur_data.astype(np.complex64)
            # with logging_redirect_tqdm():
            #     tqdm.write(
            #         f"Processing block ({rows.start}:{rows.stop})/{nrows},"
            #         f" ({cols.start}:{cols.stop})/{ncols}",
            #         end="... ",
            #     )

            # Run the phase linking process on the current ministack
            try:
                cur_mle_stack, tcorr = run_mle(
                    cur_data,
                    half_window=half_window,
                    strides=strides,
                    beta=beta,
                    reference_idx=mini_idx,
                    nodata_mask=nodata_mask[rows, cols],
                    ps_mask=ps_mask[rows, cols],
                    n_workers=n_workers,
                    gpu_enabled=gpu_enabled,
                )
            except PhaseLinkRuntimeError as e:
                # note: this is a warning instead of info, since it should
                # get caught at the "skip_empty" step
                logger.warning(f"Skipping block: {e}")
                continue

            # Save each of the MLE estimates (ignoring the compressed SLCs)
            assert len(cur_mle_stack[mini_idx:]) == len(cur_output_files)
            # Get the location within the output file, shrinking down the slices
            out_row_start = rows.start // ys
            out_col_start = cols.start // xs
            for img, f in zip(cur_mle_stack[mini_idx:], cur_output_files):
                io.write_block(img, f, out_row_start, out_col_start)

            # Save the temporal coherence blocks
            io.write_block(tcorr, tcorr_file, out_row_start, out_col_start)

            # Compress the ministack using only the non-compressed SLCs
            cur_comp_slc = compress(
                cur_data[mini_idx:],
                cur_mle_stack[mini_idx:],
            )
            # Save the compressed SLC block
            io.write_block(
                cur_comp_slc, cur_comp_slc_file, out_row_start, out_col_start
            )
            # logger.debug(f"Saved compressed block SLC to {cur_comp_slc_file}")
            # tqdm.write(" Finished block, loading next block.")

        logger.info(f"Finished ministack {mini_idx} of size {cur_vrt.shape}.")

    ##############################################
    # Set up the output folder with empty files to write into
    # final_output_folder = output_folder / "final"
    # final_output_folder.mkdir(parents=True, exist_ok=True)

    # Average the temporal coherence files in each ministack
    output_tcorr_file = output_folder / "tcorr_average.tif"
    # Find the offsets between stacks by doing a phase linking only compressed SLCs

    # ...But only if we have multiple ministacks
    if len(comp_slc_files) == 1:
        # There was only one ministack, so we can skip this step
        logger.info("Only one ministack, skipping offset calculation.")
        assert len(output_slc_files) == 1
        assert len(tcorr_files) == 1
        for slc_fname in output_slc_files[0]:
            slc_fname.rename(output_folder / slc_fname.name)
        tcorr_files[0].rename(output_tcorr_file)
        return

    # Compute the adjustments by running EVD on the compressed SLCs
    comp_output_folder = output_folder / "adjustments"
    comp_output_folder.mkdir(parents=True, exist_ok=True)
    adjustment_vrt_stack = VRTStack(
        comp_slc_files, outfile=comp_output_folder / "compressed_stack.vrt"
    )

    logger.info(f"Running EVD on compressed files: {adjustment_vrt_stack}")
    adjusted_comp_slc_files = _setup_output_folder(
        adjustment_vrt_stack, driver="GTiff", strides=strides
    )

    # Iterate over the ministack in blocks
    # Note the overlap to redo the edge effects
    block_gen = adjustment_vrt_stack.iter_blocks(
        overlaps=(yhalf, xhalf),
        return_slices=True,
        # Note: dividing by len of stack because cov is shape (rows, cols, nslc, nslc)
        max_bytes=max_bytes / len(adjustment_vrt_stack),
        skip_empty=True,
    )
    for cur_data, (rows, cols) in block_gen:
        msg = f"Processing block {rows.start}:{rows.stop}, {cols.start}:{cols.stop}"
        with logging_redirect_tqdm():
            logger.debug(msg)

        # Run the phase linking process on the current adjustment stack
        cur_mle_stack, tcorr = run_mle(
            cur_data,
            half_window=half_window,
            strides=strides,
            beta=beta,
            reference_idx=0,
            nodata_mask=nodata_mask[rows, cols],
            ps_mask=None,  # PS mask doesn't matter for the adjustments
            use_slc_amp=False,  # Make adjustments unit-amplitude
            n_workers=n_workers,
            gpu_enabled=gpu_enabled,
        )

        # Get the location within the output file, shrinking down the slices
        out_row_start = rows.start // ys
        out_col_start = cols.start // xs
        # Save each of the MLE estimates (ignoring the compressed SLCs)
        for img, f in zip(cur_mle_stack, adjusted_comp_slc_files):
            io.write_block(img, f, out_row_start, out_col_start)
        # Don't think I care about the temporal coherence here

    # Compensate for the offsets between ministacks (aka "datum adjustments")
    for mini_idx, slc_files in output_slc_files.items():
        adjustment_fname = adjusted_comp_slc_files[mini_idx]

        for slc_fname in slc_files:
            logger.info(f"Compensating {slc_fname} with {adjustment_fname}")
            outfile = output_folder / f"{slc_fname.name}"
            VRTInterferogram(
                ref_slc=slc_fname,
                sec_slc=adjustment_fname,
                path=outfile,
                pixel_function="mul",
            )

    # Can pass the list of files to gdal_calc, which interprets it
    # as a multi-band file
    logger.info(f"Averaging temporal coherence files into: {output_tcorr_file}")
    gdal_calc.Calc(
        NoDataValue=nan,
        format="GTiff",
        outfile=output_tcorr_file,
        type="Float32",
        quiet=True,
        overwrite=True,
        creation_options=io.DEFAULT_TIFF_OPTIONS,
        A=tcorr_files,
        calc="numpy.nanmean(A, axis=0)",
    )


def compress(
    slc_stack: np.ndarray,
    mle_estimate: np.ndarray,
):
    """Compress the stack of SLC data using the estimated phase.

    Parameters
    ----------
    slc_stack : np.array
        The stack of complex SLC data, shape (nslc, rows, cols)
    mle_estimate : np.array
        The estimated phase from [`run_mle`][dolphin.phase_link.mle.run_mle],
        shape (nslc, rows // strides['y'], cols // strides['x'])

    Returns
    -------
    np.array
        The compressed SLC data, shape (rows, cols)
    """
    # If the output is downsampled, we need to make `mle_estimate` the same shape
    # as the output
    mle_estimate_upsampled = upsample_nearest(mle_estimate, slc_stack.shape[1:])
    # For each pixel, project the SLCs onto the (normalized) estimated phase
    # by performing a pixel-wise complex dot product
    mle_norm = np.linalg.norm(mle_estimate_upsampled, axis=0)
    # Avoid divide by zero (there may be 0s at the upsampled boundary)
    mle_norm[mle_norm == 0] = np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        return (
            np.nansum(slc_stack * np.conjugate(mle_estimate_upsampled), axis=0)
            / mle_norm
        )


def _setup_output_folder(
    vrt_stack,
    driver: str = "GTiff",
    dtype="complex64",
    start_idx: int = 0,
    strides: Dict[str, int] = {"y": 1, "x": 1},
    creation_options: Optional[List] = None,
) -> List[Path]:
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
    strides : Dict[str, int], optional
        Strides to use when creating the empty files, by default {"y": 1, "x": 1}
        Larger strides will create smaller output files, computed using
        [dolphin.io.compute_out_shape][]
    creation_options : list, optional
        List of options to pass to the GDAL driver, by default None

    Returns
    -------
    List[Path]
        List of saved empty files.
    """
    output_folder = vrt_stack.outfile.parent

    output_files = []
    if isinstance(vrt_stack.dates[0], (list, tuple)):
        # Compressed SLCs will have 2 dates in the name marking the start and end
        date_strs = [io._format_date_pair(d0, d1) for (d0, d1) in vrt_stack.dates]
    else:
        # Otherwise, normal SLC files will have a single date
        date_strs = [d.strftime(io.DEFAULT_DATETIME_FORMAT) for d in vrt_stack.dates]

    rows, cols = vrt_stack.shape[-2:]
    for filename in date_strs[start_idx:]:
        slc_name = Path(filename).stem
        # TODO: get extension from cfg?
        output_path = output_folder / f"{slc_name}.slc.tif"

        io.write_arr(
            arr=None,
            like_filename=vrt_stack.outfile,
            output_name=output_path,
            driver=driver,
            nbands=1,
            dtype=dtype,
            shape=io.compute_out_shape((rows, cols), strides),
            options=creation_options,
        )

        output_files.append(output_path)
    return output_files
