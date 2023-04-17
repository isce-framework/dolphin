"""Estimate wrapped phase using batches of ministacks.

References
----------
    [1] Ansari, H., De Zan, F., & Bamler, R. (2017). Sequential estimator: Toward
    efficient InSAR time series analysis. IEEE Transactions on Geoscience and
    Remote Sensing, 55(10), 5637-5652.
"""
from collections import defaultdict
from itertools import chain
from os import fspath
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from osgeo_utils import gdal_calc

from dolphin import io
from dolphin._log import get_log
from dolphin._types import Filename
from dolphin.interferogram import VRTInterferogram
from dolphin.phase_link import run_mle
from dolphin.stack import VRTStack

from ._utils import setup_output_folder
from .single import run_evd_single

logger = get_log(__name__)

__all__ = ["run_evd_sequential"]


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
    beta: float = 0.01,
    max_bytes: float = 32e6,
    n_workers: int = 1,
    gpu_enabled: bool = True,
) -> Tuple[List[Path], List[Path], Path]:
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
        ps_mask = io.load_gdal(ps_mask_file, masked=True)
        # Fill the nodata values with false
        ps_mask = ps_mask.astype(bool).filled(False)
    else:
        ps_mask = np.zeros_like(nodata_mask)

    xhalf, yhalf = half_window["x"], half_window["y"]
    xs, ys = strides["x"], strides["y"]

    # Solve each ministack using the current chunk (and the previous compressed SLCs)
    ministack_starts = range(0, len(file_list_all), ministack_size)
    for mini_idx, full_stack_idx in enumerate(ministack_starts):
        cur_slice = slice(full_stack_idx, full_stack_idx + ministack_size)
        cur_files = file_list_all[cur_slice].copy()
        cur_dates = date_list_all[cur_slice].copy()

        # Make the current ministack output folder using the start/end dates
        d0 = cur_dates[0][0]
        d1 = cur_dates[-1][0]
        start_end = io._format_date_pair(d0, d1)
        cur_output_folder = output_folder / start_end
        cur_output_folder.mkdir(parents=True, exist_ok=True)

        msg = f"Processing {len(cur_files)} SLCs."
        msg += f"Output folder: {cur_output_folder}"
        logger.info(msg)
        # Add the existing compressed SLC files to the start
        cur_files = comp_slc_files + cur_files
        cur_vrt = VRTStack(
            cur_files,
            outfile=cur_output_folder / f"{start_end}.vrt",
            sort_files=False,
            subdataset=v_all.subdataset,
        )
        cur_output_files, cur_comp_slc_file, tcorr_file = run_evd_single(
            slc_vrt_file=cur_vrt,
            output_folder=cur_output_folder,
            half_window=half_window,
            strides=strides,
            reference_idx=mini_idx,
            mask_file=mask_file,
            ps_mask_file=ps_mask_file,
            beta=beta,
            max_bytes=max_bytes,
            n_workers=n_workers,
            gpu_enabled=gpu_enabled,
        )

        output_slc_files[mini_idx] = cur_output_files
        comp_slc_files.append(cur_comp_slc_file)
        tcorr_files.append(tcorr_file)

    ##############################################
    # Set up the output folder with empty files to write into

    # Average the temporal coherence files in each ministack
    # TODO: do we want to include the date span in this filename?
    output_tcorr_file = output_folder / "tcorr_average.tif"
    # Find the offsets between stacks by doing a phase linking only compressed SLCs
    # (But only if we have >1 ministacks. If only one, just rename the outputs)
    if len(comp_slc_files) == 1:
        # There was only one ministack, so we can skip this step
        logger.info("Only one ministack, skipping offset calculation.")
        assert len(output_slc_files) == 1
        assert len(tcorr_files) == 1
        for slc_fname in output_slc_files[0]:
            slc_fname.rename(output_folder / slc_fname.name)

        tcorr_files[0].rename(output_tcorr_file)

        output_comp_slc_file = output_folder / comp_slc_files[0].name
        comp_slc_files[0].rename(output_comp_slc_file)

        # return output_slc_files, comp_slc_file, tcorr_file
        # different here for sequential
        return output_slc_files[0], [output_comp_slc_file], output_tcorr_file

    # Compute the adjustments by running EVD on the compressed SLCs
    comp_output_folder = output_folder / "adjustments"
    comp_output_folder.mkdir(parents=True, exist_ok=True)
    adjustment_vrt_stack = VRTStack(
        comp_slc_files, outfile=comp_output_folder / "compressed_stack.vrt"
    )

    logger.info(f"Running EVD on compressed files: {adjustment_vrt_stack}")
    adjusted_comp_slc_files = setup_output_folder(
        adjustment_vrt_stack,
        driver="GTiff",
        strides=strides,
        nodata=0,
    )

    writer = io.Writer()
    # Iterate over the ministack in blocks
    # Note the overlap to redo the edge effects
    block_gen = adjustment_vrt_stack.iter_blocks(
        overlaps=(yhalf, xhalf),
        # Note: dividing by len of stack because cov is shape (rows, cols, nslc, nslc)
        max_bytes=max_bytes / len(adjustment_vrt_stack),
        skip_empty=True,
    )
    for cur_data, (rows, cols) in block_gen:
        msg = f"Processing block {rows.start}:{rows.stop}, {cols.start}:{cols.stop}"
        logger.debug(msg)

        # Run the phase linking process on the current adjustment stack
        cur_mle_stack, _ = run_mle(
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
        np.nan_to_num(cur_mle_stack, copy=False)

        # Get the location within the output file, shrinking down the slices
        out_row_start = rows.start // ys
        out_col_start = cols.start // xs
        # Save each of the MLE estimates (ignoring the compressed SLCs)
        for img, f in zip(cur_mle_stack, adjusted_comp_slc_files):
            writer.queue_write(img, f, out_row_start, out_col_start)
        # Don't think I care about the temporal coherence here

    writer.notify_finished()
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
        NoDataValue=0,
        format="GTiff",
        outfile=fspath(output_tcorr_file),
        type="Float32",
        quiet=True,
        overwrite=True,
        creation_options=io.DEFAULT_TIFF_OPTIONS,
        A=tcorr_files,
        calc="numpy.nanmean(A, axis=0)",
    )

    # Combine the separate SLC output lists into a single list
    all_slc_files = list(chain.from_iterable(output_slc_files.values()))
    return all_slc_files, adjusted_comp_slc_files, output_tcorr_file
