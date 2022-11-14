"""Estimate wrapped phase using batches of ministacks."""
from collections import defaultdict
from math import nan
from pathlib import Path
from typing import List

import numpy as np
from osgeo_utils import gdal_calc

from dolphin import io
from dolphin.log import get_log
from dolphin.phase_link.mle import compress
from dolphin.phase_link.mle_gpu import run_mle_gpu
from dolphin.utils import Pathlike, get_raster_xysize
from dolphin.vrt import VRTStack

logger = get_log()


def run_evd_sequential(
    *,
    slc_vrt_file: Pathlike,
    # weight_file: Pathlike,
    output_folder: Pathlike,
    window: dict,
    ministack_size: int = 10,
    mask_file: Pathlike = None,
    ps_mask_file: Pathlike = None,
    beta: float = 0.1,
    max_bytes: float = 100e6,
):
    """Estimate wrapped phase using batches of ministacks."""
    output_folder = Path(output_folder)
    v_all = VRTStack.from_vrt_file(slc_vrt_file)
    file_list_all = v_all.file_list
    logger.info(f"{v_all}: from {v_all.file_list[0]} to {v_all.file_list[-1]}")

    comp_slc_files: List[Path] = []
    output_slc_files = defaultdict(list)  # Map of {ministack_index: [output_slc_files]}

    if mask_file is not None:
        mask = io.load_gdal(mask_file).astype(bool)
    else:
        xsize, ysize = get_raster_xysize(v_all.file_list[0])
        mask = np.zeros((ysize, xsize), dtype=bool)

    if ps_mask_file is not None:
        ps_mask = io.load_gdal(ps_mask_file).astype(bool)
    else:
        ps_mask = np.zeros_like(mask)

    xhalf, yhalf = window["xhalf"], window["yhalf"]
    # Solve each ministack using the current chunk (and the previous compressed SLCs)
    ministack_starts = range(0, len(file_list_all), ministack_size)
    for mini_idx, full_stack_idx in enumerate(ministack_starts):
        cur_files = file_list_all[
            full_stack_idx : full_stack_idx + ministack_size
        ].copy()
        name_start, name_end = cur_files[0].stem, cur_files[-1].stem
        start_end = f"{name_start}_{name_end}"

        # Make a new output folder for each ministack
        cur_output_folder = output_folder / start_end
        cur_output_folder.mkdir(parents=True, exist_ok=True)
        msg = f"Processing {len(cur_files)} files + {len(comp_slc_files)} compressed. "
        msg += f"Output folder: {cur_output_folder}"
        logger.info(msg)
        # Add the existing compressed SLC files to the start
        cur_files = comp_slc_files + cur_files
        cur_vrt = VRTStack(cur_files, outfile=cur_output_folder / f"{start_end}.vrt")

        # mini_idx is first non-compressed SLC
        logger.info(
            f"{cur_vrt}: from {Path(cur_vrt.file_list[mini_idx]).name} to"
            f" {Path(cur_vrt.file_list[-1]).name}"
        )
        # Set up the output folder with empty files to write into
        cur_output_files, cur_comp_slc_file = io.setup_output_folder(
            cur_vrt, driver="GTiff", start_idx=mini_idx, make_compressed=True
        )
        # Save these for the final adjustment later
        output_slc_files[mini_idx] = cur_output_files
        # Keep the list of compressed SLCs to prepend to next VRTStack.file_list
        assert cur_comp_slc_file is not None  # note: this is for mypy
        comp_slc_files.append(cur_comp_slc_file)

        # Iterate over the ministack in blocks
        # Note the overlap to redo the edge effects
        block_gen = cur_vrt.iter_blocks(
            overlaps=(yhalf, xhalf),
            return_slices=True,
            max_bytes=max_bytes,
            skip_empty=True,
        )
        for cur_data, (rows, cols) in block_gen:
            logger.debug(
                f"Processing block {rows.start}:{rows.stop}, {cols.start}:{cols.stop}"
            )

            # Run the phase linking process on the current ministack
            cur_mle_stack = run_mle_gpu(
                cur_data,
                half_window=(xhalf, yhalf),
                beta=beta,
                reference_idx=mini_idx,
                mask=mask[rows, cols],
                ps_mask=ps_mask[rows, cols],
            )

            # Save each of the MLE estimates (ignoring the compressed SLCs)
            assert len(cur_mle_stack[mini_idx:]) == len(cur_output_files)
            io.save_block(cur_mle_stack[mini_idx:], cur_output_files, rows, cols)

            # Compress the ministack using only the non-compressed SLCs
            cur_comp_slc = compress(cur_data[mini_idx:], cur_mle_stack[mini_idx:])
            # Save the compressed SLC
            logger.debug(f"Saving compressed block SLC to {cur_comp_slc_file}")
            io.save_block(cur_comp_slc, cur_comp_slc_file, rows, cols)

        logger.info(f"Finished ministack {mini_idx} of size {cur_vrt.shape}.")

    # Find the offsets between stacks by doing a phase linking only compressed SLCs
    comp_output_folder = output_folder / "adjustments"
    comp_output_folder.mkdir(parents=True, exist_ok=True)
    adjustment_vrt_stack = VRTStack(
        comp_slc_files, outfile=comp_output_folder / "compressed_stack.vrt"
    )
    logger.info(f"Running EVD on compressed files: {adjustment_vrt_stack}")

    ##############################################
    # Set up the output folder with empty files to write into
    adjusted_comp_slc_files, _ = io.setup_output_folder(
        adjustment_vrt_stack,
        driver="GTiff",
        start_idx=0,
        make_compressed=False,
    )

    # Iterate over the ministack in blocks
    # Note the overlap to redo the edge effects
    block_gen = adjustment_vrt_stack.iter_blocks(
        overlaps=(yhalf, xhalf),
        return_slices=True,
        max_bytes=max_bytes,
        skip_empty=True,
    )
    for cur_data, (rows, cols) in block_gen:
        logger.info(
            f"Processing block {rows.start}:{rows.stop}, {cols.start}:{cols.stop}"
        )

        # Run the phase linking process on the current ministack
        cur_mle_stack = run_mle_gpu(
            cur_data,
            half_window=(xhalf, yhalf),
            beta=beta,
            reference_idx=0,
            mask=mask[rows, cols],
            ps_mask=ps_mask[rows, cols],
        )

        # Save each of the MLE estimates (ignoring the compressed SLCs)
        io.save_block(cur_mle_stack, adjusted_comp_slc_files, rows, cols)

    # TODO: TCORR!
    # Compensate for the offsets between ministacks (aka "datum adjustments")
    final_output_folder = output_folder / "final"
    # TODO: do i need to separate out these?
    # final_output_folder = output_folder
    final_output_folder.mkdir(parents=True, exist_ok=True)
    for mini_idx, slc_files in output_slc_files.items():
        adjustment_fname = adjusted_comp_slc_files[mini_idx]
        # driver = "ENVI"
        driver = "GTiff"
        for slc_fname in slc_files:
            logger.info(f"Compensating {slc_fname} with {adjustment_fname}")
            outfile = final_output_folder / f"{slc_fname.name}"

            gdal_calc.Calc(
                NoDataValue=nan,
                format=driver,
                outfile=outfile,
                A=slc_fname,
                B=adjustment_fname,
                calc="abs(A) * exp(1j * (angle(A) + angle(B)))",
                quiet=True,
                overwrite=True,
                creation_options=io.DEFAULT_TIFF_OPTIONS,
            )
            # TODO: need to copy projection?
            # TODO: delete old?

    # Average the temporal coherence files in each ministack
