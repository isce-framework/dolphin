"""Estimate wrapped phase using batches of ministacks."""
import string
from collections import defaultdict
from math import nan
from pathlib import Path
from typing import Dict, List

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

    # Map of {ministack_index: [output_slc_files]}
    output_slc_files: Dict[int, List] = defaultdict(list)
    comp_slc_files: List[Path] = []
    tcorr_files: List[Path] = []

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
        cur_vrt.write()

        # mini_idx is first non-compressed SLC
        logger.info(
            f"{cur_vrt}: from {Path(cur_vrt.file_list[mini_idx]).name} to"
            f" {Path(cur_vrt.file_list[-1]).name}"
        )
        # Set up the output folder with empty files to write into
        cur_output_files = io.setup_output_folder(
            cur_vrt, driver="GTiff", start_idx=mini_idx
        )
        # Save these for the final adjustment later
        # Keep the list of compressed SLCs to prepend to next VRTStack.file_list
        output_slc_files[mini_idx] = cur_output_files

        # Create the empty compressed SLC file
        cur_comp_slc_file = cur_output_folder / f"compressed_{start_end}.tif"
        io.save_arr_like(
            arr=None,
            like_filename=cur_vrt.outfile,
            output_name=cur_comp_slc_file,
            nbands=1,
        )
        comp_slc_files.append(cur_comp_slc_file)

        # Create the empty compressed temporal coherence file
        tcorr_file = cur_output_folder / f"tcorr_{start_end}.tif"
        io.save_arr_like(
            arr=None,
            like_filename=cur_vrt.outfile,
            output_name=tcorr_file,
            nbands=1,
            dtype=np.float32,
        )
        tcorr_files.append(tcorr_file)

        # Iterate over the ministack in blocks
        # Note the overlap to redo the edge effects
        # TODO: adjust the writing to avoid the overlap
        block_gen = cur_vrt.iter_blocks(
            overlaps=(yhalf, xhalf),
            return_slices=True,
            max_bytes=max_bytes,
            skip_empty=True,
            # TODO: get the nodata value from the vrt stack
            # this involves verifying that COMPASS correctly sets the nodata value
        )
        for cur_data, (rows, cols) in block_gen:
            logger.debug(
                f"Processing block {rows.start}:{rows.stop}, {cols.start}:{cols.stop}"
            )

            # Run the phase linking process on the current ministack
            cur_mle_stack, tcorr = run_mle_gpu(
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
            # Save the temporal coherence blocks
            io.save_block(tcorr, tcorr_file, rows, cols)

            # Compress the ministack using only the non-compressed SLCs
            cur_comp_slc = compress(cur_data[mini_idx:], cur_mle_stack[mini_idx:])
            # Save the compressed SLC block
            logger.debug(f"Saving compressed block SLC to {cur_comp_slc_file}")
            io.save_block(cur_comp_slc, cur_comp_slc_file, rows, cols)

        logger.info(f"Finished ministack {mini_idx} of size {cur_vrt.shape}.")

    # Find the offsets between stacks by doing a phase linking only compressed SLCs
    comp_output_folder = output_folder / "adjustments"
    comp_output_folder.mkdir(parents=True, exist_ok=True)
    adjustment_vrt_stack = VRTStack(
        comp_slc_files, outfile=comp_output_folder / "compressed_stack.vrt"
    )
    adjustment_vrt_stack.write()
    logger.info(f"Running EVD on compressed files: {adjustment_vrt_stack}")

    ##############################################
    # Set up the output folder with empty files to write into
    adjusted_comp_slc_files = io.setup_output_folder(
        adjustment_vrt_stack, driver="GTiff"
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
        logger.debug(
            f"Processing block {rows.start}:{rows.stop}, {cols.start}:{cols.stop}"
        )

        # Run the phase linking process on the current ministack
        cur_mle_stack, tcorr = run_mle_gpu(
            cur_data,
            half_window=(xhalf, yhalf),
            beta=beta,
            reference_idx=0,
            mask=mask[rows, cols],
            ps_mask=ps_mask[rows, cols],
        )

        # Save each of the MLE estimates (ignoring the compressed SLCs)
        io.save_block(cur_mle_stack, adjusted_comp_slc_files, rows, cols)
        # TODO: Do I care about the temporal coherence here?
        # What would it even mean for the all-compressed SLCs?

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
    # Get a sequence of "A", "B", "C", etc. for gdal_calc
    # note: string.ascii_letters = 'abcd...xyzABC...XYZ'
    name_dict = dict(zip(string.ascii_letters, tcorr_files))
    # TODO: how should i fix this when we have more than 52 ministacks?
    # 52 * 20 = 1040, so we should be fine for a while
    assert len(tcorr_files) <= 52
    names = name_dict.keys()
    # Get the numpy string taking the nanmean of all rasters
    # e.g. "nanmean(stack(A, B, C, D), axis=0)"
    calc_str = f"nanmean(stack( ({','.join(names)}), axis=0))"

    driver = "GTiff"
    output_tcorr_file = final_output_folder / "tcorr_average.tif"
    logger.info(f"Averaging temporal coherence files into: {output_tcorr_file}")
    # Make a dict so we can unpack any number of files
    gdal_calc.Calc(
        NoDataValue=nan,
        format=driver,
        outfile=output_tcorr_file,
        quiet=True,
        overwrite=True,
        creation_options=io.DEFAULT_TIFF_OPTIONS,
        calc=calc_str,
        **name_dict,
    )
