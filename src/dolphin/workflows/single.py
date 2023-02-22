"""Estimate wrapped phase for one updated SLC using the online algorithm.

References
----------
    [1] Mirzaee, Sara, Falk Amelung, and Heresh Fattahi. "Non-linear phase
    linking using joined distributed and persistent scatterers." Computers &
    Geosciences (2022): 105291.
"""
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from dolphin import io
from dolphin._log import get_log
from dolphin._types import Filename
from dolphin.phase_link import PhaseLinkRuntimeError, compress, run_mle
from dolphin.stack import VRTStack

from ._utils import setup_output_folder

logger = get_log(__name__)

__all__ = ["run_evd_update"]


def run_evd_update(
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
        ps_mask = io.load_gdal(ps_mask_file, masked=True)
        # Fill the nodata values with false
        ps_mask = ps_mask.astype(bool).filled(False)
    else:
        ps_mask = np.zeros_like(nodata_mask)

    xhalf, yhalf = half_window["x"], half_window["y"]
    xs, ys = strides["x"], strides["y"]

    # If we were passed any compressed SLCs in `file_list_all`,
    # then we want that index for when we create new compressed SLCs.
    # We skip the old compressed SLCs to create new ones
    first_non_comp_idx = 0
    for filename in file_list_all:
        if not Path(filename).name.startswith("compressed"):
            break
        first_non_comp_idx += 1

    # Solve each ministack using the current chunk (and the previous compressed SLCs)
    ministack_starts = range(0, len(file_list_all), ministack_size)
    for mini_idx, full_stack_idx in enumerate(ministack_starts):
        cur_slice = slice(full_stack_idx, full_stack_idx + ministack_size)
        cur_files = file_list_all[cur_slice].copy()
        cur_dates = date_list_all[cur_slice].copy()

        # Make the current ministack output folder using the start/end dates
        d0 = cur_dates[first_non_comp_idx][0]
        d1 = cur_dates[-1][0]
        start_end = io._format_date_pair(d0, d1)
        cur_output_folder = output_folder / start_end
        cur_output_folder.mkdir(parents=True, exist_ok=True)

        msg = (
            f"Processing {len(cur_files) - first_non_comp_idx} SLCs +"
            f" {len(comp_slc_files) + first_non_comp_idx} compressed SLCs. "
        )
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
        # Create the background writer for this ministack
        writer = io.Writer()

        # mini_idx is first non-compressed SLC
        logger.info(
            f"{cur_vrt}: from {Path(cur_vrt.file_list[mini_idx]).name} to"
            f" {Path(cur_vrt.file_list[-1]).name}"
        )
        # Set up the output folder with empty files to write into
        cur_output_files = setup_output_folder(
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
            # Note that the compressed SLC is the same size as the original SLC
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
            strides=strides,
        )
        tcorr_files.append(tcorr_file)

        # Iterate over the ministack in blocks
        # Note the overlap to redo the edge effects
        # TODO: adjust the writing to avoid the overlap

        # Note: dividing by len(stack) since cov is shape (rows, cols, nslc, nslc)
        # so we need to load less to not overflow memory
        stack_max_bytes = max_bytes / len(cur_vrt)
        overlaps = (yhalf, xhalf)
        block_gen = cur_vrt.iter_blocks(
            overlaps=overlaps,
            max_bytes=stack_max_bytes,
            skip_empty=True,
            # TODO: get the nodata value from the vrt stack
            # this involves verifying that COMPASS correctly sets the nodata value
        )
        for cur_data, (rows, cols) in block_gen:
            if np.all(cur_data == 0):
                continue
            cur_data = cur_data.astype(np.complex64)

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
                logger.warning(f"Exception at ({rows}, {cols}): {e}")
                continue

            # Save each of the MLE estimates (ignoring the compressed SLCs)
            assert len(cur_mle_stack[mini_idx:]) == len(cur_output_files)
            # Get the location within the output file, shrinking down the slices
            out_row_start = rows.start // ys
            out_col_start = cols.start // xs
            for img, f in zip(cur_mle_stack[mini_idx:], cur_output_files):
                writer.queue_write(img, f, out_row_start, out_col_start)

            # Save the temporal coherence blocks
            writer.queue_write(tcorr, tcorr_file, out_row_start, out_col_start)

            # Compress the ministack using only the non-compressed SLCs
            cur_comp_slc = compress(
                cur_data[mini_idx:],
                cur_mle_stack[mini_idx:],
            )
            # Save the compressed SLC block
            writer.queue_write(
                cur_comp_slc, cur_comp_slc_file, out_row_start, out_col_start
            )
            # logger.debug(f"Saved compressed block SLC to {cur_comp_slc_file}")
            # tqdm.write(" Finished block, loading next block.")

        # Block until all the writers for this ministack have finished
        logger.info(f"Waiting to write {writer.num_queued} blocks of data.")
        writer.notify_finished()
        logger.info(f"Finished ministack {mini_idx} of size {cur_vrt.shape}.")

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
        for slc_fname in output_slc_files[first_non_comp_idx]:
            slc_fname.rename(output_folder / slc_fname.name)

        tcorr_files[0].rename(output_tcorr_file)

        output_comp_slc_file = output_folder / comp_slc_files[0].name
        comp_slc_files[0].rename(output_comp_slc_file)
        return
