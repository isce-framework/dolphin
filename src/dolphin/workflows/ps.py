from __future__ import annotations

import logging
from pathlib import Path
from pprint import pformat

import opera_utils

import dolphin.ps
from dolphin import __version__
from dolphin._log import log_runtime, setup_logging
from dolphin.io import VRTStack
from dolphin.utils import get_max_memory_usage

from .config import PsWorkflow

logger = logging.getLogger(__name__)


@log_runtime
def run(
    cfg: PsWorkflow,
    compute_looked: bool = False,
    debug: bool = False,
) -> list[Path]:
    """Run the displacement workflow on a stack of SLCs.

    Parameters
    ----------
    cfg : PsWorkflow
        [`PsWorkflow`][dolphin.workflows.config.PsWorkflow] object
        for controlling the workflow.
    compute_looked : bool, optional
        Whether to compute the looked version of the PS mask, by default False.
    debug : bool, optional
        Enable debug logging, by default False.

    """
    # Set the logging level for all `dolphin.` modules
    setup_logging(debug=debug, filename=cfg.log_file)
    logger.debug(pformat(cfg.model_dump()))

    output_file_list = [
        cfg.ps_options._output_file,
        cfg.ps_options._amp_mean_file,
        cfg.ps_options._amp_dispersion_file,
    ]
    ps_output = cfg.ps_options._output_file
    if all(f.exists() for f in output_file_list):
        logger.info(f"Skipping making existing PS files {output_file_list}")
        return output_file_list

    # Check the number of bursts that were passed
    try:
        grouped_slc_files = opera_utils.group_by_burst(cfg.cslc_file_list)
        if len(grouped_slc_files) > 1:
            msg = "Multiple bursts not yet supported for PsWorkflow"
            raise NotImplementedError(msg)
    except ValueError as e:
        # Make sure it's not some other ValueError
        if "Could not parse burst id" not in str(e):
            raise
        # Otherwise, we have SLC files which are not OPERA burst files

    # grab the only key (either a burst, or "") and use that
    cfg.create_dir_tree()

    input_file_list = cfg.cslc_file_list
    if not input_file_list:
        msg = "No input files found"
        raise ValueError(msg)

    # #############################################
    # Make a VRT pointing to the input SLC files
    # #############################################
    subdataset = cfg.input_options.subdataset
    vrt_stack = VRTStack(
        input_file_list,
        subdataset=subdataset,
        outfile=cfg.work_directory / "slc_stack.vrt",
    )

    # Make the nodata mask from the polygons, if we're using OPERA CSLCs
    try:
        nodata_mask_file = cfg.work_directory / "nodata_mask.tif"
        opera_utils.make_nodata_mask(
            vrt_stack.file_list, out_file=nodata_mask_file, buffer_pixels=200
        )
    except Exception as e:
        logger.warning(f"Could not make nodata mask: {e}")
        nodata_mask_file = None

    logger.info(f"Creating persistent scatterer file {ps_output}")
    dolphin.ps.create_ps(
        reader=vrt_stack,
        output_file=output_file_list[0],
        output_amp_mean_file=output_file_list[1],
        output_amp_dispersion_file=output_file_list[2],
        like_filename=vrt_stack.outfile,
        amp_dispersion_threshold=cfg.ps_options.amp_dispersion_threshold,
        block_shape=cfg.worker_settings.block_shape,
    )
    # Save a looked version of the PS mask too
    strides = cfg.output_options.strides
    if compute_looked:
        ps_looked_file, amp_disp_looked = dolphin.ps.multilook_ps_files(
            strides=strides,
            ps_mask_file=cfg.ps_options._output_file,
            amp_dispersion_file=cfg.ps_options._amp_dispersion_file,
        )
        output_file_list.append(ps_looked_file)
        output_file_list.append(amp_disp_looked)

    # Print the maximum memory usage for each worker
    max_mem = get_max_memory_usage(units="GB")
    logger.info(f"Maximum memory usage: {max_mem:.2f} GB")
    logger.info(f"Config file dolphin version: {cfg._dolphin_version}")
    logger.info(f"Current running dolphin version: {__version__}")

    return output_file_list
