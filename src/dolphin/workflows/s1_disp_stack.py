#!/usr/bin/env python
# from dolphin import phase_link, ps, sequential, unwrap, utils, vrt
from dolphin._log import get_log, log_runtime

from .config import Workflow


@log_runtime
def run(cfg: Workflow, debug: bool = False):
    """Run the displacement workflow on a stack of SLCs.

    Parameters
    ----------
    cfg : Workflow
        [Workflow][dolphin.workflows.config.Workflow] object with workflow parameters
    debug : bool, optional
        Enable debug logging, by default False.
    """
    logger = get_log(debug=debug)
    # output_dir = cfg.outputs.output_directory.absolute()
    # scratch_dir = cfg.outputs.scratch_directory

    input_file_list = cfg.inputs.cslc_file_list
    if not input_file_list:
        raise ValueError("No input files found")

    # #############################################
    # 0. Make a VRT pointing to the input SLC files
    # #############################################
    # slc_vrt_file = scratch_dir / "slc_stack.vrt"
    # vrt_stack = vrt.VRTStack(input_file_list, outfile=slc_vrt_file)
    # vrt_stack.write()

    # ###############
    # 1. PS selection
    # ###############
    ps_output = cfg.ps_options.output_file
    if ps_output.exists():
        logger.info(f"Skipping making existing PS file {ps_output}")
    else:
        logger.info(f"Creating persistent scatterer file {ps_output}")
        # ps.create_ps(cfg.ps_options)

    # #########################
    # 2. phase linking/EVD step
    # #########################
    pl_path = cfg.phase_linking.directory

    existing_files = list(pl_path.glob("*.h5"))  # TODO: get ext from config
    if len(existing_files) > 0:
        logger.info(f"Skipping EVD step, {len(existing_files)} files already exist")
    else:
        logger.info(f"Running sequential EMI step in {pl_path}")
        # pl_path = sequential.run_evd_sequential( ... )

    # ###################################################
    # 3. Form interferograms from estimated wrapped phase
    # ###################################################
    # existing_ifgs = list(cfg.interferograms.directory.glob("*.int*"))
    # if len(existing_ifgs) > 0:
    #     logger.info(f"Skipping interferogram step, {len(existing_ifgs)} exists")
    # else:
    #     logger.info(f"Running interferogram formation ")
    #    # The python MLE function handles the temp coh, and the PS phase insertion
    #    # interferograms.form_ifgs( ... )

    # ###################################
    # 4. Stitch and Unwrap interferograms
    # ###################################
    # TODO: will this be a separate workflow?
    # Or will we loop through all bursts, then stitch, then unwrap all here?

    if not cfg.unwrap_options.run_unwrap:
        logger.info("Skipping unwrap step")
        return

    # unwrap.run( ... )

    # ####################
    # 5. Phase Corrections
    # ####################
    # TODO: Determine format for the tropospheric/ionospheric phase correction
