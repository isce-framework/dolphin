#!/usr/bin/env python
from dolphin import ps, sequential, vrt
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
    scratch_dir = cfg.outputs.scratch_directory

    input_file_list = cfg.inputs.cslc_file_list
    if not input_file_list:
        raise ValueError("No input files found")

    # #############################################
    # 0. Make a VRT pointing to the input SLC files
    # #############################################
    subdataset = cfg.inputs.subdataset
    vrt_stack = vrt.VRTStack(
        input_file_list, subdataset=subdataset, outfile=scratch_dir / "slc_stack.vrt"
    )

    # ###############
    # 1. PS selection
    # ###############
    ps_output = cfg.ps_options.output_file
    if ps_output.exists():
        logger.info(f"Skipping making existing PS file {ps_output}")
    else:
        logger.info(f"Creating persistent scatterer file {ps_output}")
        ps.create_ps(
            slc_vrt_file=vrt_stack.outfile,
            output_file=cfg.ps_options.output_file,
            amp_mean_file=cfg.ps_options.amp_mean_file,
            amp_dispersion_file=cfg.ps_options.amp_dispersion_file,
            amp_dispersion_threshold=cfg.ps_options.amp_dispersion_threshold,
            max_ram_gb=cfg.worker_settings.max_ram_gb,
        )

    # #########################
    # 2. phase linking/EVD step
    # #########################
    pl_path = cfg.phase_linking.directory

    existing_files = list(pl_path.glob("*.h5"))  # TODO: get ext from config
    if len(existing_files) > 0:
        logger.info(f"Skipping EVD step, {len(existing_files)} files already exist")
    else:
        logger.info(f"Running sequential EMI step in {pl_path}")
        pl_path = sequential.run_evd_sequential(
            slc_vrt_file=vrt_stack.outfile,
            output_folder=cfg.phase_linking.directory,
            half_window=cfg.phase_linking.half_window.dict(),
            strides=cfg.outputs.strides,
            ministack_size=cfg.phase_linking.ministack_size,
            # mask_file=cfg.inputs.mask_file,
            ps_mask_file=cfg.ps_options.output_file,
            max_bytes=cfg.worker_settings.max_ram_gb * 1e9,
            n_workers=cfg.worker_settings.n_workers,
            gpu_enabled=cfg.worker_settings.gpu_enabled,
            beta=0.0,
        )

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

    # output_dir = cfg.outputs.output_directory.absolute()
    # unwrap.run( ... )

    # ####################
    # 5. Phase Corrections
    # ####################
    # TODO: Determine format for the tropospheric/ionospheric phase correction
