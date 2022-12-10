#!/usr/bin/env python
# from glob import glob
# from os import fspath
# from pathlib import Path

# from dolphin import combine_ps_ds, phase_link, ps, sequential, unwrap, utils, vrt
from dolphin import sequential, vrt
from dolphin._log import get_log, log_runtime

from .config import Workflow


@log_runtime
def run(cfg: Workflow, debug: bool = False):
    """Run the displacement workflow on a stack of SLCs.

    Parameters
    ----------
    cfg : Config
        [dolphin.Config][] object with workflow parameters
    debug : bool, optional
        Enable debug logging, by default False.
    """
    logger = get_log(debug=debug)
    # output_dir = cfg.outputs.output_directory.absolute()
    # scratch_dir = cfg.outputs.scratch_directory
    scratch_dir = cfg.outputs.scratch_directory
    input_file_list = cfg.inputs.cslc_file_list
    if not input_file_list:
        raise ValueError("No input files found")

    # dem_file = full_cfg["dynamic_ancillary_file_group"]["dem_file"]

    # 0. Make a VRT pointing to the input SLC files
    slc_vrt_file = scratch_dir / "slc_stack.vrt"
    vrt_stack = vrt.VRTStack(input_file_list, outfile=slc_vrt_file)

    if slc_vrt_file.exists():
        logger.info(f"Skipping creating VRT to SLC stack {slc_vrt_file}")
    else:
        logger.info(f"Creating VRT to SLC stack file {slc_vrt_file}")
        vrt_stack.write()

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
        pl_path = sequential.run_evd_sequential(
            slc_vrt_file=slc_vrt_file,
            # weight_file: Filename,
            output_folder=pl_path,
            window=cfg["window"],
            strides=cfg["strides"],
            ministack_size=cfg["phase_linking"]["ministack_size"],
            max_bytes=1e6 * cfg["ram"],
            n_workers=cfg.worker_settings.n_workers,
            no_gpu=not cfg.worker_settings.gpu_enabled,
            # mask_file: Filename = None,
            # ps_mask_file=ps_output,
            # beta=0.1,
            # beta=cgf["phase_linking"]["beta"],
        )

    # ###################################################
    # 4. Form interferograms from estimated wrapped phase
    # ###################################################
    ps_ds_path = scratch_dir / cfg["combine_ps_ds"]["directory"]
    ps_ds_path.mkdir(parents=True, exist_ok=True)

    # # Combine the temp coh files made in last step:
    # temp_coh_file = pl_path / "tcorr_average.tif"

    existing_ifgs = list(ps_ds_path.glob("*.int*"))
    if len(existing_ifgs) > 0:
        logger.info(f"Skipping combine_ps_ds step, {len(existing_ifgs)} exists")
    else:
        logger.info(f"Running combine ps/ds step into {ps_ds_path}")

        # The python MLE function handles the temp coh, and the PS phase insertion
        # interferograms.form_ifgs( ... )

    # 6. Unwrap interferograms
    # # TODO: either combine, or figure out if we need multiple masks
    # # TODO: Do we create a new mask file here based on temporal coherence?
    # mask_files = cfg.inputs.mask_files
    # if len(mask_files) >= 1:
    #     combined_mask_file = utils.combine_mask_files(mask_files, scratch_dir)
    # else:
    #     combined_mask_file = None

    if not cfg.unwrap_options.run_unwrap:
        logger.info("Skipping unwrap step")
        return

    # unwrap.run(
    #     ifg_path=ps_ds_path,
    #     output_path=cfg.unwrap_options.output_directory,
    #     # cor_file=temp_coh_ps_ds_file,
    #     cor_file=cfg.phase_link.temp_coh_file,
    #     mask_file=combined_mask_file,
    # )
