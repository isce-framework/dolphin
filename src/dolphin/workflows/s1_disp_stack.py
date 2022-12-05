#!/usr/bin/env python
from glob import glob
from os import fspath
from pathlib import Path

# from dolphin import phase_link, ps, sequential, unwrap, utils, vrt
from dolphin.log import get_log, log_runtime


@log_runtime
def run(full_cfg: dict, debug: bool = False):
    """Run the displacement workflow on a stack of SLCs.

    Parameters
    ----------
    full_cfg : dict
        Loaded configuration from YAML workflow file.
    debug : bool, optional
        Enable debug logging, by default False.
    """
    logger = get_log(debug=debug)
    cfg = full_cfg["processing"]
    output_dir = Path(full_cfg["product_path_group"]["product_path"]).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(full_cfg["product_path_group"]["scratch_path"]).absolute()
    scratch_dir.mkdir(parents=True, exist_ok=True)
    # sas_output_file = full_cfg["product_path_group"]["sas_output_file"]
    # gpu_enabled = full_cfg["worker"]["gpu_enabled"]
    # n_workers = full_cfg["worker"]["n_workers"]

    input_file_list = full_cfg["input_file_group"]["cslc_file_list"]
    input_file_path = full_cfg["input_file_group"]["cslc_file_path"]
    if not input_file_list:
        if not input_file_path:
            raise ValueError("Must specify either cslc_file_list or cslc_file_path")

        input_file_path = Path(input_file_path).absolute()
        ext = full_cfg["input_file_group"]["cslc_file_ext"]
        # TODO : somehow accommodate inputs other than ENVI
        input_file_list = sorted(glob(fspath(input_file_path / f"*{ext}")))

    # #############################################
    # 0. Make a VRT pointing to the input SLC files
    # #############################################
    # slc_vrt_file = scratch_dir / "slc_stack.vrt"
    # vrt_stack = vrt.VRTStack(input_file_list, outfile=slc_vrt_file)
    # vrt_stack.write()

    # #######################
    # 1. Amplitude dispersion
    # #######################
    ps_path = scratch_dir / cfg["ps"]["directory"]
    ps_path.mkdir(parents=True, exist_ok=True)
    amp_disp_file = ps_path / full_cfg["dynamic_ancillary_file_group"]["amp_disp_file"]
    # amp_mean_file = ps_path / full_cfg["dynamic_ancillary_file_group"]["amp_mean_file"]

    if amp_disp_file.exists():
        logger.info(f"Skipping existing amplitude dispersion file {amp_disp_file}")
    else:
        logger.info(f"Making amplitude dispersion file {amp_disp_file}")
        # Create the amplitude dispersion using PS module

    # ###############
    # 2. PS selection
    # ###############
    ps_output = ps_path / cfg["ps_file"]
    # threshold = cfg["ps"]["amp_dispersion_threshold"]
    if ps_output.exists():
        logger.info(f"Skipping making existing PS file {ps_output}")
    else:
        logger.info(f"Creating persistent scatterer file {ps_output}")
        # ps.create_ps(...)

    # ###############################
    # nmap: find SHP neighborhoods
    # ###############################
    # # (This will be optional, possibly entirely skipped if it's too slow)
    # nmap_path = scratch_dir / cfg["nmap"]["directory"]
    # nmap_path.mkdir(parents=True, exist_ok=True)
    # weight_file = nmap_path / cfg["weight_file"]
    # # threshold = cfg["ps"]["amp_dispersion_threshold"]
    # if not cfg["nmap"]["run_nmap"] or weight_file.exists():
    #     logger.info(f"Skipping making existing NMAP file {weight_file}")
    # else:
    #     # Make the dummy nmap/count files
    #     logger.info(f"Creating NMAP file {weight_file}")
    #     # phase_link.run_nmap( ... )

    # #########################
    # 3. phase linking/EVD step
    # #########################
    pl_path = scratch_dir / cfg["phase_linking"]["directory"]
    # For some reason fringe skips this one if the directory exists...
    # pl_path.mkdir(parents=True, exist_ok=True)
    # compressed_slc_file = pl_path / cfg["phase_linking"]["compressed_slc_file"]
    # if compressed_slc_file.exists():
    # logger.info(f"Skipping making existing EVD file {compressed_slc_file}")

    existing_files = list(pl_path.glob("*.slc.tif"))  # TODO: get ext from config
    if len(existing_files) > 0:
        logger.info(f"Skipping EVD step, {len(existing_files)} files already exist")
    else:
        logger.info(f"Running sequential EMI step in {pl_path}")
        # pl_path = sequential.run_evd_sequential( ... )

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

    # ###################################
    # 5. Stitch and Unwrap interferograms
    # ###################################
    # TODO: will this be a separate workflow?
    # Or will we loop through all bursts, then stitch, then unwrap all here?

    if not cfg["unwrap"]["run_unwrap"]:
        logger.info("Skipping unwrap step")
        return

    unwrap_path = scratch_dir / cfg["unwrap"]["directory"]
    unwrap_path.mkdir(parents=True, exist_ok=True)
    # unwrap.run( ... )

    # ####################
    # 6. Phase Corrections
    # ####################
    # TODO: Determine format for the tropospheric/ionospheric phase correction
