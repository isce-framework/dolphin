#!/usr/bin/env python
from glob import glob
from os import fspath
from pathlib import Path

from dolphin import combine_ps_ds, phase_link, ps, sequential, unwrap, utils, vrt
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
    gpu_enabled = full_cfg["worker"]["gpu_enabled"]

    input_file_list = full_cfg["input_file_group"]["cslc_file_list"]
    input_file_path = full_cfg["input_file_group"]["cslc_file_path"]
    if not input_file_list:
        if not input_file_path:
            raise ValueError("Must specify either cslc_file_list or cslc_file_path")

        input_file_path = Path(input_file_path).absolute()
        ext = full_cfg["input_file_group"]["cslc_file_ext"]
        # TODO : somehow accommodate inputs other than ENVI
        input_file_list = sorted(glob(fspath(input_file_path / f"*{ext}")))

    # dem_file = full_cfg["dynamic_ancillary_file_group"]["dem_file"]

    # 0. Make a VRT pointing to the input SLC files
    slc_vrt_file = scratch_dir / "slc_stack.vrt"
    vrt_stack = vrt.VRTStack(input_file_list, outfile=slc_vrt_file)

    if slc_vrt_file.exists():
        logger.info(f"Skipping creating VRT to SLC stack {slc_vrt_file}")
    else:
        logger.info(f"Creating VRT to SLC stack file {slc_vrt_file}")
        vrt_stack.write()

    # 1. First make the amplitude dispersion file
    ps_path = scratch_dir / cfg["ps"]["directory"]
    ps_path.mkdir(parents=True, exist_ok=True)
    amp_disp_file = ps_path / full_cfg["dynamic_ancillary_file_group"]["amp_disp_file"]
    amp_mean_file = ps_path / full_cfg["dynamic_ancillary_file_group"]["amp_mean_file"]

    if amp_disp_file.exists():
        logger.info(f"Skipping existing amplitude dispersion file {amp_disp_file}")
    else:
        logger.info(f"Making amplitude dispersion file {amp_disp_file}")
        ps.create_amp_dispersion(
            slc_vrt_file=slc_vrt_file,
            output_file=amp_disp_file,
            amp_mean_file=amp_mean_file,
            reference_band=cfg["ps"]["normalizing_reference_band"],
            lines_per_block=cfg["lines_per_block"],
            ram=cfg["ram"],
        )

    # 2. PS selection
    ps_output = ps_path / cfg["ps_file"]
    threshold = cfg["ps"]["amp_dispersion_threshold"]
    if ps_output.exists():
        logger.info(f"Skipping making existing PS file {ps_output}")
    else:
        logger.info(f"Creating persistent scatterer file {ps_output}")
        ps.create_ps(
            output_file=ps_output,
            amp_disp_file=amp_disp_file,
            amp_dispersion_threshold=threshold,
        )

    # 3. nmap: find SHP neighborhoods
    nmap_path = scratch_dir / cfg["nmap"]["directory"]
    nmap_path.mkdir(parents=True, exist_ok=True)
    weight_file = nmap_path / cfg["weight_file"]
    nmap_count_file = nmap_path / cfg["nmap_count_file"]
    threshold = cfg["ps"]["amp_dispersion_threshold"]
    if not cfg["nmap"]["run_nmap"] or weight_file.exists():
        logger.info(f"Skipping making existing NMAP file {weight_file}")
    else:
        # Make the dummy nmap/count files
        logger.info(f"Creating NMAP file {weight_file}")
        phase_link.run_nmap(
            slc_vrt_file=slc_vrt_file,
            weight_file=weight_file,
            count_file=nmap_count_file,
            window=cfg["window"],
            nmap_opts=cfg["nmap"],
            lines_per_block=cfg["lines_per_block"],
            ram=cfg["ram"],
            no_gpu=not gpu_enabled,
            # mask_file=cfg["mask_file"], # TODO : add mask file if needed
        )

    # 4. phase linking/EVD step
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
        pl_path = sequential.run_evd_sequential(
            slc_vrt_file=slc_vrt_file,
            # weight_file: Pathlike,
            output_folder=pl_path,
            window=cfg["window"],
            # ministack_size=10,
            ministack_size=cfg["phase_linking"]["ministack_size"],
            max_bytes=1e6 * cfg["ram"],
            # mask_file: Pathlike = None,
            # ps_mask_file=ps_output,
            # beta=0.1,
            # beta=cgf["phase_linking"]["beta"],
        )

    # 5. Combine PS and DS phases and forming interferograms
    ps_ds_path = scratch_dir / cfg["combine_ps_ds"]["directory"]
    ps_ds_path.mkdir(parents=True, exist_ok=True)

    # Temp coh file made in last step:
    # temp_coh_file = pl_path / cfg["phase_linking"]["temp_coh_file"]
    temp_coh_file = pl_path / "tcorr_average.tif"
    # Final coh file to be created here
    # temp_coh_ps_ds_file = ps_ds_path / cfg["combine_ps_ds"]["temp_coh_file"]
    # if temp_coh_ps_ds_file.exists():
    existing_ifgs = list(ps_ds_path.glob("*.int*"))
    if len(existing_ifgs) > 0:
        logger.info(f"Skipping combine_ps_ds step, {len(existing_ifgs)} exists")
    else:
        logger.info(f"Running combine ps/ds step into {ps_ds_path}")
        # # Old way based on fringe to form ifgs and insert PS phases
        # combine_ps_ds.run_combine(
        #     slc_vrt_file=slc_vrt_file,
        #     ps_file=ps_output,
        #     pl_directory=pl_path,
        #     temp_coh_file=temp_coh_file,
        #     temp_coh_ps_ds_file=temp_coh_ps_ds_file,
        #     output_folder=ps_ds_path,
        #     ps_temp_coh=cfg["combine_ps_ds"]["ps_temp_coh"],
        #     ifg_network_options=cfg["combine_ps_ds"]["ifg_network_options"],
        # )

        # The python MLE function handles the temp coh, and the PS phase insertion
        combine_ps_ds.form_ifgs(
            slc_vrt_file=slc_vrt_file,
            pl_directory=pl_path,
            output_folder=ps_ds_path,
            ifg_network_options=cfg["combine_ps_ds"]["ifg_network_options"],
        )

    # 6. Unwrap interferograms
    # TODO: either combine, or figure out if we need multiple masks
    # TODO: Do we create a new mask file here based on temporal coherence?
    mask_files = full_cfg["dynamic_ancillary_file_group"]["mask_files"]
    if len(mask_files) >= 1:
        combined_mask_file = utils.combine_mask_files(mask_files, scratch_dir)
    else:
        combined_mask_file = None

    if not cfg["unwrap"]["run_unwrap"]:
        logger.info("Skipping unwrap step")
        return

    unwrap_path = scratch_dir / cfg["unwrap"]["directory"]
    unwrap_path.mkdir(parents=True, exist_ok=True)
    unwrap.run(
        ifg_path=ps_ds_path,
        output_path=unwrap_path,
        # cor_file=temp_coh_ps_ds_file,
        cor_file=temp_coh_file,
        mask_file=combined_mask_file,
    )
