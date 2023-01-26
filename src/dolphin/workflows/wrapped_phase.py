#!/usr/bin/env python
from typing import List

from dolphin import ps, sequential, vrt
from dolphin._log import get_log, log_runtime
from dolphin.interferogram import Network, VRTInterferogram

from .config import Workflow


@log_runtime
def run(cfg: Workflow, debug: bool = False) -> List[VRTInterferogram]:
    """Run the displacement workflow on a stack of SLCs.

    Parameters
    ----------
    cfg : Workflow
        [`Workflow`][dolphin.workflows.config.Workflow] object with workflow parameters
    debug : bool, optional
        Enable debug logging, by default False.
    """
    logger = get_log(debug=debug)
    scratch_dir = cfg.outputs.scratch_directory

    input_file_list = cfg.inputs.cslc_file_list
    if not input_file_list:
        raise ValueError("No input files found")

    # #############################################
    # 1. Make a VRT pointing to the input SLC files
    # #############################################
    subdataset = cfg.inputs.subdataset
    vrt_path = scratch_dir / "slc_stack.vrt"
    if vrt_path.exists():
        vrt_stack = vrt.VRTStack.from_vrt_file(vrt_path)
    else:
        vrt_stack = vrt.VRTStack(
            input_file_list,
            subdataset=subdataset,
            outfile=scratch_dir / "slc_stack.vrt",
        )

    # ###############
    # 2. PS selection
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
    # 3. phase linking/EVD step
    # #########################
    pl_path = cfg.phase_linking.directory

    # TODO: get intermediate ext from config
    existing_files = list(pl_path.glob("*.tif"))
    if len(existing_files) > 0:
        logger.info(f"Skipping EVD step, {len(existing_files)} files already exist")
    else:
        logger.info(f"Running sequential EMI step in {pl_path}")
        sequential.run_evd_sequential(
            slc_vrt_file=vrt_stack.outfile,
            output_folder=pl_path,
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
    # 4. Form interferograms from estimated wrapped phase
    # ###################################################
    ifg_dir = cfg.interferogram_network.directory
    existing_ifgs = list(ifg_dir.glob("*.int.vrt"))
    if len(existing_ifgs) > 0:
        logger.info(f"Skipping interferogram step, {len(existing_ifgs)} exists")
    else:
        # TODO: intermediate extension should be in config
        phase_linked_slcs = sorted(pl_path.glob("20*.tif"))
        compressed_slcs = sorted(pl_path.glob("compressed_*.tif"))
        logger.info(
            f"Creating virtual interferograms from {len(phase_linked_slcs)} files and"
            f" {len(compressed_slcs)} compressed files"
        )
        if compressed_slcs:
            slc_list = [compressed_slcs[0]] + phase_linked_slcs
        else:
            slc_list = phase_linked_slcs
        network = Network(
            slc_list=slc_list,
            reference_idx=cfg.interferogram_network.reference_idx,
            max_bandwidth=cfg.interferogram_network.max_bandwidth,
            max_temporal_baseline=cfg.interferogram_network.max_temporal_baseline,
            outdir=ifg_dir,
        )
        if len(network) == 0:
            raise ValueError("No interferograms were created")

    return network.ifg_list
