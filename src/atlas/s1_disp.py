#!/usr/bin/env python
import argparse
from pathlib import Path

from atlas import combine_ps_ds, config, phase_linking, ps
from atlas.log import get_log, log_runtime

logger = get_log()


@log_runtime
def run(config_file: str):
    """Run the displacement workflow.

    Parameters
    ----------
    config_file : str
        YAML file containing the workflow options.
    """
    cfg = config.load_workflow_yaml(config_file, workflow_name="s1_disp")
    filled_cfg_path = Path(config_file).with_suffix(".filled.yaml")
    config.save_yaml(filled_cfg_path, config.add_atlas_section(cfg))

    # 1. First make the amplitude dispersion file
    ps_path = Path(cfg["ps"]["directory"]).absolute()
    ps_path.mkdir(parents=True, exist_ok=True)
    amp_disp_file = ps_path / cfg["ps"]["amp_disp_file"]
    amp_mean_file = ps_path / cfg["ps"]["amp_mean_file"]
    if amp_disp_file.exists():
        logger.info(f"Skipping existing amplitude dispersion file {amp_disp_file}")
    else:
        logger.info(f"Making amplitude dispersion file {amp_disp_file}")
        ps.create_amp_dispersion(
            input_vrt_file=cfg["input_vrt_file"],
            output_file=str(amp_disp_file),
            amp_mean_file=str(amp_mean_file),
            reference_band=cfg["ps"]["normalizing_reference_band"],
            processing_opts=cfg["processing"],
        )

    # 2. PS selection
    ps_output = ps_path / cfg["ps_file"]
    threshold = cfg["ps"]["amp_dispersion_threshold"]
    if ps_output.exists():
        logger.info(f"Skipping making existing PS file {ps_output}")
    else:
        logger.info(f"Creating persistent scatterer file {ps_output}")
        ps.create_ps(
            outfile=ps_output,
            amp_disp_file=amp_disp_file,
            amp_dispersion_threshold=threshold,
        )

    # 3. nmap: find SHP neighborhoods
    nmap_path = Path(cfg["nmap"]["directory"]).absolute()
    nmap_path.mkdir(parents=True, exist_ok=True)
    weight_file = nmap_path / cfg["weight_file"]
    nmap_count_file = nmap_path / cfg["nmap_count_file"]
    threshold = cfg["ps"]["amp_dispersion_threshold"]
    if weight_file.exists():
        logger.info(f"Skipping making existing NMAP file {weight_file}")
    else:
        logger.info(f"Creating NMAP file {weight_file}")
        phase_linking.run_nmap(
            input_vrt_file=cfg["input_vrt_file"],
            mask_file=cfg["mask_file"],
            weight_file=str(weight_file),
            nmap_count_file=str(nmap_count_file),
            window=cfg["window"],
            nmap_opts=cfg["nmap"],
            processing_opts=cfg["processing"],
        )

    # 4. phase linking/EVD step
    pl_path = Path(cfg["phase_linking"]["directory"]).absolute()
    # For some reason fringe skips this one if the directory exists...
    # pl_path.mkdir(parents=True, exist_ok=True)
    compressed_slc_file = pl_path / cfg["phase_linking"]["compressed_slc_file"]
    if compressed_slc_file.exists():
        logger.info(f"Skipping making existing EVD file {compressed_slc_file}")
    else:
        logger.info(f"Making EVD file {compressed_slc_file}")
        phase_linking.run_evd(
            input_vrt_file=cfg["input_vrt_file"],
            weight_file=str(weight_file),
            compressed_slc_filename=str(compressed_slc_file.name),
            output_folder=str(pl_path),
            window=cfg["window"],
            pl_opts=cfg["phase_linking"],
            processing_opts=cfg["processing"],
        )

    # 5. Combinine PS and DS phases and forming interferograms
    ps_ds_path = Path(cfg["combine_ps_ds"]["directory"]).absolute()
    ps_ds_path.mkdir(parents=True, exist_ok=True)
    # Temp coh file made in last step:
    temp_coh_file = pl_path / cfg["phase_linking"]["temp_coh_file"]
    # Final coh file to be created here
    temp_coh_ps_ds_file = ps_ds_path / cfg["combine_ps_ds"]["temp_coh_file"]
    if temp_coh_ps_ds_file.exists():
        logger.info(f"Skipping combine_ps_ds step, {temp_coh_file} exists")
    else:
        logger.info(f"Running combine ps/ds step into {ps_ds_path}")
        combine_ps_ds.run_combine(
            input_vrt_file=cfg["input_vrt_file"],
            ps_file=ps_output,
            pl_directory=pl_path,
            temp_coh_file=temp_coh_file,
            temp_coh_ps_ds_file=temp_coh_ps_ds_file,
            output_folder=ps_ds_path,
            ps_temp_coh=cfg["combine_ps_ds"]["ps_temp_coh"],
        )


def get_cli_args():
    """Set up the command line interface."""
    parser = argparse.ArgumentParser(
        description="Run a displacement worflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config_file",
        help="Name of YAML configuration file describing workflow options.",
    )
    args = parser.parse_args()
    return args


def main():
    """Get the command line arguments and run the workflow."""
    args = get_cli_args()
    run(args.config_file)


if __name__ == "__main__":
    main()
