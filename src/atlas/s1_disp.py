#!/usr/bin/env python
import argparse
from pathlib import Path

from atlas import ps, utils, wrapped
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
    cfg = utils.load_yaml(config_file, workflow_name="s1_disp")

    # First make the amplitude dispersion file
    ps_path = Path(cfg["directories"]["ps"]).absolute()
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

    # PS creation
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

    # nmap creation
    nmap_path = Path(cfg["directories"]["nmap"]).absolute()
    nmap_path.mkdir(parents=True, exist_ok=True)
    weight_file = nmap_path / cfg["weight_file"]
    nmap_count_file = nmap_path / cfg["nmap_count_file"]
    threshold = cfg["ps"]["amp_dispersion_threshold"]
    if weight_file.exists():
        logger.info(f"Skipping making existing NMAP file {weight_file}")
    else:
        logger.info(f"Creating NMAP file {weight_file}")
        wrapped.run_nmap(
            input_vrt_file=cfg["input_vrt_file"],
            weight_file=str(weight_file),
            nmap_count_file=str(nmap_count_file),
            window=cfg["window"],
            pvalue=cfg["nmap"]["pvalue"],
            stat_method=cfg["nmap"]["stat_method"],
            no_gpu=cfg["nmap"]["no_gpu"],
            processing_opts=cfg["processing"],
            mask_file=cfg["mask_file"],
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
