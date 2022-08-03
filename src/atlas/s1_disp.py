#!/usr/bin/env python
import argparse
from pathlib import Path

from atlas import ps, utils


def get_cli_args():
    """Set up the command line interface."""
    parser = argparse.ArgumentParser(
        description="Run a displacement worflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config-file",
        help="Name of YAML configuration file describing workflow options.",
    )
    args = parser.parse_args()
    return args


def main(config_file: str):
    """Run the displacement workflow.

    Parameters
    ----------
    config_file : str
        YAML file containing the workflow options.
    """
    cfg = utils.load_yaml(config_file, workflow_name="s1_disp")

    # First make the amplitude dispersion file
    ps_path = Path(cfg["directories"]["ps"])
    amp_disp_output = ps_path / cfg["ps"]["amp_dispersion_file"]
    amp_mean_file = ps_path / cfg["ps"]["amp_mean_file"]
    # *, input_vrt_file, output_file, amp_mean_file, reference_band, processing_opts
    ps.create_amp_dispersion(
        input_vrt_file=cfg["input_vrt_file"],
        output_file=amp_disp_output,
        amp_mean_file=amp_mean_file,
        reference_band=cfg["ps"]["reference_band"],
        processing_opts=cfg["processing"],
    )

    # PS creation
    ps_output = ps_path / cfg["ps_file"]
    threshold = cfg["ps"]["amp_dispersion_threshold"]
    ps.create_ps(outfile=ps_output, amp_dispersion_threshold=threshold)


if __name__ == "__main__":
    args = get_cli_args()
    main(args.config_file)
