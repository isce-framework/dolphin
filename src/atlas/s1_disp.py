#!/usr/bin/env python
import argparse
from pathlib import Path

from atlas import ps, utils


def run(config_file: str):
    """Run the displacement workflow.

    Parameters
    ----------
    config_file : str
        YAML file containing the workflow options.
    """
    cfg = utils.load_yaml(config_file, workflow_name="s1_disp")

    # First make the amplitude dispersion file
    ps_path = Path(cfg["directories"]["ps"])
    ps_path.mkdir(parents=True, exist_ok=True)
    amp_disp_file = ps_path / cfg["ps"]["amp_disp_file"]
    amp_mean_file = ps_path / cfg["ps"]["amp_mean_file"]
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
    ps.create_ps(
        outfile=ps_output,
        amp_disp_file=amp_disp_file,
        amp_dispersion_threshold=threshold,
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
