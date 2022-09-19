#!/usr/bin/env python
import argparse
from pathlib import Path

from dolphin import config
from dolphin.log import get_log, log_runtime

logger = get_log()


@log_runtime
def run(config_file: str, name: str = "stack"):
    """Run the displacement workflow.

    Parameters
    ----------
    config_file : str
        YAML file containing the workflow options.
    name : str, choices = ["single", "stack"]
        Name of the workflow to run.
    """
    cfg = config.load_workflow_yaml(config_file, workflow_name=f"s1_disp_{name}")
    cfg_path = Path(config_file)
    filled_cfg_path = cfg_path.with_name(cfg_path.stem + "_filled" + cfg_path.suffix)
    config.save_yaml(filled_cfg_path, config.add_dolphin_section(cfg))
    if name == "single":
        from dolphin import s1_disp_single

        s1_disp_single.run(cfg["runconfig"]["groups"])
    elif name == "stack":
        from dolphin import s1_disp_stack

        s1_disp_stack.run(cfg["runconfig"]["groups"])


def get_cli_args():
    """Set up the command line interface."""
    parser = argparse.ArgumentParser(
        description="Run a displacement workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config_file",
        help="Name of YAML configuration file describing workflow options.",
    )
    parser.add_argument(
        "-n",
        "--name",
        default="stack",
        choices=["single", "stack"],
        help="Name workflow to run.",
    )
    args = parser.parse_args()
    return args


def main():
    """Get the command line arguments and run the workflow."""
    args = get_cli_args()
    run(args.config_file, name=args.name)


if __name__ == "__main__":
    main()
