import os


def main() -> int:
    """Top-level command line interface to the workflows."""
    import sys

    from dolphin import __version__

    # TODO: Is there any way to slot this into tyro's argparse
    # Only found this hacky way
    # https://github.com/brentyi/tyro/issues/132#issuecomment-1978319762
    if len(sys.argv) > 1 and sys.argv[1] == "--version":
        print(__version__)
        raise SystemExit(os.EX_OK)

    import tyro

    from dolphin.filtering import filter_rasters
    from dolphin.timeseries import run as run_timeseries
    from dolphin.unwrap import run as run_unwrap
    from dolphin.workflows._cli_config import ConfigCli

    tyro.extras.subcommand_cli_from_dict(
        {
            "run": run_cli,
            "config": ConfigCli,
            "unwrap": run_unwrap,
            "timeseries": run_timeseries,
            "filter": filter_rasters,
        },
        prog=__package__,
    )

    return os.EX_OK


def run_cli(
    config_file: str,
    /,
    debug: bool = False,
) -> None:
    """Run the displacement workflow.

    Parameters
    ----------
    config_file : str
        YAML file containing the workflow options.
    debug : bool, optional
        Enable debug logging, by default False.

    """
    from .workflows import displacement
    from .workflows.config import DisplacementWorkflow

    cfg = DisplacementWorkflow.from_yaml(config_file)
    displacement.run(cfg, debug=debug)
