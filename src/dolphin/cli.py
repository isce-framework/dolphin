def main() -> None:
    """Top-level command line interface to the workflows."""
    import tyro

    from dolphin.filtering import filter_rasters
    from dolphin.unwrap import run as run_unwrap
    from dolphin.workflows._cli_config import ConfigCli

    tyro.extras.subcommand_cli_from_dict(
        {
            "run": run_cli,
            "config": ConfigCli,
            "unwrap": run_unwrap,
            # "timeseries": run_timeseries,
            # "velocity": create_velocity,
            "filter": filter_rasters,
        },
        prog=__package__,
    )

    # TODO: Check how to slot this into tyro's argparse
    # parser.add_argument("--version", action="version", version=__version__)


def run_cli(
    config_file: str,
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
