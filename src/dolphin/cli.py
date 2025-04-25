# import argparse
# import sys

# import dolphin._cli_filter
# import dolphin._cli_timeseries
# import dolphin._cli_unwrap


def main() -> None:
    """Top-level command line interface to the workflows."""
    import tyro

    from dolphin.filtering import filter_rasters
    from dolphin.unwrap import run as run_unwrap
    from dolphin.workflows._cli_config import ConfigCli
    from dolphin.workflows._cli_run import run as run_cli

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
