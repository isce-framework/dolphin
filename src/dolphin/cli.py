import argparse


def main(args=None):
    """Top-level command line interface to the workflows."""
    import dolphin.workflows._config_cli
    import dolphin.workflows._run_cli

    parser = argparse.ArgumentParser(
        prog=__package__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparser = parser.add_subparsers(title="subcommands")

    # Adds the subcommand to the top-level parser
    _ = dolphin.workflows._run_cli.get_parser(subparser, "run")
    _ = dolphin.workflows._config_cli.get_parser(subparser, "config")
    # return p.parse_args()
    return parser.parse_args(args=args)
