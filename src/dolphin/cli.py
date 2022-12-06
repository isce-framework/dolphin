import argparse


def main():
    """Top-level command line interface to the workflows."""
    import dolphin.workflows.cli
    import dolphin.workflows.config

    parser = argparse.ArgumentParser(
        prog=__package__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparser = parser.add_subparsers(title="subcommands")

    # Adds the subcommand to the top-level parser
    _ = dolphin.workflows.cli.get_parser(subparser, "run")
    _ = dolphin.workflows.config.get_parser(subparser, "config")
    # return p.parse_args()
    return parser.parse_args()
