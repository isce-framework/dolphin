#!/usr/bin/env python
"""List packages installed with conda & yum in CSV format.

Optionally list packages installed inside a container image instead.
"""
import argparse
import csv
import functools
import json
import os
import subprocess
import sys
from dataclasses import astuple, dataclass
from itertools import dropwhile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TextIO

DESCRIPTION = __doc__


@dataclass
class Package:
    """An installed package."""

    name: str
    version: str
    package_manager: str
    channel: str


class CommandNotFoundError(Exception):
    """Raised when a required Unix shell command was not found."""

    pass


class YumListIsAnnoyingError(Exception):
    """Raised when 'yum list' does something annoying."""

    pass


def check_command(cmd: str) -> bool:
    """Check if a Unix shell command is available."""
    # Check if `cmd` is available on the PATH or as a builtin command.
    args = ["command", "-v", cmd]
    try:
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        # The exit status is 0 if the command was found, and 1 if not.
        # Any other status is unexpected, so re-raise the error.
        if exc.returncode == 1:
            return False
        else:
            raise
    else:
        return True


@functools.lru_cache(maxsize=None)
def conda_command() -> str:
    """Get the name of the conda executable."""
    # A list of possible "conda-like" commands to try.
    possible_commands = [
        "conda",
        "mamba",
        "micromamba",
    ]

    # Try each possible command. Return the first one that's available.
    for cmd in possible_commands:
        if check_command(cmd):
            return cmd
    else:
        errmsg = f"conda command not found -- tried {possible_commands}"
        raise CommandNotFoundError(errmsg)


def list_conda_packages(env: str = "base") -> List[Package]:
    """List conda packages installed in the specified environment."""
    # Get the list of conda packages as a JSON-formatted string.
    args = [conda_command(), "list", "--name", env, "--json"]
    res = subprocess.run(args, capture_output=True, check=True, text=True)

    # Parse the JSON string.
    package_list = json.loads(res.stdout)

    def as_package(package_info: dict) -> Package:
        name = package_info["name"]
        version = package_info["version"]
        channel = package_info["channel"]
        return Package(name, version, "conda", channel)

    # Convert to `Package` objects.
    packages = map(as_package, package_list)

    # Sort alphabetically by name.
    return sorted(packages, key=lambda package: package.name)


def list_yum_packages() -> List[Package]:
    """List system packages installed with yum."""
    # Check if 'yum' is available.
    if not check_command("yum"):
        errmsg = "'yum' command not found"
        raise CommandNotFoundError(errmsg)

    # Get the list of yum packages.
    args = ["yum", "list", "installed"]
    res = subprocess.run(args, capture_output=True, check=True, text=True)

    # Split the output into a sequence of lines.
    # Skip past the first few lines until we get to the actual list of packages.
    lines = res.stdout.splitlines()
    package_list = dropwhile(lambda line: line != "Installed Packages", lines)
    try:
        next(package_list)  # type: ignore[call-overload]
    except StopIteration:
        cmd = subprocess.list2cmdline(args)
        errmsg = f"unexpected output from {cmd!r}"
        raise RuntimeError(errmsg)

    def as_package(line: str) -> Package:
        try:
            name, version, channel = line.split()
        except ValueError as exc:
            raise YumListIsAnnoyingError from exc

        # Package names are in '{name}.{arch}' format.
        # Strip arch info from package name.
        name = ".".join(name.split(".")[:-1])

        # If the repo cannot be determined, yum prints "installed" by default.
        channel = channel.lstrip("@") if (channel != "installed") else ""

        return Package(name, version, "yum", channel)

    def parse_lines(lines: Iterator[str]) -> Iterator[Package]:
        for line in lines:
            # Sometimes package info in the output is inexplicably broken up
            # across two lines...
            try:
                package = as_package(line)
            except YumListIsAnnoyingError:
                line += next(lines)
                package = as_package(line)

            yield package

    # Parse each line as a `Package` object.
    packages = parse_lines(package_list)

    # Sort alphabetically by name.
    return sorted(packages, key=lambda package: package.name)


def list_packages() -> List[Package]:
    """List installed packages."""
    return list_conda_packages() + list_yum_packages()


def write_package_csv(csvfile: TextIO = sys.stdout) -> None:
    """Write a list of installed packages to file in CSV format."""
    packages = list_packages()
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Name", "Version", "Package Manager", "Channel"])
    csvwriter.writerows(map(astuple, packages))


def setup_parser() -> argparse.ArgumentParser:
    """Set up the command-line argument parser."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    docker_group = parser.add_argument_group("docker options")
    docker_group.add_argument(
        "image", nargs="?", type=str, help="a docker image tag or ID"
    )
    docker_group.add_argument(
        "-e",
        "--entrypoint",
        type=str,
        help="overwrite the default ENTRYPOINT of the image",
    )

    return parser


def parse_args(args: Optional[List[str]] = None) -> Dict[str, Any]:
    """Parse command-line arguments."""
    parser = setup_parser()
    params = parser.parse_args(args)

    if (params.image is None) and (params.entrypoint is not None):
        parser.error(
            "entrypoint should not be specified unless an image tag or ID is provided"
        )

    return vars(params)


def run_in_container(image: str, entrypoint: Optional[str] = None) -> None:
    """Run this module inside a container derived from the specified image."""
    # Check if 'docker' is available.
    if not check_command("docker"):
        errmsg = "'docker' command not found"
        raise CommandNotFoundError(errmsg)

    this_file = Path(__file__).resolve()
    workdir = Path("/tmp")

    args = [
        "docker",
        "run",
        "--rm",
        f"--user={os.getuid()}:{os.getgid()}",
        f"--volume={this_file}:{workdir / 'list_packages.py'}:ro",
        f"--workdir={workdir}",
    ]

    if entrypoint is not None:
        args.append(f"--entrypoint={entrypoint}")

    args += [image, "python", "-m", "list_packages"]

    # Mount this script inside the container and run it without arguments.
    subprocess.run(args, check=True)


def main(args: Optional[List[str]] = None) -> None:  # noqa: D103
    kwargs = parse_args(args)
    if kwargs.get("image") is None:
        write_package_csv()
    else:
        run_in_container(**kwargs)


if __name__ == "__main__":
    main()
