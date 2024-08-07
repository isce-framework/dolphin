import contextlib
from pathlib import Path

import pytest

from dolphin.cli import main


@pytest.mark.parametrize("option", ["-h", "--help"])
def test_help(capsys, option):
    with contextlib.suppress(SystemExit):
        main([option])
    output = capsys.readouterr().out
    assert " dolphin [-h] [--version] {run,config,unwrap,timeseries,filter}" in output


def test_empty(capsys):
    with contextlib.suppress(SystemExit):
        main([])
    output = capsys.readouterr().out
    assert " dolphin [-h] [--version] {run,config,unwrap,timeseries,filter}" in output


@pytest.mark.parametrize("sub_cmd", ["run", "config", "filter", "unwrap", "timeseries"])
@pytest.mark.parametrize("option", ["-h", "--help"])
def test_subcommand_help(capsys, sub_cmd, option):
    with contextlib.suppress(SystemExit):
        main([sub_cmd, option])
    output = capsys.readouterr().out
    assert f"usage: dolphin {sub_cmd} [-h]" in output


def test_cli_config_basic(tmpdir, slc_file_list):
    with tmpdir.as_cwd(), contextlib.suppress(SystemExit):
        main(
            [
                "config",
                "--threads-per-worker",
                "2",
                "--strides",
                "6",
                "3",
                "--slc-files",
                *list(map(str, slc_file_list)),
            ]
        )
        assert Path("dolphin_config.yaml").exists()
