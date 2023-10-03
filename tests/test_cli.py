from pathlib import Path

import pytest

from dolphin.cli import main


@pytest.mark.parametrize("option", ("-h", "--help"))
def test_help(capsys, option):
    try:
        main([option])
    except SystemExit:
        pass
    output = capsys.readouterr().out
    assert " dolphin [-h] [--version] {run,config,unwrap}" in output


def test_empty(capsys):
    try:
        main([])
    except SystemExit:
        pass
    output = capsys.readouterr().out
    assert " dolphin [-h] [--version] {run,config,unwrap}" in output


@pytest.mark.parametrize("sub_cmd", ("run", "config", "unwrap"))
@pytest.mark.parametrize("option", ("-h", "--help"))
def test_subcommand_help(capsys, sub_cmd, option):
    try:
        main([sub_cmd, option])
    except SystemExit:
        pass
    output = capsys.readouterr().out
    assert f"usage: dolphin {sub_cmd} [-h]" in output


def test_cli_config_basic(tmpdir, slc_file_list):
    with tmpdir.as_cwd():
        try:
            main(
                [
                    "config",
                    "--n-workers",
                    "1",
                    "--threads-per-workers",
                    "1",
                    "--slc-files",
                    *list(map(str, slc_file_list)),
                ]
            )
        except SystemExit:
            pass
        assert Path("dolphin_config.yaml").exists()
