import sys
from pathlib import Path

import pytest

from dolphin.cli import main

# Match the start of help output
HELP_LINE = "usage: dolphin"


def test_help(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["dolphin", "--help"])
        with pytest.raises(SystemExit):
            main()
        output = capsys.readouterr().out
        assert HELP_LINE in output


def test_empty(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["dolphin"])
        with pytest.raises(SystemExit):
            main()
        output = capsys.readouterr().out
        assert HELP_LINE in output


@pytest.mark.parametrize("sub_cmd", ["run", "config", "filter", "unwrap"])
def test_subcommand_help(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    sub_cmd: str,
):
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["dolphin", sub_cmd, "--help"])
        with pytest.raises(SystemExit):
            main()
        output = capsys.readouterr().out
        assert f"usage: dolphin {sub_cmd}" in output


def test_cli_config_basic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, slc_file_list: list[Path]
):
    config_file = tmp_path / "dolphin_config.yaml"
    with monkeypatch.context() as m:
        m.setattr(
            sys,
            "argv",
            [
                "dolphin",
                "config",
                "--strides",
                "6",
                "3",
                "--slc-files",
                *map(str, slc_file_list),
            ],
        )
        cwd = tmp_path.as_posix()
        m.chdir(cwd)
        with pytest.raises(SystemExit):
            main()
        assert config_file.exists()
