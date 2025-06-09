import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import pytest

from dolphin import __version__
from dolphin.cli import main

# Match the start of help output
HELP_LINE = "usage: dolphin"


def test_empty(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["dolphin"])
        f = StringIO()
        with redirect_stderr(f), pytest.raises(SystemExit):
            main()

        help_text = f.getvalue()
        assert "Required options" in help_text


def test_help(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["dolphin", "--help"])
        f = StringIO()
        with redirect_stdout(f), pytest.raises(SystemExit):
            main()

        # Get help text by capturing stdout when --help is passed.
        help_text = f.getvalue()
        assert HELP_LINE in help_text


def test_version(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["dolphin", "--version"])
        f = StringIO()
        with redirect_stdout(f), pytest.raises(SystemExit):
            main()

        version_text = f.getvalue()
        assert __version__ in version_text


@pytest.mark.parametrize("sub_cmd", ["run", "config", "filter", "unwrap"])
def test_subcommand_help(monkeypatch: pytest.MonkeyPatch, sub_cmd: str):
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["dolphin", sub_cmd, "--help"])

        f = StringIO()
        with redirect_stdout(f), pytest.raises(SystemExit):
            main()

        help_text = f.getvalue()
        assert f"usage: dolphin {sub_cmd}" in help_text


@pytest.mark.parametrize("strides_x_argname", ["--sx", "--output-options.strides.x"])
def test_cli_config_basic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    slc_file_list: list[Path],
    strides_x_argname: str,
):
    config_file = tmp_path / "dolphin_config.yaml"
    with monkeypatch.context() as m:
        m.setattr(
            sys,
            "argv",
            [
                "dolphin",
                "config",
                # Check both versions work
                strides_x_argname,
                "6",
                "--sy",
                "3",
                "--outfile",
                config_file.as_posix(),
                "--slc-files",
                *map(str, slc_file_list),
            ],
        )
        main()
        assert config_file.exists()
