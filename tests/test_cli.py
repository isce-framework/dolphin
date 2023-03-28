import pytest

from dolphin.cli import main


@pytest.mark.parametrize("option", ("-h", "--help"))
def test_help(capsys, option):
    try:
        main([option])
    except SystemExit:
        pass
    output = capsys.readouterr().out
    assert " dolphin [-h] [--version] {run,config}" in output


@pytest.mark.parametrize("sub_cmd", ("run", "config"))
@pytest.mark.parametrize("option", ("-h", "--help"))
def test_subcommand_help(capsys, sub_cmd, option):
    try:
        main([sub_cmd, option])
    except SystemExit:
        pass
    output = capsys.readouterr().out
    assert f"usage: dolphin {sub_cmd} [-h]" in output
