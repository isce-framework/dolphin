from dolphin._show_versions import _get_deps_info, _get_sys_info, show_versions


def test_get_sys_info():
    sys_info = _get_sys_info()

    assert "python" in sys_info
    assert "executable" in sys_info
    assert "machine" in sys_info


def test_get_deps_info():
    deps_info = _get_deps_info()

    assert "osgeo.gdal" in deps_info
    assert "numpy" in deps_info
    assert "pydantic" in deps_info
    assert "h5py" in deps_info


def test_show_versions(capsys):
    show_versions()
    out, err = capsys.readouterr()

    assert "python" in out
    assert "dolphin" in out
