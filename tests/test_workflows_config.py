import shutil
from datetime import date, datetime
from pathlib import Path

import pydantic
import pytest

from dolphin.workflows import InterferogramNetworkType, config


# Testing what the defaults look like for each class
def test_half_window_defaults():
    hw = config.HalfWindow()
    assert hw.x == 11
    assert hw.y == 5
    assert hw.dict() == dict(x=11, y=5)


def test_half_window_to_looks():
    hw = config.HalfWindow()
    row_looks, col_looks = (11, 23)
    assert (row_looks, col_looks) == hw.to_looks()
    assert hw == config.HalfWindow.from_looks(row_looks, col_looks)

    # Test half window
    hw = config.HalfWindow(x=2, y=4)
    row_looks, col_looks = (9, 5)
    assert (row_looks, col_looks) == hw.to_looks()
    assert hw == config.HalfWindow.from_looks(row_looks, col_looks)


def test_ps_options_defaults(tmpdir):
    # Change directory so the creation of the default files doesn't fail
    with tmpdir.as_cwd():
        pso = config.PsOptions()
        assert pso.amp_dispersion_threshold == 0.42
        assert pso.directory == Path("PS")
        assert pso.output_file == Path("PS/ps_pixels.tif")
        assert pso.amp_dispersion_file == Path("PS/amp_dispersion.tif")
        assert pso.amp_mean_file == Path("PS/amp_mean.tif")


def test_phase_linking_options_defaults(tmpdir):
    with tmpdir.as_cwd():
        opts = config.PhaseLinkingOptions()
        assert opts.ministack_size == 15
        assert opts.directory == Path("linked_phase")
        assert opts.half_window == config.HalfWindow()


def test_phase_linking_options_bad_size(tmpdir):
    with pytest.raises(pydantic.ValidationError):
        config.PhaseLinkingOptions(ministack_size=0)
        config.PhaseLinkingOptions(ministack_size=-1)


def test_interferogram_network_defaults(tmpdir):
    with tmpdir.as_cwd():
        opts = config.InterferogramNetwork()
        assert opts.reference_idx == 0
        assert opts.max_bandwidth is None
        assert opts.max_temporal_baseline is None
        assert opts.network_type == InterferogramNetworkType.SINGLE_REFERENCE


def test_unwrap_options_defaults(tmpdir):
    with tmpdir.as_cwd():
        opts = config.UnwrapOptions()
        assert opts.unwrap_method == config.UnwrapMethod.SNAPHU
        assert opts.tiles == [1, 1]
        assert opts.init_method == "mcf"
        assert opts.directory == Path("unwrap")


def test_outputs_defaults(tmpdir):
    with tmpdir.as_cwd():
        opts = config.Outputs()
        assert opts.output_directory == Path("output").absolute()
        assert opts.output_format == config.OutputFormat.NETCDF
        assert opts.scratch_directory == Path("scratch").absolute()
        assert opts.output_resolution is None
        assert opts.strides == {"x": 1, "y": 1}
        assert opts.hdf5_creation_options == dict(
            chunks=True,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )


def test_worker_settings_defaults():
    ws = config.WorkerSettings()
    assert ws.gpu_enabled is True
    assert ws.gpu_id == 0
    assert ws.n_workers == 16
    assert ws.max_ram_gb == 1.0


def test_worker_env_defaults(monkeypatch):
    # Change environment with monkeypatch
    # https://docs.pytest.org/en/latest/how-to/monkeypatch.html
    monkeypatch.setenv("dolphin_gpu_enabled", "False")
    ws = config.WorkerSettings()
    assert ws.gpu_enabled is False
    monkeypatch.delenv("dolphin_gpu_enabled")

    # "gpu" doesn't need the dolphin_ prefix
    monkeypatch.setenv("gpu", "False")
    ws = config.WorkerSettings()
    assert ws.gpu_enabled is False

    # Case shouldn't matter (since i'm not specifying that it does)
    monkeypatch.setenv("DOLPHIN_Gpu_Id", "1")
    ws = config.WorkerSettings()
    assert ws.gpu_id == 1

    # Check that we need the dolphin_ prefix
    monkeypatch.setenv("N_WORKERS", "8")
    ws = config.WorkerSettings()
    assert ws.n_workers == 16  # should still be old default

    monkeypatch.setenv("DOLPHIN_N_WORKERS", "8")
    ws = config.WorkerSettings()
    assert ws.n_workers == 8

    monkeypatch.setenv("DOLPHIN_MAX_RAM_GB", "4.5")
    ws = config.WorkerSettings()
    assert ws.max_ram_gb == 4.5


@pytest.fixture()
def dir_with_1_slc(tmp_path, slc_file_list_nc):
    p = tmp_path / "slc"
    p.mkdir()

    fname = "slc_20220101.nc"
    shutil.copy(slc_file_list_nc[0], p / fname)
    with open(p / "slclist.txt", "w") as f:
        f.write(fname + "\n")
    return p


@pytest.fixture()
def dir_with_2_slcs(tmp_path, slc_file_list_nc):
    p = tmp_path / "slc"
    p.mkdir()
    shutil.copy(slc_file_list_nc[0], p / "slc_20220101.nc")
    shutil.copy(slc_file_list_nc[1], p / "slc_20220102.nc")
    with open(p / "slclist.txt", "w") as f:
        f.write("slc_20220101.nc\n")
        f.write("slc_20220102.nc\n")
    return p


def test_inputs_defaults(dir_with_1_slc):
    # make a dummy file
    opts = config.Inputs(cslc_file_list=dir_with_1_slc / "slclist.txt")

    assert opts.cslc_file_list[0].parent == dir_with_1_slc.absolute()
    assert opts.cslc_date_fmt == "%Y%m%d"
    assert len(opts.cslc_file_list) == 1
    assert isinstance(opts.cslc_file_list[0], Path)

    assert opts.mask_files == []

    # check it's coerced to a list of Paths
    opts2 = config.Inputs(cslc_file_list=[str(opts.cslc_file_list[0])])
    assert isinstance(opts2.cslc_file_list[0], Path)


def test_inputs_bad_filename(tmp_path):
    # make a dummy file
    bad_cslc_file = tmp_path / "nonexistent_slclist.txt"
    with pytest.raises(pydantic.ValidationError, match="does not exist"):
        config.Inputs(cslc_file_list=bad_cslc_file)


def test_input_find_slcs(slc_file_list_nc):
    cslc_dir = Path(slc_file_list_nc[0]).parent

    opts = config.Inputs(cslc_file_list=slc_file_list_nc)
    assert opts.cslc_file_list == slc_file_list_nc

    opts2 = config.Inputs(cslc_file_list=cslc_dir / "slclist.txt")
    opts2.dict() == opts.dict()


def test_input_slc_date_fmt(dir_with_2_slcs):
    expected_slcs = [Path(str(p)) for p in sorted(dir_with_2_slcs.glob("*.nc"))]

    opts = config.Inputs(cslc_file_list=expected_slcs)
    assert opts.cslc_file_list == expected_slcs

    # If files don't all match the date format, it should error
    bad_date_slc = dir_with_2_slcs / "notadate.nc"
    shutil.copy(expected_slcs[0], bad_date_slc)
    new_file_list = dir_with_2_slcs.glob("*.nc")
    with pytest.raises(pydantic.ValidationError):
        opts = config.Inputs(cslc_file_list=new_file_list)

    slc_file1 = dir_with_2_slcs / "2022-01-01.nc"
    slc_file2 = dir_with_2_slcs / "2022-01-02.nc"
    shutil.copy(expected_slcs[0], slc_file1)
    shutil.copy(expected_slcs[1], slc_file2)
    new_file_list = dir_with_2_slcs.glob("2022-*.nc")

    opts = config.Inputs(cslc_file_list=new_file_list, cslc_date_fmt="%Y-%m-%d")
    assert opts.cslc_file_list == [slc_file1, slc_file2]

    # Check that we can get slcs by passing empty string
    # Should match all created files
    all_file_list = list(dir_with_2_slcs.glob("*.nc"))
    opts = config.Inputs(cslc_file_list=all_file_list, cslc_date_fmt="")
    assert set(opts.cslc_file_list) == set(all_file_list)


def test_input_get_dates(dir_with_2_slcs):
    opts = config.Inputs(cslc_file_list=dir_with_2_slcs / "slclist.txt")
    expected = [date(2022, 1, 1), date(2022, 1, 2)]
    assert opts.get_dates() == expected


def test_input_date_sort(dir_with_2_slcs):
    file_list = [Path(str(p)) for p in sorted(dir_with_2_slcs.glob("*.nc"))]
    opts = config.Inputs(cslc_file_list=file_list)
    assert opts.cslc_file_list == file_list

    opts = config.Inputs(cslc_file_list=reversed(file_list))
    assert opts.cslc_file_list == file_list


def test_input_cslc_empty():
    with pytest.raises(pydantic.ValidationError):
        config.Inputs(cslc_file_list=None)
        config.Inputs(cslc_file_list="")
        config.Inputs(cslc_file_list=[])


def test_config_defaults(dir_with_1_slc):
    c = config.Workflow(inputs={"cslc_file_list": dir_with_1_slc / "slclist.txt"})
    # These should be the defaults
    assert c.outputs == config.Outputs()
    assert c.worker_settings == config.WorkerSettings()
    assert c.inputs == config.Inputs(cslc_file_list=dir_with_1_slc / "slclist.txt")

    # Check the defaults for the sub-configs, where the folders
    # should have been moved to the scratch directory
    assert c.ps_options.directory == Path("scratch/PS").absolute()
    assert c.ps_options.amp_mean_file == Path("scratch/PS/amp_mean.tif").absolute()

    p = Path("scratch/PS/amp_dispersion.tif")
    assert c.ps_options.amp_dispersion_file == p.absolute()

    assert c.phase_linking.directory == Path("scratch/linked_phase").absolute()

    assert (
        c.interferogram_network.directory == Path("scratch/interferograms").absolute()
    )
    assert c.interferogram_network.reference_idx == 0
    assert (
        c.interferogram_network.network_type
        == InterferogramNetworkType.SINGLE_REFERENCE
    )

    assert c.interferogram_network.max_bandwidth is None
    assert c.interferogram_network.max_temporal_baseline is None

    assert c.unwrap_options.directory == Path("scratch/unwrap").absolute()

    now = datetime.utcnow()
    assert (now - c.creation_time_utc).seconds == 0


def test_config_create_dir_tree(tmpdir, slc_file_list_nc):
    fname0 = "slc_20220101.nc"
    shutil.copy(slc_file_list_nc[0], tmpdir / fname0)

    with tmpdir.as_cwd():
        c = config.Workflow(inputs={"cslc_file_list": [fname0]})
        c.create_dir_tree()
        assert c.ps_options.directory.exists()
        assert c.interferogram_network.directory.exists()
        assert c.phase_linking.directory.exists()
        assert c.unwrap_options.directory.exists()

        # Check that the scratch directory is created
        assert Path("scratch").exists()

        for d in c._directory_list:
            assert d.exists()
            assert d.is_dir()


def test_config_roundtrip_dict(dir_with_1_slc):
    c = config.Workflow(inputs={"cslc_file_list": dir_with_1_slc / "slclist.txt"})
    c_dict = c.dict()
    c2 = config.Workflow.parse_obj(c_dict)
    assert c == c2


def test_config_roundtrip_json(dir_with_1_slc):
    c = config.Workflow(inputs={"cslc_file_list": dir_with_1_slc / "slclist.txt"})
    c_json = c.json()
    c2 = config.Workflow.parse_raw(c_json)
    assert c == c2


def test_config_roundtrip_yaml(tmp_path, dir_with_1_slc):
    c = config.Workflow(inputs={"cslc_file_list": dir_with_1_slc / "slclist.txt"})
    outfile = tmp_path / "config.yaml"
    c = config.Workflow(inputs={"cslc_file_list": dir_with_1_slc / "slclist.txt"})
    c.to_yaml(outfile)
    c2 = config.Workflow.from_yaml(outfile)
    assert c == c2
