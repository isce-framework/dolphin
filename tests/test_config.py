import datetime
from pathlib import Path

import pydantic
import pytest

from dolphin.workflows import config


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
        assert opts.compressed_slc_file == Path("linked_phase/compressed_slc.tif")
        assert opts.temp_coh_file == Path("linked_phase/temp_coh.tif")


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
        assert opts.network_type == config.InterferogramNetworkType.SINGLE_REFERENCE


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


def test_inputs_defaults():
    opts = config.Inputs(cslc_directory=".")
    assert opts.cslc_directory == Path(".").absolute()

    assert opts.cslc_file_ext == ".nc"
    assert opts.cslc_date_fmt == "%Y%m%d"
    assert opts.cslc_file_list == []

    assert opts.mask_files == []


def test_input_find_slcs(tmpdir):
    files = []
    for n in range(20220101, 20220105):
        slc_file = tmpdir / f"{n}.nc"
        slc_file.write("")
        files.append(slc_file)

    opts = config.Inputs(cslc_directory=tmpdir)
    assert opts.cslc_file_list == files

    opts2 = config.Inputs(cslc_file_list=files)
    opts2.dict() == opts.dict()

    opts_empty = config.Inputs(cslc_directory=tmpdir, cslc_file_ext=".tif")
    assert opts_empty.cslc_file_list == []


def test_input_slc_date_fmt(tmpdir):
    slc_file0 = tmpdir / "20220101.nc"
    slc_file0.write("")
    opts = config.Inputs(cslc_directory=tmpdir)
    assert opts.cslc_file_list == [slc_file0]

    bad_date_slc = tmpdir / "notadate.nc"
    bad_date_slc.write("")
    opts = config.Inputs(cslc_directory=tmpdir)
    assert opts.cslc_file_list == [slc_file0]

    slc_file1 = tmpdir / "2022-01-01.nc"
    slc_file1.write("")
    slc_file2 = tmpdir / "2022-01-02.nc"
    slc_file2.write("")
    opts = config.Inputs(cslc_directory=tmpdir, cslc_date_fmt="%Y-%m-%d")
    assert opts.cslc_file_list == [slc_file1, slc_file2]

    # Check that we can get slcs by passing empty string
    # Should match all created files
    opts = config.Inputs(cslc_directory=tmpdir, cslc_date_fmt="")
    assert opts.cslc_file_list == [slc_file1, slc_file2, slc_file0, bad_date_slc]


def test_input_no_cslc_directory():
    with pytest.raises(pydantic.ValidationError):
        config.Inputs(cslc_directory=None)


def test_input_nones(tmpdir):
    config.Inputs(
        cslc_directory=None, cslc_file_list=["20220101.nc"], cslc_file_ext=None
    )
    with pytest.raises(pydantic.ValidationError):
        config.Inputs(cslc_file_ext=".nc")
        config.Inputs(cslc_directory=None, cslc_file_ext=".nc")
        config.Inputs(cslc_directory=".", cslc_file_ext=None)


def test_config_defaults():
    c = config.Config(inputs={"cslc_directory": "."})
    # These should be the defaults
    assert c.interferogram_network == config.InterferogramNetwork()
    assert c.outputs == config.Outputs()
    assert c.worker_settings == config.WorkerSettings()
    assert c.inputs == config.Inputs(cslc_directory=".")

    # Check the defaults for the sub-configs, where the folders
    # should have been moved to the scratch directory
    assert c.ps_options.directory == Path("scratch/PS").absolute()
    assert c.ps_options.amp_mean_file == Path("scratch/PS/amp_mean.tif").absolute()

    p = Path("scratch/PS/amp_dispersion.tif")
    assert c.ps_options.amp_dispersion_file == p.absolute()

    assert c.phase_linking.directory == Path("scratch/linked_phase").absolute()
    p = Path("scratch/linked_phase/compressed_slc.tif")
    assert c.phase_linking.compressed_slc_file == p.absolute()
    p = Path("scratch/linked_phase/temp_coh.tif")
    assert c.phase_linking.temp_coh_file == p.absolute()

    assert c.unwrap_options.directory == Path("scratch/unwrap").absolute()

    now = datetime.datetime.utcnow()
    assert (now - c.creation_time_utc).seconds == 0


def test_config_create_dir_tree(tmpdir):
    with tmpdir.as_cwd():
        c = config.Config(inputs={"cslc_directory": "."})
        c.create_dir_tree()
        assert c.ps_options.directory.exists()
        assert c.phase_linking.directory.exists()
        assert c.unwrap_options.directory.exists()

        # Check that the scratch directory is created
        assert Path("scratch").exists()

        for d in c._directory_list:
            assert d.exists()
            assert d.is_dir()


def test_config_roundtrip_dict():
    c = config.Config(inputs={"cslc_directory": "."})
    c_dict = c.dict()
    c2 = config.Config.parse_obj(c_dict)
    assert c == c2


def test_config_roundtrip_json():
    c = config.Config(inputs={"cslc_directory": "."})
    c_json = c.json()
    c2 = config.Config.parse_raw(c_json)
    assert c == c2


def test_config_roundtrip_yaml(tmp_path):
    outfile = tmp_path / "config.yaml"
    c = config.Config(inputs={"cslc_directory": "."})
    c.to_yaml(outfile)
    c2 = config.Config.from_yaml(outfile)
    assert c == c2
