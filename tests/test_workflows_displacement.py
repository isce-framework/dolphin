from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from opera_utils import group_by_burst

from dolphin.io import get_raster_nodata, get_raster_units
from dolphin.utils import flatten, full_suffix
from dolphin.workflows import config, displacement

pytestmark = pytest.mark.filterwarnings(
    "ignore::rasterio.errors.NotGeoreferencedWarning",
    "ignore:.*io.FileIO.*:pytest.PytestUnraisableExceptionWarning",
)


def test_displacement_run_single(opera_slc_files: list[Path], tmpdir):
    with tmpdir.as_cwd():
        cfg = config.DisplacementWorkflow(
            cslc_file_list=opera_slc_files,
            input_options={"subdataset": "/data/VV"},
            interferogram_network={
                "indexes": [(0, -1)],
                "max_bandwidth": 2,
            },
            phase_linking={
                "ministack_size": 500,
            },
        )
        paths = displacement.run(cfg)

        for slc_paths in paths.comp_slc_dict.values():
            assert all(p.exists() for p in slc_paths)
        assert paths.stitched_ps_file.exists()
        assert all(p.exists() for p in paths.stitched_ifg_paths)
        assert all(p.exists() for p in paths.stitched_cor_paths)
        assert paths.stitched_temp_coh_file.exists()
        assert paths.stitched_ps_file.exists()
        assert paths.stitched_amp_dispersion_file.exists()
        assert paths.unwrapped_paths is not None
        assert paths.conncomp_paths is not None
        assert paths.timeseries_paths is not None
        assert all(p.exists() for p in paths.conncomp_paths)
        assert all(p.exists() for p in paths.unwrapped_paths)
        assert all(full_suffix(p) == ".unw.tif" for p in paths.unwrapped_paths)
        assert all(p.exists() for p in paths.conncomp_paths)
        assert all(p.exists() for p in paths.timeseries_paths)
        assert all(full_suffix(p) == ".tif" for p in paths.timeseries_paths)

        # check the network size
        assert len(paths.unwrapped_paths) > len(paths.timeseries_paths)

        # Check nodata values
        assert get_raster_nodata(paths.unwrapped_paths[0]) is not None
        assert get_raster_nodata(paths.unwrapped_paths[0]) == get_raster_nodata(
            paths.timeseries_paths[0]
        )


def test_displacement_run_single_official_opera_naming(
    opera_slc_files_official: list[Path],
    # weather_model_files: list[Path],
    # tec_files: list[Path],
    # dem_file: Path,
    # opera_static_files_official: list[Path],
    tmpdir,
):
    with tmpdir.as_cwd():
        cfg = config.DisplacementWorkflow(
            cslc_file_list=opera_slc_files_official,
            input_options={"subdataset": "/data/VV"},
            output_options={"strides": {"x": 2, "y": 2}},
            interferogram_network={"max_bandwidth": 1},
            phase_linking={
                "ministack_size": 500,
            },
            # # TODO: this is not working
            # # either move to disp-s1 test with real data,
            # # or.. something else
            # correction_options={
            #     "troposphere_files": weather_model_files,
            #     "ionosphere_files": tec_files,
            #     "dem_file": dem_file,
            #     "geometry_files": opera_static_files_official,
            # },
            unwrap_options={"run_unwrap": True},
        )
        paths = displacement.run(cfg)
        assert paths.timeseries_paths is not None
        assert all(p.exists() for p in paths.timeseries_paths)
        assert paths.unwrapped_paths is not None
        assert all(get_raster_units(p) == "radians" for p in paths.unwrapped_paths)
        assert all(get_raster_units(p) == "meters" for p in paths.timeseries_paths)
        assert all(full_suffix(p) == ".tif" for p in paths.timeseries_paths)


def run_displacement_stack(
    path, file_list: list[Path], run_unwrap: bool = False, ministack_size: int = 500
):
    cfg = config.DisplacementWorkflow(
        cslc_file_list=file_list,
        input_options={"subdataset": "/data/VV"},
        work_directory=path,
        phase_linking={
            "ministack_size": ministack_size,
        },
        unwrap_options={"run_unwrap": run_unwrap},
        log_file=Path() / "dolphin.log",
    )
    paths = displacement.run(cfg)
    if run_unwrap:
        assert paths.timeseries_paths is not None
        assert all(full_suffix(p) == ".tif" for p in paths.timeseries_paths)


def test_stack_with_compSLCs(opera_slc_files, tmpdir):
    with tmpdir.as_cwd():
        p1 = Path("first_run")
        run_displacement_stack(p1, opera_slc_files, run_unwrap=True)
        # Find the compressed SLC files
        new_comp_slcs = sorted(p1.rglob("compressed_*"))

        p2 = Path("second_run")
        # Add the first compressed SLC in place of first real one and run again
        by_burst = group_by_burst(opera_slc_files)
        new_real_slcs = list(flatten(v[1:] for v in by_burst.values()))
        new_file_list = new_comp_slcs + new_real_slcs

        run_displacement_stack(p2, new_file_list)

        # Now the results should be the same (for the file names)
        # check the ifg folders
        ifgs1 = sorted((p1 / "interferograms").glob("*.int.tif"))
        ifgs2 = sorted((p2 / "interferograms").glob("*.int.tif"))
        assert len(ifgs1) > 0
        assert [f.name for f in ifgs1] == [f.name for f in ifgs2]


def test_separate_workflow_runs(slc_file_list, tmp_path):
    """Check that manually running the workflow results in the same
    interferograms as one sequential run.
    """
    p_all = tmp_path / "all"
    run_displacement_stack(p_all, slc_file_list, ministack_size=10)
    all_ifgs = sorted((p_all / "interferograms").glob("*.int.tif"))
    assert len(all_ifgs) == 29

    p1 = tmp_path / Path("first")
    ms = 10
    # Split into batches of 10
    file_batches = [slc_file_list[i : i + ms] for i in range(0, len(slc_file_list), ms)]
    assert len(file_batches) == 3
    assert all(len(b) == 10 for b in file_batches)
    run_displacement_stack(p1, file_batches[0])
    new_comp_slcs1 = sorted((p1 / "linked_phase").glob("compressed_*"))
    assert len(new_comp_slcs1) == 1
    ifgs1 = sorted((p1 / "interferograms").glob("*.int.tif"))
    assert len(ifgs1) == 9

    p2 = tmp_path / Path("second")
    files2 = new_comp_slcs1 + file_batches[1]
    run_displacement_stack(p2, files2)
    new_comp_slcs2 = sorted((p2 / "linked_phase").glob("compressed_*"))
    assert len(new_comp_slcs2) == 1
    ifgs2 = sorted((p2 / "interferograms").glob("*.int.tif"))
    assert len(ifgs2) == 10

    p3 = tmp_path / Path("third")
    files3 = new_comp_slcs1 + new_comp_slcs2 + file_batches[2]
    run_displacement_stack(p3, files3)
    ifgs3 = sorted((p3 / "interferograms").glob("*.int.tif"))
    assert len(ifgs3) == 10

    all_ifgs_names = [f.name for f in all_ifgs]
    batched_names = [f.name for f in ifgs1 + ifgs2 + ifgs3]
    assert all_ifgs_names == batched_names

    # Last, try one where we dont have the first CCSLC
    # The metadata should still tell it what the reference date is,
    # So the outputs should be the same
    p3_b = tmp_path / Path("third")
    files3_b = new_comp_slcs2 + file_batches[2]
    run_displacement_stack(p3_b, files3_b)
    ifgs3_b = sorted((p3_b / "interferograms").glob("*.int.tif"))
    assert len(ifgs3_b) == 10
    # Names should be the same as the previous run
    assert [f.name for f in ifgs3_b] == [f.name for f in ifgs3]


@pytest.mark.parametrize("unwrap_method", ["phass", "spurt"])
def test_displacement_run_extra_reference_date(
    opera_slc_files: list[Path], tmpdir, unwrap_method: str
):
    if unwrap_method == "spurt" and importlib.util.find_spec("spurt") is None:
        pytest.skip(reason="spurt unwrapper not installed")

    with tmpdir.as_cwd():
        log_file = Path() / "dolphin.log"
        cfg = config.DisplacementWorkflow(
            # start_date = 20220101
            # shape = (4, 128, 128)
            # First one is COMPRESSED_
            output_options={"extra_reference_date": "2022-01-03"},
            unwrap_options={"unwrap_method": unwrap_method},
            cslc_file_list=opera_slc_files,
            input_options={"subdataset": "/data/VV"},
            phase_linking={
                "ministack_size": 4,
            },
            log_file=log_file,
        )
        paths = displacement.run(cfg)

        for slc_paths in paths.comp_slc_dict.values():
            # The "base phase" should be 20220103
            assert slc_paths[0].name == "compressed_20220103_20220102_20220104.tif"

        # The unwrapped/timeseries files should have a changeover to the new reference
        assert paths.unwrapped_paths is not None
        assert paths.timeseries_paths is not None

        ts_names = [pp.name for pp in paths.timeseries_paths]
        assert ts_names == [
            "20220101_20220102.tif",
            "20220101_20220103.tif",
            "20220103_20220104.tif",
        ]
        assert all(get_raster_units(p) == "meters" for p in paths.timeseries_paths)

        unw_names = [pp.name for pp in paths.unwrapped_paths]
        if cfg.unwrap_options.unwrap_method == "spurt":
            assert unw_names == [
                "20220101_20220102.unw.tif",
                "20220101_20220103.unw.tif",
                "20220101_20220104.unw.tif",
                "20220102_20220103.unw.tif",
                "20220102_20220104.unw.tif",
                "20220103_20220104.unw.tif",
            ]
        else:
            assert unw_names == [
                "20220101_20220102.unw.tif",
                "20220101_20220103.unw.tif",
                "20220103_20220104.unw.tif",
            ]

        assert all(get_raster_units(p) == "radians" for p in paths.unwrapped_paths)
        log_file.unlink()
