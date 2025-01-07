import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import scipy
from packaging import version

import dolphin.unwrap
from dolphin import io
from dolphin.workflows import SpurtOptions, TophuOptions, UnwrapMethod, UnwrapOptions

TOPHU_INSTALLED = importlib.util.find_spec("tophu") is not None
SPURT_INSTALLED = importlib.util.find_spec("spurt") is not None
WHIRLWIND_INSTALLED = importlib.util.find_spec("whirlwind") is not None

if version.parse(scipy.__version__) >= version.parse("1.15.0"):
    # XXX: The current whirlwind implementation breaks for recent scipy
    WHIRLWIND_INSTALLED = False


# Dataset has no geotransform, gcps, or rpcs. The identity matrix will be returned.
pytestmark = pytest.mark.filterwarnings(
    "ignore::rasterio.errors.NotGeoreferencedWarning",
    "ignore:.*io.FileIO.*:pytest.PytestUnraisableExceptionWarning",
    # TODO: Remove this when spurt removes `fork`:
    # RuntimeWarning: os.fork() was called. ...
    # ... unraisableexception.py:80: PytestUnraisableExceptionWarning
    "ignore::pytest.PytestUnraisableExceptionWarning",
)


@pytest.fixture()
def corr_raster(raster_100_by_200):
    # Make a correlation raster of all 1s in the same directory as the raster
    d = Path(raster_100_by_200).parent
    corr_raster = d / "corr_raster.cor.tif"
    array = np.ones((100, 200), dtype=np.float32)
    array[70, 100] = 0
    io.write_arr(
        arr=array,
        output_name=corr_raster,
        like_filename=raster_100_by_200,
        driver="GTiff",
    )
    return corr_raster


@pytest.fixture()
def unwrap_options():
    return UnwrapOptions()


class TestUnwrapSingle:
    def test_unwrap_snaphu_default(self, tmp_path, list_of_gtiff_ifgs, corr_raster):
        unw_filename = tmp_path / "unwrapped.unw.tif"
        unw_path, conncomp_path = dolphin.unwrap.unwrap(
            ifg_filename=list_of_gtiff_ifgs[0],
            corr_filename=corr_raster,
            unw_filename=unw_filename,
            nlooks=1,
        )
        assert unw_path == unw_filename
        assert str(conncomp_path) == str(unw_filename).replace(
            ".unw.tif", ".unw.conncomp.tif"
        )
        assert io.get_raster_xysize(unw_filename) == io.get_raster_xysize(
            list_of_gtiff_ifgs[0]
        )

    @pytest.mark.parametrize("init_method", ["mst", "mcf"])
    def test_unwrap_snaphu(
        self, tmp_path, list_of_gtiff_ifgs, corr_raster, init_method, unwrap_options
    ):
        # test other init_method
        unw_filename = tmp_path / "unwrapped.unw.tif"
        unwrap_options.snaphu_options.init_method = init_method
        unw_path, conncomp_path = dolphin.unwrap.unwrap(
            ifg_filename=list_of_gtiff_ifgs[0],
            corr_filename=corr_raster,
            unw_filename=unw_filename,
            nlooks=1,
            unwrap_options=unwrap_options,
        )
        assert unw_path.exists()
        assert conncomp_path.exists()
        assert io.get_raster_nodata(unw_path) == 0
        assert io.get_raster_nodata(conncomp_path) == 65535

    @pytest.mark.parametrize("method", [UnwrapMethod.ICU, UnwrapMethod.PHASS])
    def test_unwrap_methods(self, tmp_path, raster_100_by_200, corr_raster, method):
        unw_filename = tmp_path / f"{method.value}_unwrapped.unw.tif"
        unwrap_options = UnwrapOptions(unwrap_method=method)
        u_path, c_path = dolphin.unwrap.unwrap(
            ifg_filename=raster_100_by_200,
            corr_filename=corr_raster,
            unw_filename=unw_filename,
            nlooks=1,
            unwrap_options=unwrap_options,
        )
        assert u_path.exists()
        assert c_path.exists()

    def test_unwrap_logfile(self, tmp_path, raster_100_by_200, corr_raster):
        unw_filename = tmp_path / "unwrapped.unw.tif"
        unwrap_options = UnwrapOptions(unwrap_method="icu")
        u_path, c_path = dolphin.unwrap.unwrap(
            ifg_filename=raster_100_by_200,
            corr_filename=corr_raster,
            unw_filename=unw_filename,
            unwrap_options=unwrap_options,
            nlooks=1,
            log_to_file=True,
        )
        logfile_name = str(unw_filename).replace(".unw.tif", ".unw.log")
        assert Path(logfile_name).exists()
        assert u_path.exists()
        assert c_path.exists()

    def test_unwrap_snaphu_nodata(
        self, tmp_path, list_of_gtiff_ifgs, corr_raster, unwrap_options
    ):
        # test other init_method
        unw_filename = tmp_path / "unwrapped.unw.tif"
        unw_path, conncomp_path = dolphin.unwrap.unwrap(
            ifg_filename=list_of_gtiff_ifgs[0],
            corr_filename=corr_raster,
            unw_filename=unw_filename,
            unwrap_options=unwrap_options,
            nlooks=1,
            ccl_nodata=123,
            unw_nodata=np.nan,
        )
        assert unw_path.exists()
        assert conncomp_path.exists()
        assert np.isnan(io.get_raster_nodata(unw_path))
        assert io.get_raster_nodata(conncomp_path) == 123

    def test_goldstein(
        self,
        tmp_path,
        list_of_gtiff_ifgs,
        corr_raster,
        unwrap_options,
    ):
        # test other init_method
        unw_filename = tmp_path / "unwrapped.unw.tif"
        scratch_dir = tmp_path / "scratch"
        scratch_dir.mkdir()

        unwrap_options.run_goldstein = True
        unw_path, conncomp_path = dolphin.unwrap.unwrap(
            ifg_filename=list_of_gtiff_ifgs[0],
            corr_filename=corr_raster,
            unw_filename=unw_filename,
            nlooks=1,
            unwrap_options=unwrap_options,
            scratchdir=scratch_dir,
        )
        assert unw_path.exists()
        assert conncomp_path.exists()
        # Check there are no extraneous ".filter" files
        unw_dir = Path(unw_path).parent
        assert set(unw_dir.glob("*.unw.tif")) == {unw_path}

    @pytest.mark.parametrize("method", [UnwrapMethod.SNAPHU, UnwrapMethod.PHASS])
    @pytest.mark.parametrize("run_interpolation", [False, True])
    @pytest.mark.parametrize("run_goldstein", [False, True])
    def test_multiple_preprocess(
        self,
        tmp_path,
        list_of_gtiff_ifgs,
        corr_raster,
        unwrap_options,
        method,
        run_interpolation,
        run_goldstein,
    ):
        # test other init_method
        unw_filename = tmp_path / "unwrapped.unw.tif"
        scratch_dir = tmp_path / "scratch"
        scratch_dir.mkdir()
        unwrap_options.unwrap_method = method
        unwrap_options.run_interpolation = run_interpolation
        unwrap_options.run_goldstein = run_goldstein
        unw_path, conncomp_path = dolphin.unwrap.unwrap(
            ifg_filename=list_of_gtiff_ifgs[0],
            corr_filename=corr_raster,
            unw_filename=unw_filename,
            nlooks=1,
            unwrap_options=unwrap_options,
            scratchdir=scratch_dir,
        )
        assert unw_path.exists()
        assert conncomp_path.exists()
        # Check there are no extraneous ".interp" files
        unw_dir = Path(unw_path).parent
        assert set(unw_dir.glob("*.unw.tif")) == {unw_path}

    @pytest.mark.parametrize("method", [UnwrapMethod.SNAPHU, UnwrapMethod.PHASS])
    def test_interpolation(self, tmp_path, list_of_gtiff_ifgs, corr_raster, method):
        # test other init_method
        unw_filename = tmp_path / "unwrapped.unw.tif"
        scratch_dir = tmp_path / "scratch"
        scratch_dir.mkdir()
        unwrap_options = UnwrapOptions(unwrap_method=method, run_interpolation=True)
        unw_path, conncomp_path = dolphin.unwrap.unwrap(
            ifg_filename=list_of_gtiff_ifgs[0],
            corr_filename=corr_raster,
            unw_filename=unw_filename,
            nlooks=1,
            unwrap_options=unwrap_options,
            scratchdir=scratch_dir,
        )
        assert unw_path.exists()
        assert conncomp_path.exists()
        # Check there are no extraneous ".interp" files
        unw_dir = Path(unw_path).parent
        assert set(unw_dir.glob("*.unw.tif")) == {unw_path}


class TestUnwrapRun:
    def test_run_gtiff(self, list_of_gtiff_ifgs, corr_raster, unwrap_options):
        ifg_path = list_of_gtiff_ifgs[0].parent
        u_paths, c_paths = dolphin.unwrap.run(
            ifg_filenames=list_of_gtiff_ifgs,
            cor_filenames=[corr_raster] * len(list_of_gtiff_ifgs),
            output_path=ifg_path,
            nlooks=1,
            unwrap_options=unwrap_options,
        )
        assert all(p.exists() for p in u_paths)
        assert all(p.exists() for p in c_paths)

    def test_run_envi(self, list_of_envi_ifgs, corr_raster, unwrap_options):
        ifg_path = list_of_envi_ifgs[0].parent
        unwrap_options.snaphu_options.init_method = "mcf"
        u_paths, c_paths = dolphin.unwrap.run(
            ifg_filenames=list_of_envi_ifgs,
            cor_filenames=[corr_raster] * len(list_of_envi_ifgs),
            output_path=ifg_path,
            nlooks=1,
            unwrap_options=unwrap_options,
        )
        assert all(p.exists() for p in u_paths)
        assert all(p.exists() for p in c_paths)


class TestTophu:
    @pytest.mark.skipif(
        not TOPHU_INSTALLED, reason="tophu not installed for multiscale unwrapping"
    )
    def test_unwrap_multiscale(self, tmp_path, raster_100_by_200, corr_raster):
        unw_filename = tmp_path / "unwrapped.unw.tif"

        to = TophuOptions(ntiles=(2, 2), downsample_factor=(3, 3))
        unwrap_options = UnwrapOptions(unwrap_method="phass", tophu_options=to)
        out_path, conncomp_path = dolphin.unwrap.unwrap(
            ifg_filename=raster_100_by_200,
            corr_filename=corr_raster,
            unw_filename=unw_filename,
            unwrap_options=unwrap_options,
            nlooks=1,
        )
        assert out_path.exists()
        assert conncomp_path.exists()

    @pytest.mark.skipif(
        not TOPHU_INSTALLED, reason="tophu not installed for multiscale unwrapping"
    )
    def test_unwrap_multiscale_callback_given(
        self, tmp_path, raster_100_by_200, corr_raster
    ):
        from tophu import ICUUnwrap

        unw_filename = tmp_path / "unwrapped.unw.tif"
        unwrap_callback = ICUUnwrap()
        out_path, conncomp_path = dolphin.unwrap.multiscale_unwrap(
            ifg_filename=raster_100_by_200,
            corr_filename=corr_raster,
            unw_filename=unw_filename,
            downsample_factor=(2, 2),
            ntiles=(2, 2),
            unwrap_callback=unwrap_callback,
            unw_nodata=0,
            nlooks=1,
        )
        assert out_path.exists()
        assert conncomp_path.exists()


class TestSpurt:
    @pytest.fixture()
    def ifg_file_list(self, tmp_path, slc_date_list):
        from dolphin import io
        from dolphin.phase_link import simulate

        slc_stack = np.exp(1j * simulate.make_defo_stack((10, 20, 30), sigma=1)).astype(
            "complex64"
        )
        ifg_stack = slc_stack[1:] * slc_stack[[0]].conj()
        # Write to a file
        d = tmp_path / "gtiff"
        d.mkdir()
        name_template = d / f"{slc_date_list[0].strftime('%Y%m%d')}_{{date}}.int.tif"

        file_list = []
        for cur_date, cur_ifg in zip(slc_date_list[1:], ifg_stack):
            fname = str(name_template).format(date=cur_date.strftime("%Y%m%d"))
            file_list.append(Path(fname))
            io.write_arr(arr=cur_ifg, output_name=fname)

        return file_list

    @pytest.fixture()
    def temp_coh_raster(self, ifg_file_list):
        d = Path(ifg_file_list[0]).parent
        coh_raster = d / "temporal_coherence.tif"
        io.write_arr(
            arr=np.random.rand(20, 30).astype(np.float32),
            output_name=coh_raster,
            driver="GTiff",
        )
        return coh_raster

    @pytest.mark.skipif(
        not SPURT_INSTALLED, reason="spurt not installed for 3d unwrapping"
    )
    def test_unwrap_spurt(self, tmp_path, ifg_file_list, temp_coh_raster):
        opts = SpurtOptions()
        unwrap_options = UnwrapOptions(unwrap_method="spurt", spurt_options=opts)
        out_paths, conncomp_paths = dolphin.unwrap.run(
            ifg_filenames=ifg_file_list,
            cor_filenames=ifg_file_list,  # NOT USED... but required for `run`?
            temporal_coherence_filename=temp_coh_raster,
            unwrap_options=unwrap_options,
            output_path=tmp_path,
            nlooks=5,
        )
        assert all(p.exists() for p in out_paths)
        # spurt gives 0 conncomps for now:
        # TODO: Uncomment this if spurt starts making conncomps
        # assert all(p.exists() for p in conncomp_paths)


@pytest.mark.skipif(not WHIRLWIND_INSTALLED, reason="whirlwind package not installed")
class TestWhirlwind:
    def test_unwrap_whirlwind(self, tmp_path, raster_100_by_200, corr_raster):
        unw_filename = tmp_path / "whirlwind-unwrapped.unw.tif"

        unwrap_options = UnwrapOptions(unwrap_method="whirlwind")
        out_path, conncomp_path = dolphin.unwrap.unwrap(
            ifg_filename=raster_100_by_200,
            corr_filename=corr_raster,
            unw_filename=unw_filename,
            unwrap_options=unwrap_options,
            nlooks=1,
        )
        assert out_path.exists()
        assert conncomp_path.exists()


@pytest.mark.skipif(os.environ.get("NUMBA_DISABLE_JIT") == "1", reason="JIT disabled")
def test_compute_phase_diffs():
    # test on a 2D array with no phase jumps > pi
    phase1 = np.array([[0, 1], [1, 2]], dtype=float)
    expected1 = np.array([[0, 0], [0, 0]], dtype=float)
    assert np.allclose(dolphin.unwrap.compute_phase_diffs(phase1), expected1)

    # test on a 2D array with some phase jumps > pi at the top-left pixel
    phase2 = np.array([[0, 3.15], [3.15, 0]], dtype=float)
    expected2 = np.array([[2, 0], [0, 0]], dtype=float)
    assert np.allclose(dolphin.unwrap.compute_phase_diffs(phase2), expected2)

    # test on a larger 2D array
    phase3 = np.full((10, 10), np.pi, dtype=float)
    expected3 = np.zeros((10, 10), dtype=float)
    assert np.allclose(dolphin.unwrap.compute_phase_diffs(phase3), expected3)
