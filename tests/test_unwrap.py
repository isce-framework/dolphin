import os
from pathlib import Path

import numpy as np
import pytest

import dolphin.unwrap
from dolphin import io
from dolphin.workflows import UnwrapMethod

try:
    import tophu

    TOPHU_INSTALLED = True
except ImportError:
    TOPHU_INSTALLED = False

# Dataset has no geotransform, gcps, or rpcs. The identity matrix will be returned.
pytestmark = pytest.mark.filterwarnings(
    "ignore::rasterio.errors.NotGeoreferencedWarning",
    "ignore:.*io.FileIO.*:pytest.PytestUnraisableExceptionWarning",
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
        self, tmp_path, list_of_gtiff_ifgs, corr_raster, init_method
    ):
        # test other init_method
        unw_filename = tmp_path / "unwrapped.unw.tif"
        unw_path, conncomp_path = dolphin.unwrap.unwrap(
            ifg_filename=list_of_gtiff_ifgs[0],
            corr_filename=corr_raster,
            unw_filename=unw_filename,
            nlooks=1,
            init_method=init_method,
        )
        assert unw_path.exists()
        assert conncomp_path.exists()
        assert io.get_raster_nodata(unw_path) == 0
        assert io.get_raster_nodata(conncomp_path) == 65535

    @pytest.mark.parametrize("method", [UnwrapMethod.ICU, UnwrapMethod.PHASS])
    def test_unwrap_methods(self, tmp_path, raster_100_by_200, corr_raster, method):
        unw_filename = tmp_path / f"{method.value}_unwrapped.unw.tif"
        u_path, c_path = dolphin.unwrap.unwrap(
            ifg_filename=raster_100_by_200,
            corr_filename=corr_raster,
            unw_filename=unw_filename,
            nlooks=1,
            unwrap_method=method,
        )
        assert u_path.exists()
        assert c_path.exists()

    def test_unwrap_logfile(self, tmp_path, raster_100_by_200, corr_raster):
        unw_filename = tmp_path / "unwrapped.unw.tif"
        u_path, c_path = dolphin.unwrap.unwrap(
            ifg_filename=raster_100_by_200,
            corr_filename=corr_raster,
            unw_filename=unw_filename,
            nlooks=1,
            unwrap_method="icu",
            log_to_file=True,
        )
        logfile_name = str(unw_filename).replace(".unw.tif", ".unw.log")
        assert Path(logfile_name).exists()
        assert u_path.exists()
        assert c_path.exists()

    def test_unwrap_snaphu_nodata(self, tmp_path, list_of_gtiff_ifgs, corr_raster):
        # test other init_method
        unw_filename = tmp_path / "unwrapped.unw.tif"
        unw_path, conncomp_path = dolphin.unwrap.unwrap(
            ifg_filename=list_of_gtiff_ifgs[0],
            corr_filename=corr_raster,
            unw_filename=unw_filename,
            nlooks=1,
            ccl_nodata=123,
            unw_nodata=np.nan,
        )
        assert unw_path.exists()
        assert conncomp_path.exists()
        assert np.isnan(io.get_raster_nodata(unw_path))
        assert io.get_raster_nodata(conncomp_path) == 123

    @pytest.mark.parametrize("method", [UnwrapMethod.SNAPHU, UnwrapMethod.PHASS])
    def test_goldstein(self, tmp_path, list_of_gtiff_ifgs, corr_raster, method):
        # test other init_method
        unw_filename = tmp_path / "unwrapped.unw.tif"
        scratch_dir = tmp_path / "scratch"
        scratch_dir.mkdir()
        unw_path, conncomp_path = dolphin.unwrap.unwrap(
            ifg_filename=list_of_gtiff_ifgs[0],
            corr_filename=corr_raster,
            unw_filename=unw_filename,
            nlooks=1,
            unwrap_method=method,
            run_goldstein=True,
            scratchdir=scratch_dir,
        )
        assert unw_path.exists()
        assert conncomp_path.exists()

    def test_interp_loop(self):
        x, y = np.meshgrid(np.arange(200), np.arange(100))
        # simulate a simple phase ramp
        phase = 0.003 * x + 0.002 * y
        # interferogram with the simulated phase ramp and with a constant amplitude
        ifg = np.exp(1j * phase)
        corr = np.ones(ifg.shape)
        # mask out the ifg/corr at a given pixel for example at pixel 50,40
        # index of the pixel of interest
        x_idx = 50
        y_idx = 40
        corr[y_idx, x_idx] = 0
        # generate indices for the pixels to be used in interpolation
        indices = np.array(
            dolphin.similarity.get_circle_idxs(
                max_radius=51, min_radius=0, sort_output=False
            )
        )
        # interpolate pixels with zero in the corr and write to interpolated_ifg
        interpolated_ifg = np.zeros((100, 200), dtype=np.complex64)
        dolphin.interpolation._interp_loop(
            ifg,
            corr,
            weight_cutoff=0.5,
            num_neighbors=20,
            alpha=0.75,
            indices=indices,
            interpolated_ifg=interpolated_ifg,
        )
        # expected phase based on the model above used for simulation
        expected_phase = 0.003 * x_idx + 0.002 * y_idx
        phase_error = np.angle(
            interpolated_ifg[y_idx, x_idx] * np.exp(-1j * expected_phase)
        )
        assert np.allclose(phase_error, 0.0, atol=1e-3)

    @pytest.mark.parametrize("method", [UnwrapMethod.SNAPHU, UnwrapMethod.PHASS])
    def test_interpolation(self, tmp_path, list_of_gtiff_ifgs, corr_raster, method):
        # test other init_method
        unw_filename = tmp_path / "unwrapped.unw.tif"
        scratch_dir = tmp_path / "scratch"
        scratch_dir.mkdir()
        unw_path, conncomp_path = dolphin.unwrap.unwrap(
            ifg_filename=list_of_gtiff_ifgs[0],
            corr_filename=corr_raster,
            unw_filename=unw_filename,
            nlooks=1,
            unwrap_method=method,
            run_interpolation=True,
            scratchdir=scratch_dir,
        )
        assert unw_path.exists()
        assert conncomp_path.exists()


class TestUnwrapRun:
    def test_run_gtiff(self, list_of_gtiff_ifgs, corr_raster):
        ifg_path = list_of_gtiff_ifgs[0].parent
        u_paths, c_paths = dolphin.unwrap.run(
            ifg_filenames=list_of_gtiff_ifgs,
            cor_filenames=[corr_raster] * len(list_of_gtiff_ifgs),
            output_path=ifg_path,
            nlooks=1,
            init_method="mst",
            max_jobs=1,
        )
        assert all(p.exists() for p in u_paths)
        assert all(p.exists() for p in c_paths)

    def test_run_envi(self, list_of_envi_ifgs, corr_raster):
        ifg_path = list_of_envi_ifgs[0].parent
        u_paths, c_paths = dolphin.unwrap.run(
            ifg_filenames=list_of_envi_ifgs,
            cor_filenames=[corr_raster] * len(list_of_envi_ifgs),
            output_path=ifg_path,
            nlooks=1,
            init_method="mst",
            max_jobs=1,
        )
        assert all(p.exists() for p in u_paths)
        assert all(p.exists() for p in c_paths)


class TestTophu:
    @pytest.mark.skipif(
        not TOPHU_INSTALLED, reason="tophu not installed for multiscale unwrapping"
    )
    def test_unwrap_multiscale(self, tmp_path, raster_100_by_200, corr_raster):
        unw_filename = tmp_path / "unwrapped.unw.tif"
        out_path, conncomp_path = dolphin.unwrap.unwrap(
            ifg_filename=raster_100_by_200,
            corr_filename=corr_raster,
            unw_filename=unw_filename,
            nlooks=1,
            ntiles=(2, 2),
            downsample_factor=(3, 3),
            unwrap_method="phass",
        )
        assert out_path.exists()
        assert conncomp_path.exists()

    @pytest.mark.skipif(
        not TOPHU_INSTALLED, reason="tophu not installed for multiscale unwrapping"
    )
    def test_unwrap_multiscale_callback_given(
        self, tmp_path, raster_100_by_200, corr_raster
    ):
        unw_filename = tmp_path / "unwrapped.unw.tif"
        unwrap_callback = tophu.ICUUnwrap()
        out_path, conncomp_path = dolphin.unwrap.multiscale_unwrap(
            ifg_filename=raster_100_by_200,
            corr_filename=corr_raster,
            unw_filename=unw_filename,
            unwrap_callback=unwrap_callback,
            unw_nodata=0,
            nlooks=1,
            ntiles=(2, 2),
            downsample_factor=(3, 3),
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
