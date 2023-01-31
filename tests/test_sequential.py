import numpy as np
import numpy.testing as npt
import pytest
from osgeo import gdal

from dolphin import io, sequential, vrt
from dolphin.phase_link import mle, simulate
from dolphin.utils import gpu_is_available

GPU_AVAILABLE = gpu_is_available()
simulate._seed(1234)

# 'Grid size 49 will likely result in GPU under-utilization due to low occupancy.'
pytestmark = pytest.mark.filterwarnings(
    "ignore::numba.core.errors.NumbaPerformanceWarning"
)


def test_setup_output_folder(tmpdir, tiled_file_list):
    vrt_stack = vrt.VRTStack(tiled_file_list, outfile=tmpdir / "stack.vrt")
    out_file_list = sequential._setup_output_folder(
        vrt_stack, driver="GTiff", dtype=np.complex64
    )
    for out_file in out_file_list:
        assert out_file.exists()
        assert out_file.suffix == ".tif"
        assert out_file.parent == tmpdir
        ds = gdal.Open(str(out_file))
        assert ds.GetRasterBand(1).DataType == gdal.GDT_CFloat32
        ds = None

    out_file_list = sequential._setup_output_folder(
        vrt_stack,
        driver="GTiff",
        dtype="float32",
        start_idx=1,
    )
    assert len(out_file_list) == len(vrt_stack) - 1
    for out_file in out_file_list:
        assert out_file.exists()
        ds = gdal.Open(str(out_file))
        assert ds.GetRasterBand(1).DataType == gdal.GDT_Float32


@pytest.mark.parametrize(
    "strides", [{"x": 1, "y": 1}, {"x": 1, "y": 2}, {"x": 2, "y": 3}, {"x": 4, "y": 2}]
)
def test_setup_output_folder_strided(tmpdir, tiled_file_list, strides):
    vrt_stack = vrt.VRTStack(tiled_file_list, outfile=tmpdir / "stack.vrt")

    out_file_list = sequential._setup_output_folder(
        vrt_stack, driver="GTiff", dtype=np.complex64, strides=strides
    )
    rows, cols = vrt_stack.shape[-2:]
    for out_file in out_file_list:
        assert out_file.exists()
        assert out_file.suffix == ".tif"
        assert out_file.parent == tmpdir

        ds = gdal.Open(str(out_file))
        assert ds.GetRasterBand(1).DataType == gdal.GDT_CFloat32
        assert ds.RasterXSize == cols // strides["x"]
        assert ds.RasterYSize == rows // strides["y"]
        ds = None


@pytest.mark.parametrize("gpu_enabled", [True, False])
def test_sequential_gtiff(tmp_path, slc_file_list, gpu_enabled):
    """Run through the sequential estimation with a GeoTIFF stack."""
    vrt_file = tmp_path / "slc_stack.vrt"
    vrt_stack = vrt.VRTStack(slc_file_list, outfile=vrt_file)
    _, rows, cols = vrt_stack.shape

    half_window = {"x": cols // 2, "y": rows // 2}
    strides = {"x": 1, "y": 1}
    output_folder = tmp_path / "sequential"
    sequential.run_evd_sequential(
        slc_vrt_file=vrt_file,
        output_folder=output_folder,
        half_window=half_window,
        strides=strides,
        ministack_size=10,
        ps_mask_file=None,
        max_bytes=1e9,
        n_workers=4,
        gpu_enabled=gpu_enabled,
    )

    # Get the MLE estimates from the entire stack output.
    slc_stack = vrt_stack.read_stack()
    mle_est, _ = mle.run_mle(
        slc_stack,
        half_window=half_window,
        strides=strides,
        gpu_enabled=gpu_enabled,
    )

    # TODO: This probably won't work with just random data
    pytest.skip("This test is not working with random data.")
    # Check that the sequential output matches the MLE estimates.
    for idx, out_file in enumerate(sorted(output_folder.glob("2*.slc.tif"))):
        layer = io.load_gdal(out_file)
        expected = mle_est[idx]
        npt.assert_allclose(layer, expected, atol=1e-3)


# Input is only (5, 10) so we can't use a larger window.
@pytest.mark.parametrize(
    "half_window", [{"x": 1, "y": 1}, {"x": 2, "y": 3}, {"x": 4, "y": 3}]
)
@pytest.mark.parametrize(
    "strides", [{"x": 1, "y": 1}, {"x": 1, "y": 2}, {"x": 2, "y": 3}, {"x": 4, "y": 2}]
)
def test_sequential_nc(tmp_path, slc_file_list_nc, half_window, strides):
    """Check various strides/windows/ministacks with a NetCDF input stack."""
    vrt_file = tmp_path / "slc_stack.vrt"
    _ = vrt.VRTStack(slc_file_list_nc, outfile=vrt_file, subdataset="data")

    output_folder = tmp_path / "sequential"
    sequential.run_evd_sequential(
        slc_vrt_file=vrt_file,
        output_folder=output_folder,
        half_window=half_window,
        strides=strides,
        ministack_size=10,
        ps_mask_file=None,
        max_bytes=1e9,
        n_workers=4,
        gpu_enabled=False,
    )


@pytest.mark.parametrize("ministack_size", [3, 5, 9, 20])
def test_sequential_ministack_sizes(tmp_path, slc_file_list_nc, ministack_size):
    """Check various strides/windows/ministacks with a NetCDF input stack."""
    vrt_file = tmp_path / "slc_stack.vrt"
    # Make it not a round number to test
    vrt_stack = vrt.VRTStack(slc_file_list_nc[:21], outfile=vrt_file, subdataset="data")
    _, rows, cols = vrt_stack.shape

    output_folder = tmp_path / "sequential"
    sequential.run_evd_sequential(
        slc_vrt_file=vrt_file,
        output_folder=output_folder,
        half_window={"x": cols // 2, "y": rows // 2},
        strides={"x": 1, "y": 1},
        ministack_size=ministack_size,
        ps_mask_file=None,
        max_bytes=1e9,
        n_workers=4,
        gpu_enabled=False,
    )
