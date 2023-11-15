import pytest

from dolphin import _readers, stack
from dolphin.phase_link import simulate
from dolphin.utils import gpu_is_available
from dolphin.workflows import single

GPU_AVAILABLE = gpu_is_available()
simulate._seed(1234)

# 'Grid size 49 will likely result in GPU under-utilization due to low occupancy.'
pytestmark = pytest.mark.filterwarnings(
    "ignore::numba.core.errors.NumbaPerformanceWarning"
)


@pytest.mark.parametrize("gpu_enabled", [True, False])
def test_sequential_gtiff(tmp_path, slc_file_list, gpu_enabled):
    """Run through the sequential estimation with a GeoTIFF stack."""
    vrt_file = tmp_path / "slc_stack.vrt"
    files = slc_file_list[:3]
    vrt_stack = _readers.VRTStack(files, outfile=vrt_file)
    _, rows, cols = vrt_stack.shape
    is_compressed = [False] * len(files)
    ministack = stack.MiniStackInfo(
        vrt_stack.file_list,
        dates=vrt_stack.dates,
        is_compressed=is_compressed,
    )

    half_window = {"x": cols // 2, "y": rows // 2}
    strides = {"x": 1, "y": 1}
    output_folder = tmp_path / "single"
    single.run_wrapped_phase_single(
        slc_vrt_file=vrt_file,
        ministack=ministack,
        output_folder=output_folder,
        half_window=half_window,
        strides=strides,
        shp_method="rect",
        n_workers=2,
        gpu_enabled=gpu_enabled,
    )

    assert output_folder.exists()
    # Check that all the expected outputs are there
    assert len(list(output_folder.glob("2*.slc.tif"))) == 3
    assert len(list(output_folder.glob("compressed_*tif"))) == 1
    assert len(list(output_folder.glob("tcorr_*tif"))) == 1
