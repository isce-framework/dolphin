import pytest

from dolphin import stack

# from dolphin._types import HalfWindow, Strides
from dolphin.io import _readers
from dolphin.phase_link import simulate
from dolphin.utils import compute_out_shape, gpu_is_available
from dolphin.workflows import sequential

GPU_AVAILABLE = gpu_is_available()
simulate._seed(1234)


def test_sequential_gtiff(tmp_path, slc_file_list):
    """Run through the sequential estimation with a GeoTIFF stack."""
    vrt_file = tmp_path / "slc_stack.vrt"
    vrt_stack = _readers.VRTStack(slc_file_list, outfile=vrt_file)
    _, rows, cols = vrt_stack.shape

    half_window = {"x": cols // 2, "y": rows // 2}
    strides = {"x": 1, "y": 1}
    out_shape = compute_out_shape((rows, cols), strides=(strides["y"], strides["x"]))
    if not all(out_shape):
        pytest.skip(f"Output shape = {out_shape}")
    output_folder = tmp_path / "sequential"
    ms_size = 10
    ms_planner = stack.MiniStackPlanner(
        file_list=slc_file_list,
        dates=vrt_stack.dates,
        is_compressed=[False] * len(slc_file_list),
        output_folder=output_folder,
    )

    sequential.run_wrapped_phase_sequential(
        slc_vrt_file=vrt_file,
        ministack_planner=ms_planner,
        ministack_size=ms_size,
        half_window=half_window,
        strides=strides,
        ps_mask_file=None,
        amp_mean_file=None,
        amp_dispersion_file=None,
        shp_method="rect",
        shp_alpha=None,
        shp_nslc=None,
    )

    assert len(list(output_folder.glob("2*.slc.tif"))) == vrt_stack.shape[0]


# Input is only (5, 10) so we can't use a larger window.
@pytest.mark.parametrize(
    "half_window, strides",
    [
        ({"x": 1, "y": 1}, {"x": 1, "y": 1}),
        ({"x": 2, "y": 1}, {"x": 3, "y": 2}),
        ({"x": 3, "y": 1}, {"x": 2, "y": 1}),
        # (HalfWindow(1, 1), Strides(1, 1)),
        # (HalfWindow(1, 2), Strides(2, 3)),
        # (HalfWindow(1, 2), Strides(1, 3)),
    ],
)
def test_sequential_nc(tmp_path, slc_file_list_nc, half_window, strides):
    """Check various strides/windows/ministacks with a NetCDF input stack."""
    vrt_file = tmp_path / "slc_stack.vrt"
    v = _readers.VRTStack(slc_file_list_nc, outfile=vrt_file, subdataset="data")

    _, rows, cols = v.shape
    out_shape = compute_out_shape((rows, cols), strides=(strides["y"], strides["x"]))
    if not all(out_shape):
        pytest.skip(f"Output shape = {out_shape}")

    ms_planner = stack.MiniStackPlanner(
        file_list=slc_file_list_nc,
        dates=v.dates,
        is_compressed=[False] * len(slc_file_list_nc),
        output_folder=tmp_path / "sequential",
    )

    sequential.run_wrapped_phase_sequential(
        slc_vrt_file=vrt_file,
        ministack_planner=ms_planner,
        ministack_size=10,
        half_window=half_window,
        strides=strides,
        ps_mask_file=None,
        amp_mean_file=None,
        amp_dispersion_file=None,
        shp_method="rect",
        shp_alpha=None,
        shp_nslc=None,
    )


@pytest.mark.parametrize("ministack_size", [5, 9, 20])
def test_sequential_ministack_sizes(tmp_path, slc_file_list_nc, ministack_size):
    """Check various strides/windows/ministacks with a NetCDF input stack."""
    vrt_file = tmp_path / "slc_stack.vrt"
    # Make it not a round number to test
    vrt_stack = _readers.VRTStack(
        slc_file_list_nc[:21], outfile=vrt_file, subdataset="data"
    )
    _, rows, cols = vrt_stack.shape
    ms_planner = stack.MiniStackPlanner(
        file_list=vrt_stack.file_list,
        dates=vrt_stack.dates,
        is_compressed=[False] * len(vrt_stack.file_list),
        output_folder=tmp_path / "sequential",
    )

    # Record the warning, check after if it's thrown
    sequential.run_wrapped_phase_sequential(
        slc_vrt_file=vrt_file,
        ministack_planner=ms_planner,
        ministack_size=ministack_size,
        half_window={"x": cols // 2, "y": rows // 2},
        strides={"x": 1, "y": 1},
        ps_mask_file=None,
        amp_mean_file=None,
        amp_dispersion_file=None,
        shp_method="rect",
        shp_alpha=None,
        shp_nslc=None,
    )
