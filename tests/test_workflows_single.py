import pytest

from dolphin import stack
from dolphin.io import _readers
from dolphin.phase_link import simulate
from dolphin.utils import gpu_is_available
from dolphin.workflows import single

GPU_AVAILABLE = gpu_is_available()
simulate._seed(1234)


@pytest.mark.parametrize("write_extra", [False, True])
def test_sequential_gtiff(tmp_path, slc_file_list, write_extra: bool):
    """Run through the sequential estimation with a GeoTIFF stack."""
    vrt_file = tmp_path / "slc_stack.vrt"
    files = slc_file_list[:3]
    vrt_stack = _readers.VRTStack(files, outfile=vrt_file)
    is_compressed = [False] * len(files)
    ministack = stack.MiniStackInfo(
        file_list=vrt_stack.file_list,
        dates=vrt_stack.dates,
        is_compressed=is_compressed,
    )

    hy, hx = 1, 2
    half_window = {"x": hx, "y": hy}
    strides = {"x": 1, "y": 1}
    output_folder = tmp_path / "single"
    single.run_wrapped_phase_single(
        vrt_stack=vrt_stack,
        ministack=ministack,
        output_folder=output_folder,
        half_window=half_window,
        strides=strides,
        shp_method="rect",
        write_crlb=write_extra,
        write_closure_phase=write_extra,
    )

    assert output_folder.exists()
    # Check that all the expected outputs are there
    assert len(list(output_folder.glob("2*.slc.tif"))) == 3
    assert len(list(output_folder.glob("compressed_*tif"))) == 1
    assert len(list(output_folder.glob("temporal_coherence*tif"))) == 1


def test_sequential_gtiff_interleaved_compressed(tmp_path, slc_file_list):
    """A pre-existing compressed SLC need not be a contiguous prefix.

    Regression test: `run_wrapped_phase_single` used to assume all compressed
    SLCs occupied a contiguous prefix of the ministack (via a scalar
    `first_real_slc_idx` boundary). Mark the *middle* input as an
    already-existing compressed SLC to exercise the mask-based real/compressed
    selection end-to-end (output counts/names, and the CRLB write path).
    """
    vrt_file = tmp_path / "slc_stack.vrt"
    files = slc_file_list[:3]
    vrt_stack = _readers.VRTStack(files, outfile=vrt_file)
    # Middle entry marked as an already-existing compressed SLC -- not a prefix.
    is_compressed = [False, True, False]
    ministack = stack.MiniStackInfo(
        file_list=vrt_stack.file_list,
        dates=vrt_stack.dates,
        is_compressed=is_compressed,
    )
    assert ministack.is_compressed == is_compressed
    assert ministack.last_compressed_slc_idx == 1

    hy, hx = 1, 2
    half_window = {"x": hx, "y": hy}
    strides = {"x": 1, "y": 1}
    output_folder = tmp_path / "single_interleaved"
    single.run_wrapped_phase_single(
        vrt_stack=vrt_stack,
        ministack=ministack,
        output_folder=output_folder,
        half_window=half_window,
        strides=strides,
        shp_method="rect",
        write_crlb=True,
        write_closure_phase=True,
    )

    assert output_folder.exists()
    # Only the 2 real dates get their own phase-linked/CRLB output files --
    # the pre-existing compressed SLC (in the middle) does not, regardless
    # of its position.
    assert len(list(output_folder.glob("2*.slc.tif"))) == 2
    assert len(list((output_folder / "crlb").glob("crlb*.tif"))) == 2
    # The newly-created compressed SLC excludes the old one from its averaging.
    assert len(list(output_folder.glob("compressed_*tif"))) == 1
