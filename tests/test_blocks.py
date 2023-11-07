import numpy as np
import pytest

from dolphin._blocks import (
    BlockIndices,
    BlockManager,
    dilate_block,
    get_slice_length,
    iter_blocks,
)
from dolphin.io import compute_out_shape
from dolphin.utils import upsample_nearest


def test_block_indices_create():
    b = BlockIndices(0, 3, 1, 5)
    assert b.row_start == 0
    assert b.row_stop == 3
    assert b.col_start == 1
    assert b.col_stop == 5
    assert tuple(b) == (slice(0, 3, None), slice(1, 5, None))


def test_compute_out_size():
    strides = {"x": 1, "y": 1}
    assert (6, 6) == compute_out_shape((6, 6), strides)

    strides = {"x": 3, "y": 3}
    assert (2, 2) == compute_out_shape((6, 6), strides)

    # 1,2 more in each direction shouldn't change it
    assert (2, 2) == compute_out_shape((7, 7), strides)
    assert (2, 2) == compute_out_shape((8, 8), strides)

    # 1,2 fewer should bump down to 1
    assert (1, 1) == compute_out_shape((5, 5), strides)
    assert (1, 1) == compute_out_shape((4, 4), strides)


def test_iter_blocks():
    out_blocks = iter_blocks((3, 5), (2, 2))
    assert hasattr(out_blocks, "__iter__")
    assert list(out_blocks) == [
        BlockIndices(0, 2, 0, 2),
        BlockIndices(0, 2, 2, 4),
        BlockIndices(0, 2, 4, 5),
        BlockIndices(2, 3, 0, 2),
        BlockIndices(2, 3, 2, 4),
        BlockIndices(2, 3, 4, 5),
    ]


@pytest.mark.parametrize("block_shape", [(5, 5), (10, 20), (13, 27)])
def test_iter_blocks_coverage(block_shape):
    shape = (100, 200)
    check_out = np.zeros(shape)

    for rs, cs in iter_blocks(shape, block_shape):
        check_out[rs, cs] += 1

    # Everywhere should have been touched once by the iteration
    assert np.all(check_out == 1)


def test_iter_blocks_overlap():
    # Block size that is a multiple of the raster size
    shape = (100, 200)
    check_out = np.zeros(shape)

    for rs, cs in iter_blocks(shape, (30, 30), overlaps=(5, 5)):
        check_out[rs, cs] += 1

    # Everywhere should have been touched *at least* once by the iteration
    assert np.all(check_out >= 1)


def test_iter_blocks_offset_margin():
    # Block size that is a multiple of the raster size
    shape = (100, 200)
    check_out = np.zeros(shape)

    for rs, cs in iter_blocks(shape, (30, 30), start_offsets=(2, 3)):
        check_out[rs, cs] += 1

    # Everywhere should have been touched once by the iteration
    assert np.all(check_out[2:, 3:] == 1)
    # offset should still be 0
    assert np.all(check_out[:2, :3] == 0)

    check_out[:] = 0
    for rs, cs in iter_blocks(shape, (30, 30), end_margin=(4, 5)):
        check_out[rs, cs] += 1
    # Everywhere except the end should be 1
    assert np.all(check_out[:4, :5] == 1)
    assert np.all(check_out[-4:, -5:] == 0)


def test_nonzero_block_size_with_margin():
    shape = (33, 67)
    block_shape = (5, 5)
    offset = margin = (0, 1)
    check_out = np.zeros(shape)
    for rs, cs in iter_blocks(
        shape, block_shape, start_offsets=offset, end_margin=margin
    ):
        assert get_slice_length(rs) > 0
        assert get_slice_length(cs) > 0
        check_out[rs, cs] += 1
    assert np.all(check_out[:, 1:-1] == 1)


def test_dilate_block():
    # Iterate over the output, decimated raster
    out_blocks = list(iter_blocks((3, 5), (2, 2)))
    assert out_blocks == [
        BlockIndices(row_start=0, row_stop=2, col_start=0, col_stop=2),
        BlockIndices(row_start=0, row_stop=2, col_start=2, col_stop=4),
        BlockIndices(row_start=0, row_stop=2, col_start=4, col_stop=5),
        BlockIndices(row_start=2, row_stop=3, col_start=0, col_stop=2),
        BlockIndices(row_start=2, row_stop=3, col_start=2, col_stop=4),
        BlockIndices(row_start=2, row_stop=3, col_start=4, col_stop=5),
    ]
    # Dilate each out block
    strides = {"x": 1, "y": 1}
    in_blocks = [dilate_block(b, strides=strides) for b in out_blocks]
    assert in_blocks == out_blocks

    strides = {"x": 3, "y": 1}
    in_blocks = [dilate_block(b, strides=strides) for b in out_blocks]
    assert in_blocks == [
        BlockIndices(row_start=0, row_stop=2, col_start=0, col_stop=6),
        BlockIndices(row_start=0, row_stop=2, col_start=6, col_stop=12),
        BlockIndices(row_start=0, row_stop=2, col_start=12, col_stop=15),
        BlockIndices(row_start=2, row_stop=3, col_start=0, col_stop=6),
        BlockIndices(row_start=2, row_stop=3, col_start=6, col_stop=12),
        BlockIndices(row_start=2, row_stop=3, col_start=12, col_stop=15),
    ]


def test_dilate_block_is_multiple():
    # Iterate over the output, decimated raster
    b = BlockIndices(0, 3, 0, 5)
    strides = {"x": 3, "y": 3}
    # the sizes should be multiples of 3
    b_big = dilate_block(b, strides=strides)
    check_arr = np.ones((100, 100))[tuple(b_big)]
    assert check_arr.shape == (3 * 3, 5 * 3)


def test_block_manager():
    # Check no stride version
    bm = BlockManager((5, 5), (2, 3))
    assert list(bm.iter_outputs()) == [
        BlockIndices(row_start=0, row_stop=2, col_start=0, col_stop=3),
        BlockIndices(row_start=0, row_stop=2, col_start=3, col_stop=5),
        BlockIndices(row_start=2, row_stop=4, col_start=0, col_stop=3),
        BlockIndices(row_start=2, row_stop=4, col_start=3, col_stop=5),
        BlockIndices(row_start=4, row_stop=5, col_start=0, col_stop=3),
        BlockIndices(row_start=4, row_stop=5, col_start=3, col_stop=5),
    ]

    outs, trimming, ins, in_no_pads = zip(*list(bm.iter_blocks()))
    assert outs == ins
    assert outs == in_no_pads
    assert all((rs, cs) == (slice(0, None), slice(0, None)) for (rs, cs) in trimming)


def test_block_manager_no_trim():
    # Check no stride version
    bm = BlockManager(
        (5, 10), (100, 100), strides={"x": 3, "y": 3}, half_window={"x": 1, "y": 1}
    )

    trimming_rows, trimming_cols = bm.get_trimming_block()
    assert trimming_rows == trimming_cols == slice(0, None)


def test_block_manager_iter_outputs():
    nrows, ncols = (100, 200)
    xs, ys = 3, 3  # strides
    hx, hy = 3, 1  # half window
    bm = BlockManager(
        arr_shape=(nrows, ncols),
        block_shape=(17, 27),
        strides={"x": xs, "y": xs},
        half_window={"x": hx, "y": hy},
    )

    out_row_margin = hy // ys
    out_col_margin = hx // ys
    for row_slice, col_slice in bm.iter_outputs():
        assert row_slice.start >= out_row_margin
        assert col_slice.start >= out_col_margin
        assert row_slice.stop < nrows - out_row_margin
        assert col_slice.stop < ncols - out_col_margin


def _fake_process(in_arr, strides, half_window):
    """Dummy processing which has same nodata pattern as `phase_link.run_mle`."""
    nrows, ncols = in_arr.shape
    row_half, col_half = half_window["y"], half_window["x"]
    rs, cs = strides["y"], strides["x"]
    out_nrows, out_ncols = compute_out_shape(in_arr.shape, strides=strides)
    out = np.ones((out_nrows, out_ncols))
    for out_r in range(out_nrows):
        for out_c in range(out_ncols):
            # the input indexes computed from the output idx and strides
            # Note: weirdly, moving these out of the loop causes r_start
            # to be 0 in some cases...
            in_r_start = rs // 2
            in_c_start = cs // 2
            in_r = in_r_start + out_r * rs
            in_c = in_c_start + out_c * cs

            # Check if the window is completely in bounds
            if in_r + row_half >= nrows or in_r - row_half < 0:
                out[out_r, out_c] = np.nan
            if in_c + col_half >= ncols or in_c - col_half < 0:
                out[out_r, out_c] = np.nan
    return out


def fake_process_blocks(in_shape, half_window, strides, block_shape):
    out_shape = compute_out_shape(in_shape, strides)

    # full_res_data = np.random.randn(*in_shape) + 1j * np.random.randn(*in_shape)
    # full_res_data = full_res_data.astype(np.complex64)
    rng = np.random.default_rng()
    full_res_data = rng.normal(size=in_shape).astype("float32")
    out_arr = np.zeros(out_shape, dtype=full_res_data.dtype)
    out_full_res = np.zeros_like(full_res_data)
    counts = np.zeros(out_shape, dtype=int)

    bm = BlockManager(
        in_shape, block_shape=block_shape, strides=strides, half_window=half_window
    )
    for (
        (out_rows, out_cols),
        (trimming_rows, trimming_cols),
        (in_rows, in_cols),
        (in_no_pad_rows, in_no_pad_cols),
    ) in bm.iter_blocks():
        in_data = full_res_data[in_rows, in_cols]
        out_data = _fake_process(in_data, strides, half_window)

        data_trimmed = out_data[trimming_rows, trimming_cols]
        assert np.all(~np.isnan(data_trimmed))
        assert get_slice_length(out_rows) == data_trimmed.shape[0]
        assert get_slice_length(out_cols) == data_trimmed.shape[1]

        out_arr[out_rows, out_cols] = data_trimmed
        counts[out_rows, out_cols] += 1

        out_full_nrows = get_slice_length(in_no_pad_rows)
        out_full_ncols = get_slice_length(in_no_pad_cols)
        out_upsampled = upsample_nearest(data_trimmed, (out_full_nrows, out_full_ncols))
        out_full_res[in_no_pad_rows, in_no_pad_cols] = out_upsampled

    # Now check the inner part, away from the expected border of zeros
    out_row_margin, out_col_margin = bm._out_margin
    inner = (
        slice(out_row_margin, -out_row_margin),
        slice(out_col_margin, -out_col_margin),
    )
    assert not np.any(out_arr[inner] == 0)
    assert np.all(counts[inner] == 1)


@pytest.mark.parametrize("in_shape", [(100, 200), (101, 201)])
@pytest.mark.parametrize(
    "half_window", [{"x": 1, "y": 1}, {"x": 3, "y": 1}, {"x": 3, "y": 3}]
)
@pytest.mark.parametrize(
    "strides", [{"x": 1, "y": 1}, {"x": 3, "y": 1}, {"x": 3, "y": 3}]
)
@pytest.mark.parametrize("block_shape", [(15, 15), (20, 30), (17, 27)])
def test_block_manager_fake_process(in_shape, half_window, strides, block_shape):
    fake_process_blocks(in_shape, half_window, strides, block_shape)


def test_failing_block_params():
    # Extra test from real-data params
    half_window, strides = {"x": 11, "y": 5}, {"x": 6, "y": 3}
    in_shape, block_shape = (2050, 4050), (1024, 1024)
    fake_process_blocks(in_shape, half_window, strides, block_shape)
