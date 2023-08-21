import numpy as np
import pytest

from dolphin._blocks import (
    BlockIndices,
    BlockManager,
    compute_out_shape,
    dilate_block,
    iter_blocks,
)


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

    outs, trimmed, ins, in_no_pads = zip(*list(bm.iter_blocks()))
    assert outs == ins
    assert outs == in_no_pads
    assert all((rs, cs) == (slice(0, None), slice(0, None)) for (rs, cs) in trimmed)


def test_block_manager_no_trim():
    # Check no stride version
    bm = BlockManager(
        (5, 10), (100, 100), strides={"x": 2, "y": 3}, half_window={"x": 1, "y": 1}
    )

    trimmed_rows, trimmed_cols = bm.get_trimmed_block()
    assert trimmed_rows == trimmed_cols == slice(0, None)
