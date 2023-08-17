from dolphin._blocks import (  # , dilate_block, iter_blocks
    BlockIndices,
    compute_out_shape,
)


def test_block_indices_create():
    b = BlockIndices(0, 3, 1, 5)
    assert b.row_start == 0
    assert b.row_stop == 3
    assert b.row_start == 1
    assert b.row_start == 5
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
