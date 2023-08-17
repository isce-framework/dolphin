"""Module for handling blocked/decimated input and output."""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Iterator, Optional

# from numpy.typing import ArrayLike

# from dolphin._types import Filename

# 1. iterate without overlap over output, decimated array
#   - Start with an offset, shrink the size, both by `half_window//strides`
#   - since we want every one wot be full, just skipping the incomplete blocks
# 2. map the current output pixels to their input locations (using strides)
# 3. pad the input block (using half window)


@dataclass
class BlockIndices:
    """Class holding slices for 2D array access."""

    row_start: int
    row_stop: int
    col_start: int
    col_stop: int

    def __iter__(self):
        return iter(
            (slice(self.row_start, self.row_stop), slice(self.col_start, self.col_stop))
        )


def iter_blocks(
    arr_shape: tuple[int, int],
    block_shape: tuple[int, int],
    overlaps: tuple[int, int] = (0, 0),
    start_offsets: tuple[int, int] = (0, 0),
    end_margin: tuple[int, int] = (0, 0),
) -> Iterator[BlockIndices]:
    """Create a generator to get indexes for accessing blocks of a raster.

    Parameters
    ----------
    arr_shape : tuple[int, int]
        (num_rows, num_cols), full size of array to access
    block_shape : tuple[int, int]
        (height, width), size of blocks to load
    overlaps : tuple[int, int], default = (0, 0)
        (row_overlap, col_overlap), number of pixels to re-include from
        the previous block after sliding
    start_offsets : tuple[int, int], default = (0, 0)
        Offsets from top left to start reading from
    end_margin : tuple[int, int], default = (0, 0)
        Margin to avoid at the bottom/right of array

    Yields
    ------
    BlockIndices
        Iterator of BlockIndices, which can be unpacked into
        (slice(row_start, row_stop), slice(col_start, col_stop))

    Examples
    --------
        >>> list(_slice_iterator((180, 250), (100, 100)))
        [(slice(0, 100, None), slice(0, 100, None)), (slice(0, 100, None), \
slice(100, 200, None)), (slice(0, 100, None), slice(200, 250, None)), \
(slice(100, 180, None), slice(0, 100, None)), (slice(100, 180, None), \
slice(100, 200, None)), (slice(100, 180, None), slice(200, 250, None))]
        >>> list(_slice_iterator((180, 250), (100, 100), overlaps=(10, 10)))
        [(slice(0, 100, None), slice(0, 100, None)), (slice(0, 100, None), \
slice(90, 190, None)), (slice(0, 100, None), slice(180, 250, None)), \
(slice(90, 180, None), slice(0, 100, None)), (slice(90, 180, None), \
slice(90, 190, None)), (slice(90, 180, None), slice(180, 250, None))]
    """
    rows, cols = arr_shape
    height, width = block_shape
    row_overlap, col_overlap = overlaps
    row_off, col_off = start_offsets
    last_row = rows - end_margin[0]
    last_col = cols - end_margin[1]

    if height is None:
        height = rows
    if width is None:
        width = cols

    # Check we're not moving backwards with the overlap:
    if row_overlap >= height and height != rows:
        raise ValueError(f"{row_overlap = } must be less than block height {height}")
    if col_overlap >= width and width != cols:
        raise ValueError(f"{col_overlap = } must be less than block width {width}")

    while row_off < rows:
        while col_off < cols:
            row_stop = min(row_off + height, last_row)
            col_stop = min(col_off + width, last_col)
            # yield (slice(row_off, row_stop), slice(col_off, col_stop))
            yield BlockIndices(row_off, row_stop, col_off, col_stop)

            col_off += width
            if col_off < last_col:  # dont bring back if already at edge
                col_off -= col_overlap

        row_off += height
        if row_off < last_row:
            row_off -= row_overlap
        col_off = 0


def dilate_block(
    in_block: BlockIndices,
    strides: dict[str, int],
) -> BlockIndices:
    """Dilate the slices in `BlockIndices` for the larger array.

    Assumes in in_block is the smaller one which has been made
    by taking `strides` from the larger block.

    Parameters
    ----------
    in_block : BlockIndices
        Slices for an output array.
    strides : dict[str, int]
        Decimation factor in x and y which was used for `in_block`
        {'x': col_strides, 'y': row_strides}

    Returns
    -------
    BlockIndices
        Output slices for larger array
    """
    block = copy(in_block)
    row_strides, col_strides = strides["y"], strides["x"]
    block.row_start = block.row_start * row_strides - row_strides // 2
    block.row_stop = block.row_stop * row_strides + row_strides // 2
    block.col_start = block.col_start * col_strides - col_strides // 2
    block.col_stop = block.col_stop * col_strides + col_strides // 2
    return block


# def load_padded_block(
#     input_raster: ArrayLike,
#     block: BlockIndices,
# ) -> np.ndarray:
#     # Do some smart padding in order to handle negative indices
#     # and/or indices that are greater than the corresponding input dimension
#     ...


# def foo(input_block: np.ndarray) -> np.ndarray:
#     # Does something interesting and returns a decimated block
#     ...


# def do_the_thing(
#     input_raster: ArrayLike,
#     strides: dict[str, int],
#     block_shape: tuple[int, int],
# ) -> ArrayLike:
#     input_shape = input_raster.shape
#     output_shape = compute_out_shape(input_shape, strides)
#     output = make_my_output_raster(output_shape)
#     for output_block in iter_blocks(output_shape):
#         input_block = dilate_block(output_block, strides)
#         input_block_data = load_padded_block(input_raster, input_block)
#         output_block_data = foo(input_block_data)
#         write_block(output, output_block, output_block_data)


def compute_out_shape(
    shape: tuple[int, int], strides: dict[str, int]
) -> tuple[int, int]:
    """Calculate the output size for an input `shape` and row/col `strides`.

    Parameters
    ----------
    shape : tuple[int, int]
        Input size: (rows, cols)
    strides : dict[str, int]
        {"x": x strides, "y": y strides}

    Returns
    -------
    out_shape : tuple[int, int]
        Size of output after striding

    Notes
    -----
    If there is not a full window (of size `strides`), the end
    will get cut off rather than padded with a partial one.
    This should match the output size of `[dolphin.utils.take_looks][]`.

    As a 1D example, in array of size 6 with `strides`=3 along this dim,
    we could expect the pixels to be centered on indexes
    `[1, 4]`.

        [ 0  1  2   3  4  5]

    So the output size would be 2, since we have 2 full windows.
    If the array size was 7 or 8, we would have 2 full windows and 1 partial,
    so the output size would still be 2.
    """
    rows, cols = shape
    rs, cs = strides["y"], strides["x"]
    return (rows // rs, cols // cs)


@dataclass
class BlockManager:
    """Class to handle slicing/trimming overlapping blocks with strides."""

    arr_shape: tuple[int, int]
    """(row, col) of the total 2D image"""
    block_shape: tuple[int, int]
    """(row, col) size of each block to load at a time"""
    strides: dict[str, int] = field(default_factory=lambda: {"x": 1, "y": 1})
    """Decimation/downsampling factor in y/row and x/column direction"""
    overlaps: Optional[tuple[int, int]] = (0, 0)
    """Option to manually specify the (row, col) overlap between blocks"""
    half_window: Optional[dict[str, int]] = None
    """Size of the Decimation/downsampling factor in y/row and x/column direction.
    Determines the `overlaps` if specified."""

    @property
    def output_shape(self):
        return compute_out_shape(self.arr_shape, self.strides)

    @property
    def output_slices(self):
        return iter_blocks(
            arr_shape=self.output_shape,
            block_shape=self.block_shape,
            overlaps=self.overlaps,
        )
