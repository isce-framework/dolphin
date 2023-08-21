"""Module for handling blocked/decimated input and output."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

# 1. iterate without overlap over output, decimated array
#   - Start with an offset, shrink the size, both by `half_window//strides`
#   - since we want every one wot be full, just skipping the incomplete blocks
# 2. map the current output pixels to their input locations (using strides)
# 3. pad the input block (using half window)


@dataclass(frozen=True)
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
    """Grow slices in `BlockIndices` to fit a larger array.

    This is to undo the "stride"/decimation index changes, so
     we can go from smaller, strided array indices to the original.

    Assumes in in_block is the smaller one which has been made
    by taking `strides` from the larger block.

    Parameters
    ----------
    in_block : BlockIndices
        Slices for an smaller, strided array.
    strides : dict[str, int]
        Decimation factor in x and y which was used for `in_block`
        {'x': col_strides, 'y': row_strides}

    Returns
    -------
    BlockIndices
        Output slices for larger array
    """
    row_strides, col_strides = strides["y"], strides["x"]
    row_start = in_block.row_start * row_strides
    row_stop = in_block.row_stop * row_strides
    col_start = in_block.col_start * col_strides
    col_stop = in_block.col_stop * col_strides
    return BlockIndices(row_start, row_stop, col_start, col_stop)

    # block.row_start = block.row_start * row_strides - row_strides // 2
    # block.row_stop = block.row_stop * row_strides + row_strides // 2
    # block.col_start = block.col_start * col_strides - col_strides // 2
    # block.col_stop = block.col_stop * col_strides + col_strides // 2


def pad_block(in_block: BlockIndices, margins: tuple[int, int]) -> BlockIndices:
    """Pad `in_block` by the (row_margin, col_margin) pixels in `margins`.

    Will clip the row/column slice starts to be >= 0.

    Parameters
    ----------
    in_block : BlockIndices
        Slices original array block.
    margins : dict[int, int]
        Number of pixels to extend `in_block` in the (row, col) directions.
        The margins are subtracted from the beginning and added to the end,
        so the block size grows by (2 * row_margin, 2 * col_margin)

    Returns
    -------
    BlockIndices
        Output slices for larger block
    """
    r_margin, c_margin = margins
    r_slice, c_slice = in_block
    return BlockIndices(
        max(r_slice.start - r_margin, 0),
        r_slice.stop + r_margin,
        max(c_slice.start - c_margin, 0),
        c_slice.stop + c_margin,
    )


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
    half_window: dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})
    """For multi-looking iterations, size of the half window in y/row
    and x/column direction. Used to find `overlaps` between blocks and
    `start_offset`/`end_margin` for `iter_blocks."""

    def __post_init__(self):
        self._half_rowcol = (self.half_window["y"], self.half_window["x"])
        self._overlaps = (2 * self.half_window["y"], 2 * self.half_window["x"])

    @property
    def output_shape(self):
        return compute_out_shape(self.arr_shape, self.strides)

    def iter_outputs(self) -> Iterator[BlockIndices]:
        yield from iter_blocks(
            arr_shape=self.output_shape,
            block_shape=self.block_shape,
            overlaps=self._overlaps,
            start_offsets=self._half_rowcol,
            end_margin=self._half_rowcol,
        )

    def dilate_block(self, out_block: BlockIndices) -> BlockIndices:
        return dilate_block(out_block, strides=self.strides)

    def pad_block(self, unpadded_input_block: BlockIndices) -> BlockIndices:
        return pad_block(unpadded_input_block, margins=self._half_rowcol)

    def get_trimmed_block(self) -> BlockIndices:
        """Compute the slices which trim output nodata values.

        When the BlockIndex gets dilated (using `strides`) and padded (using
        `half_window`), the result will have nodata around the edges.
        The size of the nodata pixels in the full-res block is just
            (half_window['y'], half_window['x'])
        In the output (strided) coordinates, the number of nodata pixels is
        shrunk by how many strides are taken.

        Note that this is independent of which block we're on; the number of
        nodata pixels on the border is always the same.
        """
        half_row, half_col = self._half_rowcol
        row_strides, col_strides = self.strides["y"], self.strides["x"]
        row_nodata_size = round(half_row / row_strides)
        col_nodata_size = round(half_col / col_strides)

        return BlockIndices(
            row_nodata_size, -row_nodata_size, col_nodata_size, -col_nodata_size
        )

    def iter_blocks(
        self,
    ) -> Iterator[tuple[BlockIndices, BlockIndices, BlockIndices, BlockIndices]]:
        """Iterate over the input/output blocks.

        Yields
        ------
        output_block : BlockIndices
            The current slices for the output raster
        trimmed_block : BlockIndices
            The slices to use on a processed output block to remove nodata border pixels.
            These may be relative (e.g. slice(1, -1)), not absolute like `output_block`.
        input_block : BlockIndices
            Slices used to load the full-res input data
        input_no_padding : BlockIndices
            Slices which point to the position within the full-res data without padding
        """
        trimmed_block = self.get_trimmed_block()
        for out_block in self.iter_outputs():
            input_no_padding = self.dilate_block(out_block)
            input_block = self.pad_block(input_no_padding)
            yield (out_block, trimmed_block, input_block, input_no_padding)
