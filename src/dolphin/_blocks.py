"""Module for handling blocked/decimated input and output."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

# 1. iterate without overlap over output, decimated array
#   - Start with an offset, shrink the size, both by `half_window//strides`
#   - since we want every one wot be full, just skipping the incomplete blocks
# 2. map the current output pixels to their input locations (using strides)
# 3. pad the input block (using half window)


@dataclass(frozen=True)
class BlockIndices:
    """Class holding slices for 2D array access."""

    row_start: int
    row_stop: Optional[int]  # Can be None if we want slice(0, None)
    col_start: int
    col_stop: Optional[int]

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
        >>> list(iter_blocks((180, 250), (100, 100)))
        [BlockIndices(row_start=0, row_stop=100, col_start=0, col_stop=100), \
BlockIndices(row_start=0, row_stop=100, col_start=100, col_stop=200), \
BlockIndices(row_start=0, row_stop=100, col_start=200, col_stop=250), \
BlockIndices(row_start=100, row_stop=180, col_start=0, col_stop=100), \
BlockIndices(row_start=100, row_stop=180, col_start=100, col_stop=200), \
BlockIndices(row_start=100, row_stop=180, col_start=200, col_stop=250)]
        >>> list(map(tuple, iter_blocks((180, 250), (100, 100), overlaps=(10, 10))))
        [(slice(0, 100, None), slice(0, 100, None)), (slice(0, 100, None), \
slice(90, 190, None)), (slice(0, 100, None), slice(180, 250, None)), \
(slice(90, 180, None), slice(0, 100, None)), (slice(90, 180, None), \
slice(90, 190, None)), (slice(90, 180, None), slice(180, 250, None))]
    """
    total_rows, total_cols = arr_shape
    height, width = block_shape
    row_overlap, col_overlap = overlaps
    start_row_offset, start_col_offset = start_offsets
    last_row = total_rows - end_margin[0]
    last_col = total_cols - end_margin[1]

    if height is None:
        height = total_rows
    if width is None:
        width = total_cols

    # Check we're not moving backwards with the overlap:
    if row_overlap >= height and height != total_rows:
        raise ValueError(f"{row_overlap = } must be less than block height {height}")
    if col_overlap >= width and width != total_cols:
        raise ValueError(f"{col_overlap = } must be less than block width {width}")

    # Set up the iterating indices
    cur_row = start_row_offset
    cur_col = start_col_offset
    while cur_row < total_rows:
        while cur_col < total_cols:
            row_stop = min(cur_row + height, last_row)
            col_stop = min(cur_col + width, last_col)
            # yield (slice(cur_row, row_stop), slice(cur_col, col_stop))
            yield BlockIndices(cur_row, row_stop, cur_col, col_stop)

            cur_col += width
            if cur_col < last_col:  # dont bring back if already at edge
                cur_col -= col_overlap

        cur_row += height
        if cur_row < last_row:
            cur_row -= row_overlap
        cur_col = start_col_offset  # reset back to the starting offset


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
    col_start = in_block.col_start * col_strides
    row_stop = None if in_block.row_stop is None else in_block.row_stop * row_strides
    col_stop = None if in_block.col_stop is None else in_block.col_stop * col_strides
    return BlockIndices(row_start, row_stop, col_start, col_stop)


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
    """(row, col) of the full-res 2D image"""
    block_shape: tuple[int, int]
    """(row, col) size of each input block to operate on at one time"""
    strides: dict[str, int] = field(default_factory=lambda: {"x": 1, "y": 1})
    """Decimation/downsampling factor in y/row and x/column direction"""
    half_window: dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})
    """For multi-looking iterations, size of the (full-res) half window
    in y/row and x/column direction.
    Used to find `overlaps` between blocks and `start_offset`/`end_margin` for
    `iter_blocks`."""

    def __post_init__(self):
        self._half_rowcol = (self.half_window["y"], self.half_window["x"])
        # self._overlaps = (2 * self.half_window["y"], 2 * self.half_window["x"])
        # The output margins that we'll skip depend on the half window
        # Now that the `half_window` is in full-res coordinates, so the output
        # margin size is smaller
        out_row_margin = self._half_rowcol[0] // self.strides["y"]
        out_col_margin = self._half_rowcol[1] // self.strides["x"]
        self._out_margin = (out_row_margin, out_col_margin)

        # # The amount of extra padding the input blocks need depends on the
        # # window and the strides.
        # # The full-res slice is "automatically" padded by strides//2
        # # But if our window is even bigger, we need a little extra to ensure
        # # we have a full input window of data
        # self._extra_row_pad = max(self._half_rowcol[0] - self.strides["y"] // 2, 0)
        # self._extra_col_pad = max(self._half_rowcol[1] - self.strides["x"] // 2, 0)
        # self._in_padding = (self._extra_row_pad, self._extra_col_pad)
        self._in_padding = (
            self.strides["y"] * self._get_out_nodata_size("y"),
            self.strides["x"] * self._get_out_nodata_size("x"),
        )

    @property
    def output_shape(self):
        return compute_out_shape(self.arr_shape, self.strides)

    @property
    def out_block_shape(self):
        return compute_out_shape(self.block_shape, self.strides)

    def iter_outputs(self) -> Iterator[BlockIndices]:
        yield from iter_blocks(
            arr_shape=self.output_shape,
            block_shape=self.out_block_shape,
            # overlaps=self._overlaps,
            overlaps=(0, 0),  # We're not overlapping in the *output* grid
            start_offsets=self._out_margin,
            end_margin=self._out_margin,
        )

    def dilate_block(self, out_block: BlockIndices) -> BlockIndices:
        return dilate_block(out_block, strides=self.strides)

    def pad_block(self, unpadded_input_block: BlockIndices) -> BlockIndices:
        return pad_block(unpadded_input_block, margins=self._in_padding)

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
        row_nodata = self._get_out_nodata_size("y")
        col_nodata = self._get_out_nodata_size("x")
        # Extra check if we have no trimming to do: use slice(0, None)
        row_end = -row_nodata if row_nodata > 0 else None
        col_end = -col_nodata if col_nodata > 0 else None
        return BlockIndices(row_nodata, row_end, col_nodata, col_end)

    def _get_out_nodata_size(self, direction: str) -> int:
        nodata_size = round(self.half_window[direction] / self.strides[direction])
        return nodata_size

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
