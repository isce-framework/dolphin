from __future__ import annotations

from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from typing import Protocol, Sequence

from numpy.typing import ArrayLike
from tqdm.auto import tqdm

from dolphin import HalfWindow, Strides
from dolphin.utils import DummyProcessPoolExecutor

from ._blocks import BlockIndices, StridedBlockManager
from ._readers import StackReader
from ._writers import DatasetWriter

__all__ = ["BlockProcessor", "process_blocks"]


class BlockProcessor(Protocol):
    """Protocol for a block-wise processing function.

    Reads a block of data from each reader, processes it, and returns the result
    as an array-like object.
    """

    def __call__(
        self,
        readers: Sequence[StackReader],
        rows: slice,
        cols: slice,
    ) -> tuple[ArrayLike, slice, slice]: ...


def process_blocks(
    readers: Sequence[StackReader],
    writer: DatasetWriter,
    func: BlockProcessor,
    block_shape: tuple[int, int] = (512, 512),
    overlaps: tuple[int, int] = (0, 0),
    num_threads: int = 5,
):
    """Perform block-wise processing over blocks in `readers`, writing to `writer`.

    Parameters
    ----------
    readers : Sequence[StackReader]
        Sequence of input readers to read data from.
    writer : DatasetWriter
        Output writer to write data to.
    func : BlockProcessor
        Function to process each block.
    block_shape : tuple[int, int], optional
        Shape of each block to process.
    overlaps : tuple[int, int], optional
        Amount of overlap between blocks in (row, col) directions.
        By default (0, 0).
    num_threads : int, optional
        Number of threads to use, by default 5.

    """
    block_manager = StridedBlockManager(
        arr_shape=readers[0].shape[-2:],
        block_shape=block_shape,
        # Here we are not using the striding mechanism
        strides=Strides(1, 1),
        # The "half window" is how much overlap we read in
        half_window=HalfWindow(*overlaps),
    )
    total_blocks = sum(1 for _ in block_manager.iter_blocks())
    pbar = tqdm(total=total_blocks)

    def write_callback(fut: Future):
        data, rows, cols = fut.result()
        writer[..., rows, cols] = data
        pbar.update()

    Executor = ThreadPoolExecutor if num_threads > 1 else DummyProcessPoolExecutor
    futures: set[Future] = set()

    # Create a helper function to perform the trimming after processing
    def _run_and_trim(
        func,
        readers,
        out_idxs: BlockIndices,
        trim_idxs: BlockIndices,
        in_idxs: BlockIndices,
    ):
        in_rows, in_cols = in_idxs
        out_rows, out_cols = out_idxs
        trim_rows, trim_cols = trim_idxs
        out_data, _, _ = func(readers=readers, rows=in_rows, cols=in_cols)
        return out_data[..., trim_rows, trim_cols], out_rows, out_cols

    with Executor(num_threads) as exc:
        for out_idxs, trim_idxs, in_idxs, _, _ in block_manager.iter_blocks():
            future = exc.submit(
                _run_and_trim, func, readers, out_idxs, trim_idxs, in_idxs
            )
            future.add_done_callback(write_callback)
            futures.add(future)

        while futures:
            done, futures = wait(futures, timeout=1, return_when=FIRST_EXCEPTION)
            for future in done:
                e = future.exception()
                if e is not None:
                    raise e
