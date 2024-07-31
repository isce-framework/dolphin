from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from typing import Protocol, Sequence

from numpy.typing import ArrayLike
from tqdm.auto import tqdm

from dolphin.utils import DummyProcessPoolExecutor

from ._blocks import iter_blocks
from ._readers import StackReader
from ._writers import DatasetWriter

__all__ = ["BlockProcessor", "process_blocks"]


class BlockProcessor(Protocol):
    """Protocol for a block-wise processing function.

    Reads a block of data from each reader, processes it, and returns the result
    as an array-like object.
    """

    def __call__(
        self, readers: Sequence[StackReader], rows: slice, cols: slice
    ) -> tuple[ArrayLike, slice, slice]: ...


def process_blocks(
    readers: Sequence[StackReader],
    writer: DatasetWriter,
    func: BlockProcessor,
    block_shape: tuple[int, int] = (512, 512),
    num_threads: int = 5,
):
    """Perform block-wise processing over blocks in `readers`, writing to `writer`.

    Used to read and process a stack of rasters in parallel, setting up a queue
    of results for the `writer` to save.

    Note that the parallelism happens using a `ThreadPoolExecutor`, so `func` should
    be a function which releases the GIL during computation (e.g. using numpy).
    """
    shape = readers[0].shape[-2:]
    slices = list(iter_blocks(shape, block_shape=block_shape))

    pbar = tqdm(total=len(slices))

    # Define the callback to write the result to an output DatasetWrite
    def write_callback(fut: Future):
        data, rows, cols = fut.result()
        writer[..., rows, cols] = data
        pbar.update()

    Executor = ThreadPoolExecutor if num_threads > 1 else DummyProcessPoolExecutor
    futures: set[Future] = set()
    with Executor(num_threads) as exc:
        for rows, cols in slices:
            future = exc.submit(func, readers=readers, rows=rows, cols=cols)
            future.add_done_callback(write_callback)
            futures.add(future)

        while futures:
            done, futures = wait(futures, timeout=1, return_when=FIRST_EXCEPTION)
            for future in done:
                e = future.exception()
                if e is not None:
                    raise e
