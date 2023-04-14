import abc
import os
import time
from collections.abc import Callable
from concurrent.futures import Executor, Future
from queue import Empty, Full, Queue
from threading import Event, Thread
from threading import enumerate as threading_enumerate
from typing import Any, Optional

from dolphin._log import get_log

logger = get_log(__name__)

_DEFAULT_TIMEOUT = 0.5


def is_main_thread_active() -> bool:
    """Check if the main thread is still active.

    Used to check if the writing thread should exit if there was
    some exception in the main thread.

    Source: https://stackoverflow.com/a/23443397/4174466
    """
    return any((i.name == "MainThread") and i.is_alive() for i in threading_enumerate())


class BackgroundWorker(abc.ABC):
    """Base class for doing work in a background thread.

    After instantiating an object, a client sends it work with the `queue_work`
    method and retrieves the result with the `get_result` method (hopefully
    after doing something else useful in between).  The worker remains active
    until `notify_finished` is called.  Subclasses must define the `process`
    method.

    Parameters
    ----------
    num_work_queue : int
        Max number of work items to queue before blocking, <= 0 for unbounded.
    num_results_queue : int
        Max number of results to generate before blocking, <= 0 for unbounded.
    store_results : bool
        Whether to store return values of `process` method.  If True then
        `get_result` must be called once for every `queue_work` call.
    timeout : float
        Interval in seconds used to check for finished notification once work
        queue is empty.

    Notes
    -----
    The usual caveats about Python threading apply.  It's typically a poor
    choice for concurrency unless the global interpreter lock (GIL) has been
    released, which can happen in IO calls and compiled extensions.
    """

    def __init__(
        self,
        num_work_queue=0,
        num_results_queue=0,
        store_results=True,
        drop_unfinished_results=False,
        timeout=_DEFAULT_TIMEOUT,
        name="BackgroundWorker",
    ):
        self.name = name
        self.store_results = store_results
        self.timeout = timeout
        self._finished_event = Event()
        self._work_queue = Queue(num_work_queue)
        if self.store_results:
            self._results_queue = Queue(num_results_queue)
        self._thread = Thread(target=self._consume_work_queue, name=name)
        self._thread.start()
        self._drop_unfinished_results = drop_unfinished_results

    def _consume_work_queue(self):
        while True:
            if not is_main_thread_active():
                break

            logger.debug(f"{self.name} getting work")
            if self._finished_event.is_set():
                do_exit = self._drop_unfinished_results or (
                    self._work_queue.unfinished_tasks == 0
                )
                if do_exit:
                    break
                else:
                    # Keep going even if finished event is set
                    logger.debug(
                        f"{self.name} Finished... but waiting for work queue to empty,"
                        f" {self._work_queue.qsize()} items left,"
                        f" {self._work_queue.unfinished_tasks} unfinished"
                    )
            try:
                args, kw = self._work_queue.get(timeout=self.timeout)
                logger.debug(f"{self.name} processing")
                result = self.process(*args, **kw)
                self._work_queue.task_done()
                # Notify the queue that processing is done
                logger.debug(f"{self.name} got result")
            except Empty:
                logger.debug(f"{self.name} timed out, checking if done")
                continue

            if self.store_results:
                logger.debug(f"{self.name} saving result in queue")
                while True:
                    try:
                        self._results_queue.put(result, timeout=2)
                        break
                    except Full:
                        logger.debug(f"{self.name} result queue full, waiting...")
                        continue

    @abc.abstractmethod
    def process(self, *args, **kw):
        """User-defined task to operate in background thread."""
        pass

    def queue_work(self, *args, **kw):
        """Add a job to the work queue to be executed.

        Blocks if work queue is full.
        Same input interface as `process`.
        """
        if self._finished_event.is_set():
            raise RuntimeError("Attempted to queue_work after notify_finished!")
        self._work_queue.put((args, kw))

    def get_result(self):
        """Get the least-recent value from the result queue.

        Blocks until a result is available.
        Same output interface as `process`.
        """
        while True:
            try:
                result = self._results_queue.get(timeout=self.timeout)
                self._results_queue.task_done()
                break
            except Empty:
                logger.debug(f"{self.name} get_result timed out, checking if done")
                if self._finished_event.is_set():
                    raise RuntimeError("Attempted to get_result after notify_finished!")
                continue
        return result

    def notify_finished(self, timeout=None):
        """Signal that all work has finished.

        Indicate that no more work will be added to the queue, and block until
        all work has been processed.
        If `store_results=True` also block until all results have been retrieved.
        """
        self._finished_event.set()
        if self.store_results and not self._drop_unfinished_results:
            self._results_queue.join()
        self._thread.join(timeout)

    def __del__(self):
        self.notify_finished()


class BackgroundWriter(BackgroundWorker):
    """Base class for writing data in a background thread.

    After instantiating an object, a client sends it data with the `queue_write`
    method.  The writer remains active until `notify_finished` is called.
    Subclasses must define the `write` method.

    Parameters
    ----------
    nq : int
        Number of write jobs that can be queued before blocking, <= 0 for
        unbounded.  Default is 1.
    timeout : float
        Interval in seconds used to check for finished notification once write
        queue is empty.
    """

    def __init__(self, nq=1, timeout=_DEFAULT_TIMEOUT, **kwargs):
        super().__init__(
            num_work_queue=nq,
            store_results=False,
            timeout=timeout,
            **kwargs,
        )

    # rename queue_work -> queue_write
    def queue_write(self, *args, **kw):
        """Add data to the queue to be written.

        Blocks if write queue is full.
        Same interfaces as `write`.
        """
        self.queue_work(*args, **kw)

    # rename process -> write
    def process(self, *args, **kw):
        self.write(*args, **kw)

    @abc.abstractmethod
    def write(self, *args, **kw):
        """User-defined method for writing data."""
        pass


class BackgroundReader(BackgroundWorker):
    """Base class for reading data in a background thread (pre-fetching).

    After instantiating an object, a client sends it data selection parameters
    (slices, indices, etc.) via the `queue_read` method and retrives the result
    with the `get_data` method.  In order to get useful concurrency, that
    usually means you'll want to queue the read for the next data block before
    starting work on the current block.  The reader remains active until
    `notify_finished` is called and all blocks have been retrieved.  Subclasses
    must define the `read` method.

    Parameters
    ----------
    nq : int
        Number of read results that can be stored before blocking, <= 0 for
        unbounded.  Default is 1.
    timeout : float
        Interval in seconds used to check for finished notification once write
        queue is empty.
    """

    def __init__(self, nq=1, timeout=_DEFAULT_TIMEOUT, **kwargs):
        super().__init__(
            num_results_queue=nq,
            timeout=timeout,
            store_results=True,
            # If we're reading data, we don't care about the result queue
            drop_unfinished_results=True,
            **kwargs,
        )

    # rename queue_work -> queue_read
    def queue_read(self, *args, **kw):
        """Add selection parameters (slices, etc.) to the read queue to be processed.

        Same input interface as `read`.
        """
        self.queue_work(*args, **kw)

    # rename get_result -> get_data
    def get_data(self):
        """Retrieve the least-recently read chunk of data.

        Blocks until a result is available.
        Same output interface as `read`.
        """
        return self.get_result()

    # rename process -> read
    def process(self, *args, **kw):
        return self.read(*args, **kw)

    @abc.abstractmethod
    def read(self, *args, **kw):
        """User-defined method for reading a chunk of data."""
        pass


class NvidiaMemoryWatcher(Thread):
    """Watch the memory usage of the GPU and log it to a file.

    Parameters
    ----------
    log_file : str
        The file to write the memory usage to.
    refresh_rate : float, optional
        The refresh_rate in seconds to check the memory usage, by default 1.0
    """

    def __init__(
        self,
        log_file: str = "nvidia_memory.log",
        refresh_rate: float = 1.0,
        gpu_id: int = 0,
    ):
        try:
            from pynvml.smi import nvidia_smi  # noqa: F401
        except ImportError:
            raise ImportError("Please install pynvml through pip or conda")

        super().__init__(name="NvidiaMemoryWatcher")
        self.log_file = log_file
        self.pid = os.getpid()
        self.t0 = time.time()
        self.refresh_rate = refresh_rate
        self.gpu_id = gpu_id
        # The query lag is the time it takes to query the GPU memory
        # This is used to try and refresh close to the refresh rate
        self._query_lag = 0.5
        self._finished_event = Event()
        self._thread = Thread(target=self.run)

        self._thread.start()

    def run(self):
        """Run the background task."""
        logger.info(
            f"Logging GPU memory usage to {self.log_file} every {self.refresh_rate} s"
        )
        with open(self.log_file, "w") as f:
            # Write the header
            f.write("time(s),memory(GB)\n")

        while not self._finished_event.is_set() and is_main_thread_active():
            mem = self._get_gpu_memory()
            t_cur = time.time() - self.t0
            with open(self.log_file, "a") as f:
                row = f"{t_cur:.3f},{mem:.2f}\n"
                f.write(row)

            # Sleep until the next refresh
            time.sleep(max(0, self.refresh_rate - self._query_lag))

    def join(self):
        """Wait for the thread to finish."""
        self._thread.join()

    def notify_finished(self):
        """Notify the thread that it should finish."""
        self._finished_event.set()
        self._thread.join()

    def _get_gpu_memory(self) -> float:
        """Get the memory usage (in GiB) of the GPU for the current pid."""
        from dolphin.utils import get_gpu_memory

        return get_gpu_memory(pid=self.pid, gpu_id=self.gpu_id)


class DummyProcessPoolExecutor(Executor):
    """Dummy ProcessPoolExecutor for to avoid forking for single_job purposes."""

    def __init__(self, max_workers: Optional[int] = None):
        self._max_workers = max_workers

    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        future: Future = Future()
        result = fn(*args, **kwargs)
        future.set_result(result)
        return future

    def shutdown(self, wait: bool = True):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        pass
