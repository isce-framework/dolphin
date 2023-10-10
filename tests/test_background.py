import os
import time

import pytest

from dolphin._background import CPURecorder, NvidiaRecorder
from dolphin.utils import gpu_is_available

GPU_AVAILABLE = gpu_is_available() and not (os.environ.get("NUMBA_DISABLE_JIT") == "1")


def test_cpu_recorder():
    recorder = CPURecorder(interval=0.2)
    time.sleep(0.5)
    recorder.notify_finished()
    # Should be at {0, 0.2, 0.4}, but CI can be shaky
    assert len(recorder.results) >= 1


def test_cpu_recorder_filename(tmp_path):
    filename = tmp_path / "cpu.log"
    with CPURecorder(interval=0.2, filename=filename) as recorder:
        time.sleep(0.5)
    assert not recorder._thread.is_alive()

    # Check that the output file exists
    assert filename.exists()
    # Check that the output file is not empty
    assert filename.stat().st_size > 0
    # Check that there is one line per interval, plus header
    assert len(recorder.results) >= 1
    lines = filename.read_text().splitlines()
    assert len(lines) >= 1

    header = lines[0]
    assert header == ",".join(recorder.columns)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_gpu_recorder(tmp_path):
    filename = tmp_path / "gpu.log"
    recorder = NvidiaRecorder(interval=0.2, filename=filename)
    time.sleep(0.5)
    recorder.notify_finished()
    assert len(recorder.results) >= 1

    lines = filename.read_text().splitlines()
    assert len(lines) >= 1

    header = lines[0]
    assert header == ",".join(recorder.columns)
