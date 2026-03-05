import os

from dolphin.utils import gpu_is_available

GPU_AVAILABLE = gpu_is_available() and os.environ.get("NUMBA_DISABLE_JIT") != "1"
