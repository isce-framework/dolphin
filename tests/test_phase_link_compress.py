import os

import numpy as np
import numpy.testing as npt
import pytest

from dolphin._types import HalfWindow, Strides
from dolphin.phase_link import _compress, _core, simulate
from dolphin.utils import gpu_is_available

GPU_AVAILABLE = gpu_is_available() and os.environ.get("NUMBA_DISABLE_JIT") != "1"


@pytest.fixture
def slc_samples():
    C = np.eye(10)
    ns = 11 * 11
    return simulate.simulate_neighborhood_stack(C, ns)


@pytest.mark.parametrize("strides", [1, 3])
def test_compression(slc_samples, strides):
    slc_stack = slc_samples.reshape(10, 11, 11)

    # cpx_phase, temp_coh, eigs, _ = _core.run_cpl(
    pl_out = _core.run_cpl(
        slc_stack,
        HalfWindow(x=3, y=3),
        Strides(x=strides, y=strides),
    )
    comp_slc = _compress.compress(slc_stack=slc_stack, pl_cpx_phase=pl_out.cpx_phase)
    assert comp_slc.shape == slc_stack.shape[1:]

    # When striding, the upsampling will leave some nans at the edges
    valid_rows = slc_stack.shape[1] // strides * strides
    valid_cols = slc_stack.shape[2] // strides * strides

    stack_mean = np.mean(np.abs(slc_stack), axis=0)
    npt.assert_array_almost_equal(
        stack_mean[:valid_rows, :valid_cols], np.abs(comp_slc)[:valid_rows, :valid_cols]
    )
    assert np.isnan(np.abs(comp_slc)[valid_rows:, :]).all()
    assert np.isnan(np.abs(comp_slc)[:, valid_cols:]).all()
