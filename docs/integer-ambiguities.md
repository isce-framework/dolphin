# Integer ambiguities in an interferogram network

`dolphin.timeseries.invert_stack_l1_integer` is an opt-in array API for estimating
integer cycle offsets from a connected, redundant interferogram network. It
uses the existing LAD solver followed by shared-threshold rounding. The default
displacement workflow continues to use continuous phase inversion.

The input is **integer cycles**, not unwrapped radians. Let `A` be the oriented
edge/date incidence matrix with the reference date removed, `u` the unwrapped
pair phases, and `theta` the per-date wrapped phase anchors. All must have the
same sign convention, grid, and spatial and temporal references. In particular,
the pair convention may have the opposite sign to the saved complex SLC phase.

```python
import numpy as np
from dolphin.timeseries import invert_stack_l1_integer

# u: (pairs, rows, cols); theta: (dates - 1, rows, cols).
# A: (pairs, dates - 1), with the same reference date removed as in theta.
residual = u - np.einsum("ij,jkl->ikl", A, theta)
k = np.rint(residual / (2 * np.pi))
if not np.all(np.isfinite(residual)) or np.max(
    np.abs(residual - 2 * np.pi * k)
) > 1e-3:
    raise ValueError("Pair phases are not congruent with the date anchors")
offsets, residual_cycles = invert_stack_l1_integer(A, k, max_iter=200)
phase = theta + 2 * np.pi * np.asarray(offsets)
```

The example's congruence tolerance must be justified for the data precision.
Independently wrapping each pair does not supply a consistent per-date anchor.
Filtering, multilooking, mosaicking, or inconsistent spatial references can
break the required relation even when every unwrapped pair is individually
congruent with its own wrapped input. Do not apply this construction to an
ionosphere screen, whose frequency-dependent coefficients are not integers.

The API supports one unweighted graph shared by all pixels. Remove missing
edges from **both** `A` and `k`; filling a missing observation with zero asserts
a measured zero-cycle constraint. Group pixels by validity pattern before using
this API. Dates disconnected from the reference have no defined offsets; the
function rejects disconnected inputs. Per-pixel weights, masked raster I/O,
phase-anchor extraction, and workflow configuration are outside this API.

## What rounding guarantees

For a continuous date solution `x`, use `floor(x + t)` with one common threshold
`t` in `[0, 1)`, including a fixed zero reference. Each edge difference takes
at most two adjacent integer values with mean equal to its continuous value.
For an integer observation, absolute error is linear between those adjacent
values, so its expected rounded error equals its continuous error. The same is
true after summing over the edges.

The best threshold therefore cannot increase the LAD objective, up to floating
point precision. If `x` is an exact continuous optimum, the rounded solution is
an integer optimum. Independent nearest-integer rounding lacks this guarantee.
The implementation evaluates changes at sorted fractional node values; each
edge contributes at most two cost changes. Equal thresholds move together.

The default 20 ADMM iterations do **not** certify convergence or optimality.
Increasing `max_iter` can help difficult networks, but no fixed budget provides
a general certificate. The returned residual is the rounded solution's actual
LAD objective, not an optimality gap. Compare with an exact LP when validating
new network regimes. Integer output and temporal consistency also do not prove
the true unwrapped phase: acquisition-consistent errors remain invisible, and
multiple integer solutions may have the same objective.

Use `benchmarks/benchmarks_integer.py` to measure the added cost on the intended
stack dimensions and device. The method adds sorting, gathering, and scatter
work; it does not have the same runtime as the continuous solver.
