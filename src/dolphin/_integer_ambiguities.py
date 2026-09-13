"""Shared-threshold rounding for integer observations on a graph."""

import jax
import jax.numpy as jnp


@jax.jit
def _round_ambiguities(A, b, x):
    """Minimize the LAD objective over shared-threshold roundings of ``x``.

    ``A`` must be a reduced oriented incidence matrix and ``b`` integer-valued.
    The fixed reference node is zero. Each edge changes its residual at most
    twice as the threshold moves. Accumulate these cost changes at the sorted
    node events, using O(M + N*log(N)) work after extracting edge endpoints,
    and O(M + N) memory. Output retains the floating input dtype.
    """
    base = jnp.floor(x)
    fractions = x - base
    order = jnp.argsort(-fractions, stable=True)
    ranks = jnp.zeros(x.size, dtype=jnp.int32).at[order].set(jnp.arange(x.size))
    # Integer nodes, including the reference, never move for t in [0, 1).
    events = jnp.append(jnp.where(fractions > 0, ranks, x.size), x.size)
    positive = jnp.where(jnp.any(A == 1, axis=1), jnp.argmax(A, axis=1), x.size)
    negative = jnp.where(jnp.any(A == -1, axis=1), jnp.argmin(A, axis=1), x.size)
    full_base = jnp.append(base, 0)
    residual = full_base[positive] - full_base[negative] - b
    objective = jnp.sum(jnp.abs(residual))
    pos_event, neg_event = events[positive], events[negative]
    first, second = jnp.minimum(pos_event, neg_event), jnp.maximum(pos_event, neg_event)
    sign = jnp.where(pos_event < neg_event, 1, -1)
    delta = jnp.abs(residual + sign) - jnp.abs(residual)
    # The second endpoint restores the original edge residual. Events at N
    # belong to t=1 and are excluded. Simultaneous events cancel exactly.
    changes = jnp.zeros(x.size + 1, dtype=x.dtype)
    changes = changes.at[first].add(delta).at[second].add(-delta)
    costs = objective + jnp.cumsum(changes[:-1])
    sorted_fractions = fractions[order]
    end_group = sorted_fractions != jnp.append(sorted_fractions[1:], -1)
    valid = end_group & (sorted_fractions > 0)
    costs = jnp.append(objective, jnp.where(valid, costs, jnp.inf))
    best_count = jnp.argmin(costs)
    return base + (ranks < best_count), costs[best_count]
