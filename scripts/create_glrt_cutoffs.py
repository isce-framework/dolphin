"""Compute Generalized Likelihood Ratio Test (GLRT) cutoffs.

Runs over different sample sizes (N), significance levels (alpha), and scales.
Results are saved in a CSV file.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy import log as ln
from scipy.stats import rayleigh


def get_test_stat_glrt(
    N: int,
    scale: float = 10,
    nsim: int = 500,
) -> np.ndarray:
    """Compute test statistics for the GLRT.

    Parameters
    ----------
    N : int
        Number of observations
    scale : float
        Scale parameter for the Rayleigh distribution
    nsim : float
        Number of simulations

    Returns
    -------
    ndarray:
        A numpy array of test statistics

    """
    x = rayleigh.rvs(scale=scale, size=(nsim, 2 * N))

    scale2_p = (x[:, :N] ** 2).mean(axis=1) / 2
    scale2_q = (x[:, N:] ** 2).mean(axis=1) / 2
    scale2_pooled = (scale2_p + scale2_q) / 2
    return 2 * ln(scale2_pooled) - ln(scale2_p) - ln(scale2_q)


def get_alpha_cutoff(
    alpha: float, N: int, nsim: int = 50000, scale: float = 10
) -> float:
    """Compute alpha cutoff for the GLRT test.

    Parameters
    ----------
    alpha : float
        Significance level
    N : int
        Number of observations
    scale : float
        Scale parameter for the Rayleigh distribution
    nsim : float
        Number of simulations


    Returns
    -------
    float:
        Alpha cutoff value

    """
    return np.percentile(
        get_test_stat_glrt(N=N, nsim=nsim, scale=scale), 100 * (1 - alpha)
    )


def compute_cutoff(
    alpha: float, N: int, scale: float
) -> tuple[int, float, float, float]:
    """Run computation for a single combination of N, alpha, and scale.

    Parameters
    ----------
    alpha : float
        Significance level
    N : int
        Number of observations
    scale : float
        Scale parameter for the Rayleigh distribution

    Returns
    -------
    (N, alpha, scale, computed cutoff)

    """
    return (N, alpha, scale, get_alpha_cutoff(alpha=alpha, N=N, scale=scale))


if __name__ == "__main__":
    Narr = list(range(1, 301))
    scales = (1, 10, 50)
    alphas = [0.05, 0.01, 0.005, 0.001]

    results = Parallel(n_jobs=30)(
        delayed(compute_cutoff)(*row) for row in product(Narr, alphas, scales)
    )

    df = pd.DataFrame(data=results, columns=["N", "alpha", "scale", "cutoff"])
    d2 = df.groupby(["N", "alpha"]).mean().round(4)[["cutoff"]]
    d2.to_csv("dolphin/shp/glrt_cutoffs.csv", index=True)
