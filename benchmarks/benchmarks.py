import numpy as np
from dolphin import shp
from dolphin.phase_link import covariance, mle, simulate

# Shared for all tests
HALF_WINDOW = {"x": 11, "y": 5}
STRIDES = {"x": 6, "y": 3}
SHAPE = (512, 512)


def _make_slc_samples():
    """Create some sample SLC data."""
    # Make it as large as the biggest test
    cov_mat, _ = simulate.simulate_coh(
        num_acq=30,
        Tau0=72,
        gamma_inf=0.3,
        gamma0=0.99,
        add_signal=True,
        signal_std=0.01,
    )
    return simulate.simulate_neighborhood_stack(
        cov_mat, neighbor_samples=np.prod(SHAPE)
    )


class CovarianceBenchmark:
    """Benchmark results for covariance matrix formation."""

    # https://asv.readthedocs.io/en/v0.6.1/writing_benchmarks.html#parameterized-benchmarks
    # Parameterize by the number of SLCs
    params = [10, 20, 30]
    param_names = ["nslc"]

    def setup_cache(self):
        # Run the several-second generation of samples once in setup_cache
        # https://asv.readthedocs.io/en/v0.6.1/writing_benchmarks.html
        np.save("slc_samples.npy", _make_slc_samples())

    def setup(self, nslc):
        self.slc_samples = np.load("slc_samples.npy")[:nslc, :]
        self.slc_stack = self.slc_samples.reshape((nslc, *SHAPE))

    def time_covariance_single(self, nslc):
        # Test one covariance matrix on part of the samples
        covariance.coh_mat_single(self.slc_samples[:, :200])

    def time_covariance_stack(self, nslc):
        covariance.estimate_stack_covariance_cpu(
            self.slc_stack, half_window=HALF_WINDOW, strides=STRIDES
        )

    def peakmem_covariance_stack(self, nslc):
        covariance.estimate_stack_covariance_cpu(
            self.slc_stack, half_window=HALF_WINDOW, strides=STRIDES
        )


class PhaseLinkingBenchmark:
    """Benchmark phase linking algorithms."""

    # (nslc, use_evd)
    params = ([10, 20, 30], [True, False])
    param_names = ["nslc", "use_evd"]

    def setup_cache(self):
        np.save("slc_samples.npy", _make_slc_samples())

    def setup(self, nslc: int, use_evd: bool):
        self.slc_samples = np.load("slc_samples.npy")[:nslc, :]
        self.slc_stack = self.slc_samples.reshape((nslc, *SHAPE))

    def time_phase_link(self, nslc: int, use_evd: bool):
        mle.run_mle(
            self.slc_stack,
            half_window=HALF_WINDOW,
            strides=STRIDES,
            use_evd=use_evd,
        )

    def peakmem_phase_link(self, nslc: int, use_evd: bool):
        mle.run_mle(
            self.slc_stack,
            half_window=HALF_WINDOW,
            strides=STRIDES,
            use_evd=use_evd,
        )


class ShpBenchmark:
    """Benchmark suite for SHP estimation functions."""

    def setup(self):
        self.nslc = 30
        slc_samples = _make_slc_samples()
        slc_stack = slc_samples.reshape((self.nslc, *SHAPE))

        self.amp_mean = np.mean(np.abs(slc_stack), axis=0)
        self.amp_variance = np.var(np.abs(slc_stack), axis=0)

    def time_estimate_neighbors(self):
        shp.estimate_neighbors(
            halfwin_rowcol=(HALF_WINDOW["y"], HALF_WINDOW["x"]),
            alpha=0.001,
            strides=STRIDES,
            mean=self.amp_mean,
            var=self.amp_variance,
            nslc=30,
            method=shp.ShpMethod.GLRT,
        )
