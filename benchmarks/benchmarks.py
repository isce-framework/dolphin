import datetime
from pathlib import Path

import numpy as np
from osgeo import gdal

from dolphin import io, shp
from dolphin.phase_link import covariance, mle, simulate
from dolphin.stack import MiniStackPlanner
from dolphin.workflows import sequential

# Shared for all tests
HALF_WINDOW = {"x": 11, "y": 5}
STRIDES = {"x": 6, "y": 3}
SHAPE = (512, 512)


def _make_slc_samples(shape=SHAPE):
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
        cov_mat, neighbor_samples=np.prod(shape)
    )


def _make_slc_stack(
    out_path: Path = Path(), shape: tuple[int, int, int] = (10, 1024, 1024)
):
    sigma = 0.5
    slc_stack = np.random.normal(0, sigma, size=shape) + 1j * np.random.normal(
        0, sigma, size=shape
    )
    slc_stack = slc_stack.astype(np.complex64)

    start_date = datetime.datetime(2022, 1, 1)
    slc_date_list = []
    dt = datetime.timedelta(days=1)
    for i in range(len(slc_stack)):
        slc_date_list.append(start_date + i * dt)

    shape = slc_stack.shape
    # Write to a file
    driver = gdal.GetDriverByName("GTiff")
    d = out_path / "gtiff"
    d.mkdir()
    name_template = d / "{date}.slc.tif"

    file_list = []
    for cur_date, cur_slc in zip(slc_date_list, slc_stack):
        fname = str(name_template).format(date=cur_date.strftime("%Y%m%d"))
        file_list.append(Path(fname))
        ds = driver.Create(fname, shape[-1], shape[-2], 1, gdal.GDT_CFloat32)
        ds.GetRasterBand(1).WriteArray(cur_slc)
        ds = None

    return file_list, slc_date_list


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


class SingleMinistackBenchmark:
    """Benchmark block-by-block phase linking on a stack of rasters."""

    def setup_cache(self):
        self.slc_file_list, self.dates = _make_slc_stack(Path())

    def setup(self):
        self.vrt_stack = io.VRTStack(self.slc_file_list)
        self.ministack_planner = MiniStackPlanner(
            file_list=self.slc_file_list,
            dates=self.dates,
            is_compressed=[False] * len(self.slc_file_list),
            output_folder=Path(),
        )
        # We're using "sequential", but just making it one ministack
        self.ministack_size = len(self.slc_file_list)

    def time_single_ministack(self):
        sequential.run_wrapped_phase_sequential(
            slc_vrt_file=self.vrt_stack.outfile,
            ministack_planner=self.ministack_planner,
            ministack_size=self.ministack_size,
            half_window=HALF_WINDOW,
            strides=STRIDES,
            block_shape=(512, 512),
            # use_evd=cfg.phase_linking.use_evd,
            # n_workers=cfg.worker_settings.n_workers,
        )
