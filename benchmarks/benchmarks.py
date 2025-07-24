import datetime
import shutil
import subprocess
from pathlib import Path

import numpy as np
from opera_utils import get_dates
from osgeo import gdal

from dolphin import io, shp
from dolphin._types import HalfWindow, Strides
from dolphin.phase_link import _core, covariance, simulate
from dolphin.stack import MiniStackPlanner
from dolphin.workflows import sequential

# Shared for all tests
HALF_WINDOW = HalfWindow(11, 5)
STRIDES = Strides(3, 6)
HALF_WINDOW_DICT = {"x": HALF_WINDOW.x, "y": HALF_WINDOW.y}
STRIDES_DICT = {"x": STRIDES.x, "y": STRIDES.y}

SHAPE = (512, 512)

# monkeypatch the old names for single covariance computation
if hasattr(covariance, "estimate_stack_covariance_cpu"):
    # The old one wants a dict for half window/strides

    def f(slc_stack, half_window, strides, neighbor_arrays):
        half_window_dict = {"x": half_window.x, "y": half_window.y}
        strides_dict = {"x": strides.x, "y": strides.y}

        return covariance.estimate_stack_covariance_cpu(
            slc_stack,
            half_window=half_window_dict,
            strides=strides_dict,
            neighbor_arrays=neighbor_arrays,
        )

    covariance.estimate_stack_covariance = f  # type: ignore[assignment]


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
        covariance.estimate_stack_covariance(
            self.slc_stack, half_window=HALF_WINDOW, strides=STRIDES
        )

    def peakmem_covariance_stack(self, nslc):
        covariance.estimate_stack_covariance(
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
        _core.run_phase_linking(
            self.slc_stack,
            half_window=HALF_WINDOW,
            strides=STRIDES,
            use_evd=use_evd,
        )

    def peakmem_phase_link(self, nslc: int, use_evd: bool):
        _core.run_phase_linking(
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


def _make_slc_stack(out_path: Path, shape: tuple[int, int, int] = (10, 1024, 1024)):
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
    out_path.mkdir(exist_ok=True)
    name_template = out_path / "{date}.slc.tif"

    file_list = []
    for cur_date, cur_slc in zip(slc_date_list, slc_stack, strict=False):
        fname = str(name_template).format(date=cur_date.strftime("%Y%m%d"))
        file_list.append(Path(fname))
        ds = driver.Create(fname, shape[-1], shape[-2], 1, gdal.GDT_CFloat32)
        ds.GetRasterBand(1).WriteArray(cur_slc)
        ds = None

    return file_list, slc_date_list


class SingleMinistackBenchmark:
    """Benchmark block-by-block phase linking on a stack of rasters."""

    def setup_cache(self):
        _make_slc_stack(Path("slcs"))

    def setup(self):
        # TODO: If we change the dates/size, we'll need a way to remove the last run
        output_folder = Path("pl")
        if output_folder.exists():
            shutil.rmtree(output_folder, ignore_errors=True)

        output_folder.mkdir()
        v_file = output_folder / "stack.vrt"

        subprocess.run("vmtouch -e .", shell=True)  # noqa: PLW1510
        self.slc_file_list = sorted(Path("slcs").glob("20*.slc.tif"))
        assert (
            len(self.slc_file_list) > 0
        ), f"No SLC files found: {list(Path('slcs').glob('*'))}"
        self.dates = [get_dates(f) for f in self.slc_file_list]

        io.VRTStack(self.slc_file_list, outfile=Path("pl") / "stack.vrt")
        self.v_file = v_file

    def time_single_ministack(self):
        ministack_planner = MiniStackPlanner(
            file_list=self.slc_file_list,
            dates=self.dates,
            is_compressed=[False] * len(self.slc_file_list),
            output_folder=Path("pl"),
        )
        # We're using "sequential", but just making it one ministack
        ministack_size = len(self.slc_file_list)
        sequential.run_wrapped_phase_sequential(
            slc_vrt_file=self.v_file,
            ministack_planner=ministack_planner,
            ministack_size=ministack_size,
            half_window=HALF_WINDOW,
            strides=STRIDES,
            block_shape=(512, 512),
            # use_evd=cfg.phase_linking.use_evd,
            # n_workers=cfg.worker_settings.n_workers,
        )
