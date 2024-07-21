import argparse
import logging
import time
from pathlib import Path

from pydantic import BaseModel, Field

from dolphin._log import setup_logging
from dolphin.utils import get_max_memory_usage
from dolphin.workflows import (
    SnaphuOptions,
    SpurtOptions,
    UnwrapMethod,
    UnwrapOptions,
    unwrapping,
)

logger = logging.getLogger("dolphin")


NO_TILES = (1, 1)


# @dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class CookoffRun(BaseModel):
    """Configuration for one unwrapping test run."""

    path: Path
    unwrapper: UnwrapMethod
    ntiles: tuple[int, int] = NO_TILES
    run_interpolation: bool = False
    run_goldstein: bool = False
    n_parallel_jobs: int = 1
    spurt_cost: SpurtOptions = Field(default_factory=SpurtOptions)
    spurt_coh_threshold: float = 0.6
    tile_overlap: tuple[int, int] = (400, 400)

    @property
    def name(self) -> str:
        """Name of the unwrapping configuration."""
        n = str(self.unwrapper.value)
        if self.ntiles != NO_TILES:
            n += f"_tiling_{self.ntiles[0]}-{self.ntiles[1]}"
        if self.run_interpolation:
            n += "_interpolated"
        if self.run_goldstein:
            n += "_goldstein"
        if self.unwrapper == UnwrapMethod.SPURT:
            n += f"_threshold_{self.spurt_coh_threshold}"
            cost = self.spurt_cost.solver_settings.t_cost_type
            if cost != "constant":
                n += f"_cost_{cost}"

        return n

    @property
    def work_dir(self) -> Path:
        """Working directory to store results."""
        return self.path / self.name

    def get_options(self) -> UnwrapOptions:
        """Convert the specified options to an `UnwrapOptions` object."""
        common_options = {
            "run_unwrap": True,
            "run_interpolation": self.run_interpolation,
            "run_goldstein": self.run_goldstein,
            "unwrap_method": self.unwrapper,
            "n_parallel_jobs": self.n_parallel_jobs,
        }

        if self.unwrapper == UnwrapMethod.SNAPHU:
            snaphu_options = SnaphuOptions(
                ntiles=self.ntiles,
                tile_overlap=self.tile_overlap,
            )
            opts = UnwrapOptions(**common_options, snaphu_options=snaphu_options)
        elif self.unwrapper == UnwrapMethod.SPURT:
            spurt_options = SpurtOptions(
                temporal_coherence_threshold=self.spurt_coh_threshold,
                solver_settings={
                    "solver_settings": self.spurt_cost,
                },
            )
            opts = UnwrapOptions(**common_options, spurt_options=spurt_options)
        else:
            opts = UnwrapOptions(**common_options)
        opts._directory = self.work_dir
        return opts


def create_unwrap_options(
    base_dir: Path = Path(),
    n_parallel_jobs: int = 1,
    ntiles: tuple[int, int] = (2, 2),
    tile_overlap: tuple[int, int] = (300, 300),
) -> list[CookoffRun]:
    """Generate all unwrapping configurations to test."""
    base_dir.mkdir(exist_ok=True)

    common = {
        "path": base_dir,
        "n_parallel_jobs": n_parallel_jobs,
        "tile_overlap": tile_overlap,
    }
    all_options = [
        # SNAPHU baseline
        CookoffRun(
            unwrapper=UnwrapMethod.SNAPHU,
            ntiles=ntiles,
            run_interpolation=True,
            run_goldstein=True,
            **common,
        ),
        # SNAPHU variations
        CookoffRun(
            unwrapper=UnwrapMethod.SNAPHU,
            ntiles=ntiles,
            run_interpolation=True,
            **common,
        ),
        CookoffRun(unwrapper=UnwrapMethod.SNAPHU, ntiles=ntiles, **common),
        CookoffRun(unwrapper=UnwrapMethod.SNAPHU, run_interpolation=True, **common),
        CookoffRun(unwrapper=UnwrapMethod.SNAPHU, run_goldstein=True, **common),
        CookoffRun(unwrapper=UnwrapMethod.SNAPHU, **common),
        # PHASS attempts
        CookoffRun(unwrapper=UnwrapMethod.PHASS, **common),
        CookoffRun(unwrapper=UnwrapMethod.PHASS, run_interpolation=True, **common),
        # Spurt attempts
        CookoffRun(unwrapper=UnwrapMethod.SPURT, **common),
        CookoffRun(unwrapper=UnwrapMethod.SPURT, spurt_coh_threshold=0.7, **common),
        CookoffRun(
            unwrapper=UnwrapMethod.SPURT,
            spurt_cost={
                "solver_settings": {
                    "t_cost_type": "distance",
                    "s_cost_type": "distance",
                }
            },
            **common,
        ),
        # Whirlwind attempts
        CookoffRun(unwrapper=UnwrapMethod.WHIRLWIND, **common),
        CookoffRun(unwrapper=UnwrapMethod.WHIRLWIND, run_interpolation=True, **common),
    ]

    return all_options


def _get_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unwrapping cookoff tests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("unwrap_cookoff"),
        help="Base directory for output (default: ./unwrap_cookoff)",
    )
    # Get Inputs from the command line
    inputs = parser.add_argument_group("File Inputs")
    inputs.add_argument(
        "--ifg-dir",
        type=Path,
        required=True,
        help="Directory containing *.int.tif and *.cor.tif. and temporal_coherence.tif",
    )
    inputs.add_argument(
        "--mask-filename",
        help=(
            "Path to Byte mask file used to ignore low correlation/bad data (e.g water"
            " mask). Convention is 0 for no data/invalid, and 1 for good data."
        ),
    )
    parser.add_argument(
        "--nlooks",
        type=int,
        default=30,
        help="Effective number of looks used to form correlation",
    )
    parser.add_argument(
        "--n-parallel-jobs",
        type=int,
        default=1,
        help="Number of interferograms to unwrap in parallel for each",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _get_cli_args()
    ifg_dir: Path = args.ifg_dir

    ifg_file_list = sorted(ifg_dir.glob("*.int.tif"))
    cor_file_list = sorted(ifg_dir.glob("*.cor.tif"))
    temporal_coherence_file = next(ifg_dir.glob("*temporal_coherence*tif"))

    base_dir: Path = args.base_dir
    all_options = create_unwrap_options(base_dir)

    for run in all_options:
        log_file = base_dir / f"{run.name}.log"
        setup_logging(filename=log_file)
        if run.unwrapper == UnwrapMethod.SPURT:
            setup_logging(logger_name="spurt", filename=log_file)
        if run.unwrapper == UnwrapMethod.WHIRLWIND:
            setup_logging(logger_name="whirlwind", filename=log_file)
        logger.info(f"Running cookoff for: {run.name}")
        logger.info(f"Path: {run.path}")
        options = run.get_options()
        logger.info(f"Options: {options}")

        # Here you would typically run your unwrapping process
        # For example:
        t0 = time.perf_counter()
        unwrapping.run(
            ifg_file_list=ifg_file_list,
            cor_file_list=cor_file_list,
            nlooks=args.nlooks,
            temporal_coherence_file=temporal_coherence_file,
            unwrap_options=options,
        )

        elapsed = time.perf_counter() - t0
        max_mem = get_max_memory_usage(units="GB")

        # Dump extra information for the JSON logger
        extra = {
            "elapsed": elapsed,
            "max_memory_gb": max_mem,
            "cookoff_options": run.model_dump(mode="json"),
        }
        logger.info(f"Completed {run.name}", extra=extra)

    logger.info("Cookoff completed")
