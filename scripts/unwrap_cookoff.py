import argparse
import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
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
DEFAULT_COH_THRESHOLD = 0.6


# @dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class CookoffRun(BaseModel):
    """Configuration for one unwrapping test run."""

    path: Path
    unwrapper: UnwrapMethod
    ntiles: tuple[int, int] = NO_TILES
    run_interpolation: bool = False
    run_goldstein: bool = False
    n_parallel_jobs: int = 1
    spurt_options: SpurtOptions = Field(default_factory=SpurtOptions)
    spurt_coh_threshold: float = DEFAULT_COH_THRESHOLD
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
            if (threshold := self.spurt_coh_threshold) != DEFAULT_COH_THRESHOLD:
                n += f"_threshold_{threshold}"
            if (cost := self.spurt_options.solver_settings.t_cost_type) != "constant":
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
                    "solver_settings": self.spurt_options.solver_settings,
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
            spurt_options={
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


def parse_log_files(log_files: Iterable[Path | str]) -> pd.DataFrame:
    """Parse all .log files and assemble a DataFrame of results.

    Parameters
    ----------
    log_files : Iterable[Path | str]
        Iterable of log files to attempt parsing on.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the parsed information from all log files.

    """
    data: list[dict[str, Any]] = []

    for log_file in log_files:
        with open(log_file, "r") as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not log_entry["message"].startswith("Completed"):
                    continue
                options = log_entry["cookoff_options"]
                opts = CookoffRun.model_validate(options)
                row = {
                    "name": opts.name,
                    "elapsed_seconds": log_entry["elapsed"],
                    "max_memory_gb": log_entry["max_memory_gb"],
                    "spurt_cost": opts.spurt_options.solver_settings.s_cost_type,
                    "spurt_temporal_coherence_threshold": opts.spurt_coh_threshold,
                    **opts.model_dump(exclude={"spurt_options"}),
                }
                # prettify the enum
                row["unwrapper"] = row["unwrapper"].value

                data.append(row)
                break
            else:
                print(f"No `Completed` message found in {log_file}")

    return pd.DataFrame(data)


def _get_cli_args() -> argparse.Namespace:
    run_descriptions = [
        f"{idx}:{o.name}" for idx, o in enumerate(create_unwrap_options(), start=1)
    ]

    def parse_steps(string_value: str):
        """Parse ranges of steps, from https://stackoverflow.com/a/4726287."""
        if string_value is None:
            return []

        step_nums: set[int] = set()
        try:
            for part in string_value.split(","):
                x = part.split("-")
                step_nums.update(range(int(x[0]), int(x[-1]) + 1))
        except (ValueError, AttributeError) as e:
            raise TypeError(
                "Must be comma separated integers and/or dash separated range."
            ) from e
        max_step = len(run_descriptions)
        if any((num < 1 or num > max_step) for num in step_nums):
            raise TypeError(f"Must be ints between 1 and {max_step}")

        return sorted(step_nums)

    parser = argparse.ArgumentParser(
        description="Run unwrapping cookoff tests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("unwrap_cookoff"),
        help="Base directory for output",
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
        "--run",
        type=parse_steps,
        default=list(range(1, 1 + len(run_descriptions))),
        help=(
            "Indexes of configurations to run. None runs all options. "
            " Examples: --run 0,1,4 --run 3-6 --step 1,9-10."
            " Options: {}".format(", ".join(run_descriptions))
        ),
    )
    parser.add_argument(
        "--nlooks",
        type=int,
        default=30,
        help="Effective number of looks used to form correlation",
    )
    parser.add_argument(
        "--ntiles",
        type=int,
        nargs=2,
        default=(2, 2),
        help="Number of tiles to use for SNAPHU runs using tiling.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        nargs=2,
        default=(300, 300),
        help="Amount of tile overlap for SNAPHU runs using tiling.",
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
    all_options = create_unwrap_options(
        base_dir,
        n_parallel_jobs=args.n_parallel_jobs,
        ntiles=args.ntiles,
        tile_overlap=args.tile_overlap,
    )

    selected_options = [all_options[i - 1] for i in args.run]
    for run in selected_options:
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
