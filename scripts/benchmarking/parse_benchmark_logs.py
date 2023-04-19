#!/usr/bin/env python
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from dolphin._types import Filename
from dolphin.workflows._pge_runconfig import RunConfig
from dolphin.workflows.config import Workflow


def get_df(dolphin_config_file: Filename):
    """Create a dataframe from a directory of log files."""
    if Path(dolphin_config_file).name == "runconfig.yaml":
        rc = RunConfig.from_yaml(dolphin_config_file)
        alg_file = rc.dynamic_ancillary_file_group.algorithm_parameters_file
        if not alg_file.is_absolute():
            alg_file = (
                Path(dolphin_config_file).parent
                / rc.dynamic_ancillary_file_group.algorithm_parameters_file
            )
        w = rc.to_workflow()
    else:
        w = Workflow.from_yaml(dolphin_config_file)

    log_file = Path(w.log_file)

    result = _get_memory_and_runtimes(log_file)
    cfg_data = _parse_config(w)

    result.update(cfg_data)
    return pd.DataFrame([result])


def _get_memory_and_runtimes(logfile):
    out = defaultdict(list)
    out["file"] = str(Path(logfile).resolve())
    mempat = r"Maximum memory usage: (\d\.\d{2}) GB"
    timepat = (
        r"Total elapsed time for dolphin.workflows.s1_disp.run : (\d*\.\d{2}) minutes"
        r" \((\d*\.\d{2}) seconds\)"
    )
    wrapped_phase_timepat = (
        r"Total elapsed time for dolphin.workflows.wrapped_phase.run : (\d*\.\d{2})"
        r" minutes"
        r" \((\d*\.\d{2}) seconds\)"
    )
    for line in open(logfile).readlines():
        if m := re.search(mempat, line):
            out["memory"] = float(m.groups()[0])
            continue
        if m := re.search(timepat, line):
            out["runtime"] = float(m.groups()[1])
            continue
        if m := re.search(wrapped_phase_timepat, line):
            out["wrapped_phase_runtimes"].append(float(m.groups()[1]))
            continue
    return out


def _parse_config(workflow: Workflow):
    """Grab the relevant parameters from the config file."""
    return {
        "block": workflow.worker_settings.block_size_gb,
        "strides": workflow.output_options.strides,
        "threads_per_worker": workflow.worker_settings.threads_per_worker,
        "n_slc": len(workflow.cslc_file_list),
        "n_workers": workflow.worker_settings.n_workers,
        "creation_time": workflow.creation_time_utc,
        "dolphin_version": workflow.dolphin_version,
    }


def _get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-files",
        nargs="*",
        help="Path(s) to config file",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        help="Output file (HTML). If None, outputs as `log_file`.html ",
        default=None,
    )
    return parser.parse_args()


def main():
    """Run main entry point."""
    args = _get_cli_args()
    dfs = []
    for config_file in args.config_files:
        dfs.append(get_df(config_file))

    df = pd.concat(dfs)

    from datetime import datetime

    if not args.outfile:
        # Save as csv file with same directory as log_file
        outfile = (
            Path(args.config_files[0]).parent
            / f"benchmarks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    else:
        outfile = args.outfile

    if not outfile.endswith(".csv"):
        outfile = outfile + ".csv"
        print(f"Output file must be csv. Writing to {outfile}", file=sys.stderr)

    df.to_csv(outfile, index=False)


if __name__ == "__main__":
    main()
