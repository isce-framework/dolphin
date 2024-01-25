# Benchmark CI

<!-- Taken from scikit-image -->
<!-- https://github.com/scikit-image/scikit-image/pull/5424 -->

## How it works

The `asv` suite can be run for any PR on GitHub Actions (check workflow `.github/workflows/benchmarks.yml`) by adding a `run-benchmark` label to said PR. This will trigger a job that will run the benchmarking suite for the current PR head (merged commit) against the PR base (usually `main`).

We use `asv continuous` to run the job, which runs a relative performance measurement. This means that there's no state to be saved and that regressions are only caught in terms of performance ratio (absolute numbers are available but they are not useful since we do not use stable hardware over time). `asv continuous` will:

## Running the benchmarks on GitHub Actions

1. On a PR, add the label `run-benchmark`.
2. The CI job will be started. Checks will appear in the usual dashboard panel above the comment box.
3. If more commits are added, the label checks will be grouped with the last commit checks _before_ you added the label.
4. Alternatively, you can always go to the `Actions` tab in the repo and [filter for `workflow:Benchmark`](https://github.com/isce-framework/dolphin/actions?query=workflow%3ABenchmark). Your username will be assigned to the `actor` field, so you can also filter the results with that if you need it.

## The artifacts

The CI job will also generate an artifact. This is the `.asv/results` directory compressed in a zip file. Its contents include:

* `fv-xxxxx-xx/`. A directory for the machine that ran the suite. It contains three files:
  * `<baseline>.json`, `<contender>.json`: the benchmark results for each commit, with stats.
  * `machine.json`: details about the hardware.
* `benchmarks.json`: metadata about the current benchmark suite.
* `benchmarks.log`: the CI logs for this run.
* This README.
