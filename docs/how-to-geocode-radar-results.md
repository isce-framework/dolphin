# How to geocode dolphin results (ISCE2 / ISCE3)

After running dolphin on radar-geometry data processed with ISCE2 or ISCE3, the
outputs (interferograms, unwrapped phase, time series, velocity) are still in
swath/radar coordinates. The `dolphin geocode` command transforms them to
geographic coordinates using per-pixel latitude/longitude arrays from ISCE's
geometry products.

It auto-detects the geolocation file naming convention:

| Processor           | Geometry directory       | Lat file  | Lon file  |
| ------------------- | ------------------------ | --------- | --------- |
| ISCE3 / dolphin     | `geometry/`              | `y.tif`   | `x.tif`   |
| ISCE2 topsStack     | `merged/geom_reference/` | `lat.rdr` | `lon.rdr` |
| ISCE2 stripmapStack | `geom_reference/`        | `lat.rdr` | `lon.rdr` |

## Bulk geocode a dolphin work directory

The simplest way to geocode all outputs at once. This discovers rasters in
`timeseries/`, `unwrapped/`, etc. and writes geocoded results to
`<dolphin-dir>/geocoded/`, mirroring the directory structure.

```bash
dolphin geocode \
    -d ./dolphin_output \
    -g ./geometry \
    -c dolphin_config.yaml
```

The `-c` flag reads `output_options.strides` from your config so the
geolocation arrays (which are at full resolution) are properly subsampled to
match your multilooked outputs.

## ISCE2 topsStack example

```bash
dolphin geocode \
    -d ./dolphin_output \
    -g ./merged/geom_reference \
    -c dolphin_config.yaml
```

## ISCE2 stripmapStack example

```bash
dolphin geocode \
    -d ./dolphin_output \
    -g ./geom_reference \
    -c dolphin_config.yaml
```

## Geocode specific files

You can also pass individual files instead of a whole directory:

```bash
# Single file
dolphin geocode -i velocity.tif -g geometry/

# Multiple files to an output directory
dolphin geocode \
    -i timeseries/velocity.tif \
    -i unwrapped/20220101_20220201.unw.tif \
    -g geometry/ \
    -o geocoded/
```

## Reproject to UTM with a specific pixel spacing

```bash
dolphin geocode \
    -d ./dolphin_output \
    -g ./geometry \
    -c dolphin_config.yaml \
    --srs 32610 \
    -s 30
```

This reprojects to UTM zone 10N (EPSG:32610) with 30-meter pixel spacing.

## Apply a mask during geocoding

Pass a binary mask (0 = invalid, nonzero = valid) to mark pixels as nodata in
the geocoded output. The mask can be at the strided resolution or at full
resolution (it will be subsampled automatically if strides are set).

```bash
dolphin geocode \
    -i velocity.tif \
    -g geometry/ \
    -c dolphin_config.yaml \
    --mask water_mask.tif
```

## Include additional products

By default, bulk mode geocodes time series and unwrapped interferograms. Use
flags to include more:

```bash
dolphin geocode \
    -d ./dolphin_output \
    -g ./geometry \
    --include-interferograms \
    --include-auxiliary
```

| Flag                       | What it adds                                                       |
| -------------------------- | ------------------------------------------------------------------ |
| `--include-unwrapped`      | Unwrapped interferograms (on by default)                           |
| `--include-interferograms` | Wrapped interferograms, similarity, temporal/multilooked coherence |
| `--include-auxiliary`      | CRLB, amplitude dispersion                                         |

## Parallel processing

Geocoding is parallelized across files. Control the number of workers with `-j`:

```bash
dolphin geocode -d ./dolphin_output -g ./geometry -j 4
```
