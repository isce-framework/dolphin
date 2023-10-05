"""Combine estimated DS phases with PS phases to form interferograms."""
from __future__ import annotations

import itertools
from os import fspath
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike
from osgeo import gdal
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from dolphin import io, utils
from dolphin._log import get_log
from dolphin._types import Filename

gdal.UseExceptions()

logger = get_log(__name__)


class VRTInterferogram(BaseModel, extra="allow"):
    """Create an interferogram using a VRTDerivedRasterBand.

    Attributes
    ----------
    ref_slc : Union[Path, str]
        Path to reference SLC file
    sec_slc : Union[Path, str]
        Path to secondary SLC file
    path : Optional[Path], optional
        Path to output interferogram. Defaults to Path('<date1>_<date2>.vrt'),
        placed in the same directory as `ref_slc`.
    outdir : Optional[Path], optional
        Directory to place output interferogram. Defaults to the same directory as
        `ref_slc`. If only `outdir` is specified, the output interferogram will
        be named '<date1>_<date2>.vrt', where the dates are parsed from the
        inputs. If `path` is specified, this is ignored.
    subdataset : Optional[str], optional
        Subdataset to use for the input files (if passing file paths
        to NETCDF/HDF5 files).
        Defaults to None.
    date_format : str, optional
        Date format to use when parsing dates from the input files.
        Defaults to '%Y%m%d'.
    pixel_function : str, optional
        GDAL Pixel function to use, choices={'cmul', 'mul'}.
        Defaults to 'cmul', which performs `ref_slc * sec_slc.conj()`.
        See https://gdal.org/drivers/raster/vrt.html#default-pixel-functions
    dates : tuple[datetime.date, datetime.date]
        Date of the interferogram (parsed from the input files).

    """

    subdataset: Optional[str] = Field(
        None,
        description="Subdataset to use for the input files. Defaults to None.",
    )
    ref_slc: Union[Path, str] = Field(..., description="Path to reference SLC file")
    sec_slc: Union[Path, str] = Field(..., description="Path to secondary SLC file")
    outdir: Optional[Path] = Field(
        None,
        description=(
            "Directory to place output interferogram. Defaults to the same"
            " directory as `ref_slc`. If only `outdir` is specified, the output"
            " interferogram will be named '<date1>_<date2>.vrt', where the dates"
            " are parsed from the inputs. If `path` is specified, this is ignored."
        ),
        validate_default=True,
    )
    path: Optional[Path] = Field(
        None,
        description=(
            "Path to output interferogram. Defaults to '<date1>_<date2>.vrt', where the"
            " dates are parsed from the input files, placed in the same directory as"
            " `ref_slc`."
        ),
        validate_default=True,
    )
    date_format: str = "%Y%m%d"
    write: bool = Field(True, description="Write the VRT file to disk")

    pixel_function: Literal["cmul", "mul"] = "cmul"
    _template = """\
<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">
    <VRTRasterBand dataType="CFloat32" band="1" subClass="VRTDerivedRasterBand">
        <PixelFunctionType>{pixel_function}</PixelFunctionType>
        <SimpleSource>
            <SourceFilename relativeToVRT="0">{ref_slc}</SourceFilename>
        </SimpleSource>
        <SimpleSource>
            <SourceFilename relativeToVRT="0">{sec_slc}</SourceFilename>
        </SimpleSource>
    </VRTRasterBand>
</VRTDataset>
    """

    @field_validator("ref_slc", "sec_slc")
    @classmethod
    def _check_gdal_string(cls, v: Union[Path, str], info: ValidationInfo):
        subdataset = info.data.get("subdataset")
        # If we're using a subdataset, create a the GDAL-readable string
        gdal_str = io.format_nc_filename(v, subdataset)
        try:
            # First make sure it's openable
            gdal.Info(fspath(gdal_str))
        except RuntimeError:
            raise ValueError(f"File {gdal_str} is not a valid GDAL dataset")
        # Then, if we passed a string like 'NETCDF:"file.nc":band', make sure
        # the file is absolute
        if ":" in str(gdal_str):
            try:
                gdal_str = str(utils._resolve_gdal_path(gdal_str))
            except Exception:
                # if the file had colons for some reason but
                # it didn't match, just ignore
                pass
        return gdal_str

    @field_validator("outdir")
    @classmethod
    def _check_output_dir(cls, v, info: ValidationInfo):
        if v is not None:
            return Path(v)
        # If outdir is not set, use the directory of the reference SLC
        ref_slc = str(info.data.get("ref_slc"))
        return utils._get_path_from_gdal_str(ref_slc).parent

    @model_validator(mode="after")
    def _form_path(self) -> "VRTInterferogram":
        """Create the filename (if not provided) from the provided SLCs."""
        ref_slc, sec_slc = self.ref_slc, self.sec_slc

        fmt = self.date_format
        date1 = utils.get_dates(ref_slc, fmt=fmt)[0]
        date2 = utils.get_dates(sec_slc, fmt=fmt)[0]

        if self.path is not None:
            return self

        if self.outdir is None:
            # If outdir is not set, use the directory of the reference SLC
            self.outdir = utils._get_path_from_gdal_str(ref_slc).parent

        path = self.outdir / (io._format_date_pair(date1, date2, fmt) + ".vrt")
        if Path(path).exists():
            logger.info(f"Removing {path}")
            path.unlink()
        self.path = path
        return self

    @model_validator(mode="after")
    def _validate_files(self) -> "VRTInterferogram":
        """Check that the inputs are the same size and geotransform."""
        ref_slc = self.ref_slc
        sec_slc = self.sec_slc
        if not ref_slc or not sec_slc:
            # Skip validation if files are not set
            return self
        ds1 = gdal.Open(fspath(ref_slc))
        ds2 = gdal.Open(fspath(sec_slc))
        xsize, ysize = ds1.RasterXSize, ds1.RasterYSize
        xsize2, ysize2 = ds2.RasterXSize, ds2.RasterYSize
        if xsize != xsize2 or ysize != ysize2:
            raise ValueError(
                f"Input files {ref_slc} and {sec_slc} are not the same size"
            )
        gt1 = ds1.GetGeoTransform()
        gt2 = ds2.GetGeoTransform()
        if gt1 != gt2:
            raise ValueError(
                f"Input files {ref_slc} and {sec_slc} have different GeoTransforms"
            )

        return self

    def __init__(self, **data):
        """Create a VRTInterferogram object and write the VRT file."""
        super().__init__(**data)
        date1 = utils.get_dates(self.ref_slc, fmt=self.date_format)[0]
        date2 = utils.get_dates(self.sec_slc, fmt=self.date_format)[0]
        self.dates = (date1, date2)
        if self.write:
            self._write_vrt()

    def _write_vrt(self):
        xsize, ysize = io.get_raster_xysize(self.ref_slc)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            f.write(
                self._template.format(
                    xsize=xsize,
                    ysize=ysize,
                    ref_slc=self.ref_slc,
                    sec_slc=self.sec_slc,
                    pixel_function=self.pixel_function,
                )
            )
        io.copy_projection(self.ref_slc, self.path)

    def load(self):
        """Load the interferogram as a numpy array."""
        return io.load_gdal(self.path)

    @property
    def shape(self):
        xsize, ysize = io.get_raster_xysize(self.path)
        return (ysize, xsize)

    @classmethod
    def from_vrt_file(cls, path: Filename) -> "VRTInterferogram":
        """Load a VRTInterferogram from an existing VRT file.

        Parameters
        ----------
        path : Filename
            Path to VRT file.

        Returns
        -------
        VRTInterferogram
            VRTInterferogram object.

        """
        from dolphin.stack import VRTStack

        # Use the parsing function
        (ref_slc, sec_slc), subdataset = VRTStack._parse_vrt_file(path)
        if subdataset is not None:
            ref_slc = io.format_nc_filename(ref_slc, subdataset)
            sec_slc = io.format_nc_filename(sec_slc, subdataset)

        return cls.construct(
            ref_slc=ref_slc,
            sec_slc=sec_slc,
            path=Path(path).resolve(),
            subdataset=subdataset,
            date_format="%Y%m%d",
        )


class Network:
    """A network of interferograms from a list of SLCs.

    Attributes
    ----------
    slc_list : list[Filename]
        list of SLCs to use to form interferograms.
    slc_dates : list[datetime.date]
        list of dates corresponding to the SLCs.
    ifg_list : list[tuple[Filename, Filename]]
        list of `VRTInterferogram`s created from the SLCs.
    max_bandwidth : int | None, optional
        Maximum number of SLCs to include in an interferogram, by index distance.
        Defaults to None.
    max_temporal_baseline : Optional[float], optional
        Maximum temporal baseline to include in an interferogram, in days.
        Defaults to None.
    reference_idx : int | None, optional
        Index of the SLC to use as the reference for all interferograms.
        Defaults to None.
    """

    def __init__(
        self,
        slc_list: Sequence[Filename],
        outdir: Optional[Filename] = None,
        max_bandwidth: int | None = None,
        max_temporal_baseline: Optional[float] = None,
        reference_idx: int | None = None,
        indexes: Optional[Sequence[tuple[int, int]]] = None,
        subdataset: Optional[Union[str, Sequence[str]]] = None,
        write: bool = True,
    ):
        """Create a network of interferograms from a list of SLCs.

        Parameters
        ----------
        slc_list : list
            list of SLCs to use to form interferograms
        outdir : Optional[Filename], optional
            Directory to write the VRT files to.
            If not set, defaults to the directory of the reference SLC.
        max_bandwidth : int | None, optional
            Maximum number of SLCs to include in an interferogram, by index distance.
            Defaults to None.
        max_temporal_baseline : Optional[float]
            Maximum temporal baseline to include in an interferogram, in days.
            Defaults to None.
        reference_idx : int | None
            Index of the SLC to use as the reference for all interferograms.
            Defaults to None.
        indexes : Optional[Sequence[tuple[int, int]]]
            list of (ref_idx, sec_idx) pairs to use to create interferograms.
        subdataset : Optional[str]
            If passing NetCDF files in `slc_list, the subdataset of the image data
            within the file.
            Can also pass a sequence of one subdataset per entry in `slc_list`
            Defaults to None.
        write : bool
            Whether to write the VRT files to disk. Defaults to True.
        """
        self.slc_list, dates = utils.sort_files_by_date(slc_list)
        self.slc_file_pairs = self._make_ifg_pairs(
            self.slc_list,
            max_bandwidth=max_bandwidth,
            max_temporal_baseline=max_temporal_baseline,
            reference_idx=reference_idx,
            indexes=indexes,
        )
        # Save the parameters used to create the network
        self.slc_dates = [dates[0] for dates in dates]
        self.max_bandwidth = max_bandwidth
        self.max_temporal_baseline = max_temporal_baseline
        self.reference_idx = reference_idx
        if subdataset is None or isinstance(subdataset, str):
            self._slc_to_subdataset = {slc: subdataset for slc in slc_list}
        else:
            # We're passing a sequence
            assert len(subdataset) == len(slc_list)
            self._slc_to_subdataset = {slc: sd for slc, sd in zip(slc_list, subdataset)}
        self._subdataset = subdataset

        # Create each VRT file
        self.ifg_list: list[VRTInterferogram] = self._create_vrt_ifgs(
            outdir=outdir, write=write
        )

    def _create_vrt_ifgs(
        self, outdir: Optional[Filename] = Path.cwd(), write: bool = True
    ) -> list[VRTInterferogram]:
        """Write out a VRTInterferogram for each ifg.

        Parameters
        ----------
        outdir : Path, optional
            Directory to write the VRT files to.
            By default, the directory of the first SLC.
        write : bool, optional
            Whether to write the VRT files to disk.
        """
        if outdir is None:
            outdir = Path(self.slc_list[0]).parent
        ifg_list: list[VRTInterferogram] = []
        for ref, sec in self._gdal_file_strings:
            v = VRTInterferogram(ref_slc=ref, sec_slc=sec, outdir=outdir, write=write)
            ifg_list.append(v)
        return ifg_list

    @property
    def _gdal_file_strings(self):
        # format each file in each pair
        out = []
        for slc1, slc2 in self.slc_file_pairs:
            sd1 = self._slc_to_subdataset[slc1]
            sd2 = self._slc_to_subdataset[slc2]
            out.append(
                [io.format_nc_filename(slc1, sd1), io.format_nc_filename(slc2, sd2)]
            )
        return out

    def __repr__(self):
        return (
            f"Network(ifg_list={self.ifg_list}, slc_list={self.slc_list},"
            f" max_bandwidth={self.max_bandwidth},"
            f" max_temporal_baseline={self.max_temporal_baseline},"
            f" reference_idx={self.reference_idx})"
        )

    def __str__(self):
        return (
            f"Network of {len(self.ifg_list)} interferograms, "
            f"max_bandwidth={self.max_bandwidth}, "
            f"max_temporal_baseline={self.max_temporal_baseline}, "
            f"reference_idx={self.reference_idx}"
        )

    @staticmethod
    def _make_ifg_pairs(
        slc_list: Sequence[Filename],
        max_bandwidth: int | None = None,
        max_temporal_baseline: Optional[float] = None,
        reference_idx: int | None = None,
        indexes: Optional[Sequence[tuple[int, int]]] = None,
    ) -> list[tuple]:
        """Form interferogram pairs from a list of SLC files sorted by date."""
        if indexes is not None:
            # Give the option to select exactly which interferograms to create
            return [
                (slc_list[ref_idx], slc_list[sec_idx]) for ref_idx, sec_idx in indexes
            ]
        elif max_bandwidth is not None:
            return Network._limit_by_bandwidth(slc_list, max_bandwidth)
        elif max_temporal_baseline is not None:
            return Network._limit_by_temporal_baseline(slc_list, max_temporal_baseline)
        elif reference_idx is not None:
            return Network._single_reference_network(slc_list, reference_idx)
        else:
            raise ValueError("No valid ifg list generation method specified")

    @staticmethod
    def _single_reference_network(
        slc_file_list: Sequence[Filename], reference_idx=0
    ) -> list[tuple]:
        """Form a list of single-reference interferograms."""
        if len(slc_file_list) < 2:
            raise ValueError("Need at least two dates to make an interferogram list")
        ref = slc_file_list[reference_idx]
        ifgs = [tuple(sorted([ref, date])) for date in slc_file_list if date != ref]
        return ifgs

    @staticmethod
    def _limit_by_bandwidth(slc_file_list: Iterable[Filename], max_bandwidth: int):
        """Form a list of the "nearest-`max_bandwidth`" ifgs.

        Parameters
        ----------
        slc_file_list : Iterable[Filename]
            list of dates of SLCs
        max_bandwidth : int
            Largest allowed span of ifgs, by index distance, to include.
            max_bandwidth=1 will only include nearest-neighbor ifgs.

        Returns
        -------
        list
            Pairs of (date1, date2) ifgs
        """
        slc_to_idx = {s: idx for idx, s in enumerate(slc_file_list)}
        return [
            (a, b)
            for (a, b) in Network._all_pairs(slc_file_list)
            if slc_to_idx[b] - slc_to_idx[a] <= max_bandwidth
        ]

    @staticmethod
    def _limit_by_temporal_baseline(
        slc_file_list: Iterable[Filename],
        max_temporal_baseline: Optional[float] = None,
    ):
        """Form a list of the ifgs limited to a maximum temporal baseline.

        Parameters
        ----------
        slc_file_list : Iterable[Filename]
            Iterable of input SLC files
        max_temporal_baseline : float, optional
            Largest allowed span of ifgs, by index distance, to include.
            max_bandwidth=1 will only include nearest-neighbor ifgs.

        Returns
        -------
        list
            Pairs of (date1, date2) ifgs

        Raises
        ------
        ValueError
            If any of the input files have more than one date.
        """
        ifg_strs = Network._all_pairs(slc_file_list)
        slc_date_lists = [utils.get_dates(f) for f in slc_file_list]
        # Check we've got all single-date files
        if any(len(d) != 1 for d in slc_date_lists):
            raise ValueError(
                "Cannot form ifgs from multiple dates by temporal baseline."
            )
        slc_dates = [d[0] for d in slc_date_lists]

        ifg_dates = Network._all_pairs(slc_dates)
        baselines = [Network._temp_baseline(ifg) for ifg in ifg_dates]
        return [
            ifg for ifg, b in zip(ifg_strs, baselines) if b <= max_temporal_baseline
        ]

    @staticmethod
    def _all_pairs(slclist):
        """Create the list of all possible ifg pairs from slclist."""
        return list(itertools.combinations(slclist, 2))

    @staticmethod
    def _temp_baseline(ifg_pair):
        return (ifg_pair[1] - ifg_pair[0]).days

    def __len__(self):
        return len(self.ifg_list)

    def __getitem__(self, idx):
        return self.ifg_list[idx]

    def __iter__(self):
        return iter(self.ifg_list)

    def __contains__(self, item):
        return item in self.ifg_list

    def __eq__(self, other):
        return self.ifg_list == other.ifg_list


def estimate_correlation_from_phase(
    ifg: Union[VRTInterferogram, ArrayLike], window_size: Union[int, tuple[int, int]]
) -> np.ndarray:
    """Estimate correlation from only an interferogram (no SLCs/magnitudes).

    This is a simple correlation estimator that takes the (complex) average
    in a moving window in an interferogram. Used to get some estimate of interferometric
    correlation on the result of phase-linking interferograms.

    Parameters
    ----------
    ifg : Union[VRTInterferogram, ArrayLike]
        Interferogram to estimate correlation from.
        If a VRTInterferogram, will load and take the phase.
        If `ifg` is complex, will normalize to unit magnitude before estimating.
    window_size : Union[int, tuple[int, int]]
        Size of window to use for correlation estimation.
        If int, will use a square window of that size.
        If tuple, the rectangular window has shape  `size=(row_size, col_size)`.

    Returns
    -------
    np.ndarray
        Correlation array
    """
    if isinstance(ifg, VRTInterferogram):
        ifg = ifg.load()
    nan_mask = np.isnan(ifg)
    zero_mask = ifg == 0
    if not np.iscomplexobj(ifg):
        # If they passed phase, convert to complex
        inp = np.exp(1j * np.nan_to_num(ifg))
    else:
        # If they passed complex, normalize to unit magnitude
        inp = np.exp(1j * np.nan_to_num(np.angle(ifg)))

    # Note: the clipping is from possible partial windows producing correlation
    # above 1
    cor = np.clip(np.abs(utils.moving_window_mean(inp, window_size)), 0, 1)
    # Return the input nans to nan
    cor[nan_mask] = np.nan
    # If the input was 0, the correlation is 0
    cor[zero_mask] = 0
    return cor
