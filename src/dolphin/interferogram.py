"""Combine estimated DS phases with PS phases to form interferograms."""
from os import fspath
from pathlib import Path
from typing import Optional, Union

from osgeo import gdal
from pydantic import BaseModel, Field, root_validator, validator

from dolphin import io
from dolphin.log import get_log
from dolphin.utils import _get_path_from_gdal_str, _resolve_gdal_path, parse_slc_strings

gdal.UseExceptions()

logger = get_log()


class VRTInterferogram(BaseModel):
    """Create an interferogram using a VRTDerivedRasterBand.

    Attributes
    ----------
    ref_slc : Union[str, Path]
        Path to reference SLC file
    sec_slc : Union[str, Path]
        Path to secondary SLC file
    outdir : Optional[Path], optional
        Directory to place output interferogram. Defaults to the same directory as
        `ref_slc`. If only `outdir` is specified, the output interferogram will
        be named '<date1>_<date2>.vrt', where the dates are parsed from the
        inputs. If `outfile` is specified, this is ignored.
    outfile : Optional[Path], optional
        Path to output interferogram. Defaults to '<date1>_<date2>.vrt',
        placed in the same directory as `ref_slc`.
    date_format : str, optional
        Date format to use when parsing dates from the input files.
        Defaults to '%Y%m%d'.
    pixel_function : str, optional
        GDAL Pixel function to use, choices={'cmul', 'mul'}.
        Defaults to 'cmul', which performs `ref_slc * sec_slc.conj()`.
        See https://gdal.org/drivers/raster/vrt.html#default-pixel-functions

    """

    ref_slc: Union[str, Path] = Field(..., description="Path to reference SLC file")
    sec_slc: Union[str, Path] = Field(..., description="Path to secondary SLC file")
    outdir: Optional[Path] = Field(
        None,
        description=(
            "Directory to place output interferogram. Defaults to the same directory as"
            " `ref_slc`. If only `outdir` is specified, the output interferogram will"
            " be named '<date1>_<date2>.vrt', where the dates are parsed from the"
            " inputs. If `outfile` is specified, this is ignored."
        ),
    )
    outfile: Optional[Path] = Field(
        None,
        description=(
            "Path to output interferogram. Defaults to '<date1>_<date2>.vrt', where the"
            " dates are parsed from the input files, placed in the same directory as"
            " `ref_slc`."
        ),
    )
    date_format: str = "%Y%m%d"

    pixel_function: str = "cmul"
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

    @validator("ref_slc", "sec_slc")
    def _check_gdal_string(cls, v):
        # First make sure it's openable
        try:
            gdal.Info(fspath(v))
        except RuntimeError:
            raise ValueError(f"File {v} is not a valid GDAL dataset")
        # Then, if we passed a string like 'NETCDF:"file.nc":band', make sure
        # the file is absolute
        if ":" in str(v):
            try:
                v = _resolve_gdal_path(v)
            except Exception:
                # if the file had colons for some reason but
                # it didn't match, just ignore
                pass
        return v

    @validator("pixel_function")
    def _validate_pixel_func(cls, v):
        if v not in ["mul", "cmul"]:
            raise ValueError("pixel function must be 'mul' or 'cmul'")
        return v.lower()

    @validator("outdir", always=True)
    def _check_output_dir(cls, v, values):
        if v is not None:
            return Path(v)
        # If outdir is not set, use the directory of the reference SLC
        ref_slc = values.get("ref_slc")
        return _get_path_from_gdal_str(ref_slc).parent

    @validator("outfile", always=True)
    def _output_cant_exist(cls, v, values):
        if not v:
            # from the output file name from the dates within input files
            ref_slc, sec_slc = values.get("ref_slc"), values.get("sec_slc")
            fmt = values.get("date_format", "%Y%m%d")
            date1 = parse_slc_strings(ref_slc, fmt=fmt)
            date2 = parse_slc_strings(sec_slc, fmt=fmt)

            outdir = values.get("outdir")
            v = outdir / f"{date1.strftime(fmt)}_{date2.strftime(fmt)}.vrt"
        elif Path(v).exists():
            raise ValueError(f"Output file {v} already exists")
        return v

    @root_validator
    def _validate_files(cls, values):
        """Check that the inputs are the same size and geotransform."""
        ref_slc = values.get("ref_slc")
        sec_slc = values.get("sec_slc")
        if not ref_slc or not sec_slc:
            # Skip validation if files are not set
            return values
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

        return values

    def __init__(self, **data):
        """Create a VRTInterferogram object and write the VRT file."""
        super().__init__(**data)
        self._write()

    def _write(self):
        xsize, ysize = io.get_raster_xysize(self.ref_slc)
        with open(self.outfile, "w") as f:
            f.write(
                self._template.format(
                    xsize=xsize,
                    ysize=ysize,
                    ref_slc=self.ref_slc,
                    sec_slc=self.sec_slc,
                    pixel_function=self.pixel_function,
                )
            )
        io.copy_projection(self.ref_slc, self.outfile)

    def load(self):
        """Load the interferogram as a numpy array."""
        return io.load_gdal(self.outfile)

    @property
    def shape(self):
        xsize, ysize = io.get_raster_xysize(self.outfile)
        return (ysize, xsize)
