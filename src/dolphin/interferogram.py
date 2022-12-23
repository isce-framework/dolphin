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


class DerivedVRTInterferogram(BaseModel):
    """Form an interferogram using VRT Pixel Functions."""

    ref_slc: Union[str, Path] = Field(..., description="Path to reference SLC file")
    sec_slc: Union[str, Path] = Field(..., description="Path to secondary SLC file")
    outfile: Optional[Path] = Field(
        None,
        description=(
            "Path to output interferogram. Defaults to '<date1>_<date2>.vrt', where the"
            " dates are parsed from the input files, placed in the same directory as"
            " `ref_slc`."
        ),
    )
    pixel_func: str = "mul"

    date_format: str = "%Y%m%d"
    _template = """\
<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">
    <VRTRasterBand dataType="CFloat32" band="1" subClass="VRTDerivedRasterBand">
        <PixelFunctionType>{pixel_func}</PixelFunctionType>
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

    @validator("pixel_func")
    def _validate_pixel_func(cls, v):
        if v not in ["mul", "cmul"]:
            raise ValueError("pixel function must be 'mul' or 'cmul'")
        return v.lower()

    @validator("outfile", always=True)
    def _output_cant_exist(cls, v, values):
        if not v:
            # from the output file name from the dates within input files
            ref_slc, sec_slc = values.get("ref_slc"), values.get("sec_slc")
            fmt = values.get("date_format", "%Y%m%d")
            date1 = parse_slc_strings(ref_slc, fmt=fmt)
            date2 = parse_slc_strings(sec_slc, fmt=fmt)
            path = _get_path_from_gdal_str(ref_slc).parent
            v = path / f"{date1.strftime(fmt)}_{date2.strftime(fmt)}.vrt"
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

    # def __init__(self, **data):
    #     super().__init__(**data)
    def write(self):
        xsize, ysize = io.get_raster_xysize(self.ref_slc)
        with open(self.outfile, "w") as f:
            f.write(
                self._template.format(
                    xsize=xsize,
                    ysize=ysize,
                    ref_slc=self.ref_slc,
                    sec_slc=self.sec_slc,
                    pixel_func=self.pixel_func,
                )
            )
        io.copy_projection(self.ref_slc, self.outfile)

    def load(self):
        """Load the interferogram as a numpy array."""
        if not self.outfile.exists():
            self.write()
        return io.load_gdal(self.outfile)
