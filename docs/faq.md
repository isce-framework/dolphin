## Frequently Asked Questions

### 1. What's the sign convention of the line-of-sight data?

**Summary**:

- In the `timeseries/` folder, positive means motion *toward* the sensor, and negative means motion *away* from the sensor.
- In the `interferograms/` folder, the sign is flipped and positive phase means motion *away* from the sensor.

`dolphin` follows similar sign conventions to other InSAR software such as isce2 and MintPy.
Inteferograms in `dolphin` are formed as `reference_slc * conj(secondary_slc)`. The phase of the complex values in the `interferograms/` folder indicates an increase in the line-of-sight (LOS) direction from the sensor, while a negative value indicates a decrease (motion towards the sensor).

After unwrapping, the sign convention is flipped and converted to meters (if the `wavelength` parameter is set) for the outputs in the `timeseries` folder.
Assuming the `wavelength` parameter is set, the unwrapped phase is multiplied by $-4\pi / \lambda$, so positive values in the timeseries rasters indicate motion *toward* the sensor, and negative values indicate motion *away* from the sensor. Thus, when combining two LOS rasters from ascending and descending geometries, postive values in both indicate that uplift is occurring.

Note that `dolphin` is able to process coregistered SLCs from arbitrary processing software, and does not always attempt to guess what sensor the SLCs are derived from. If the `wavelength` (within the `input_options` group) is not set in during configuration, the outputs in the `timeseries` folder will remain in radians, but keep the same sign convention (positive = motion toward the sensor).
