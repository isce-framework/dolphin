---
title: "Dolphin: A Python package for large-scale InSAR PS/DS processing"
tags:
  - Python
  - InSAR
  - remote sensing
  - phase linking
  - time series
  - persistent scatterers
  - distributed scatterers
  - interferometry
  - surface deformation
  - earthquake
authors:
  - name: Scott J. Staniewicz
    orcid: 0000-0003-0872-7098
    affiliation: 1
  - name: Sara Mirzaee
    orcid: 0000-0001-8194-5951
    affiliation: 1
  - name: Geoffrey M. Gunter
    orcid: 0000-0003-4612-0887
    affiliation: 1
  - name: Talib Oliver-Cabrera
    orcid: 0000-0002-2315-4710
    affiliation: 1
  - name: Emre Havazli
    orcid: 0000-0002-1236-7067
    affiliation: 1
  - name: Heresh Fattahi
    orcid: 0000-0001-6926-4387
    affiliation: 1
affiliations:
  - name: Jet Propulsion Laboratory, California Institute of Technology
    index: 1
date: 5 March 2024
bibliography: references.bib
---

# Summary

Interferometric Synthetic Aperture Radar (InSAR) is a remote sensing technique used for measuring land surface deformation.
Conventional InSAR uses pairs of SAR images to get a single map of the relative displacement between the two acquisition times.
Dolphin is a Python library which uses state-of-the-art multi-temporal algorithms to reduce the impact of noise sources and produce long time series of displacement at fine resolution.

![Average surface displacement velocity along the radar line-of-sight between February, 2017 and December, 2020. Red (blue) indicates motion towards (away from) the satellite.\label{fig:mojave}](figures/bristol-velocity-sequential.png)

# Statement of need

InSAR has been a powerful tool for decades, both in geophysical studies including tectonics, volcanism, and glacier dynamics, as well as human applications such as urban development, mining, and groundwater extraction. The launch of the European Space Agency's Sentinel-1 satellite in 2014 dramatically increased the availability of free, open-access SAR data. However, processing InSAR data remains challenging, particularly for non-experts.

Advanced algorithms combining persistent scatterer (PS) and distributed scatterer (DS) techniques, also known as phase linking, have been developed over the past decade to help overcome decorrelation noise in longer time series [@Guarnieri2008ExploitationTargetStatistics]. Despite their potential, these methods have only recently begun to appear in open-source tools.

The phase linking first prototype was the [FRInGE](https://github.com/isce-framework/fringe) C++ library [@Fattahi2019FRInGEFullResolutionInSAR], which implements algorithms and workflows from @Ferretti2011NewAlgorithmProcessing and @Ansari2018EfficientPhaseEstimation. The [MiaplPy](https://github.com/insarlab/MiaplPy) Python library contains a superset of the features in FRInGE, as well as new algorithms developed in @Mirzaee2023NonlinearPhaseLinking. Additionally, the MATLAB [TomoSAR](https://github.com/DinhHoTongMinh/TomoSAR) library was made public in 2022, which implements the "Compressed SAR" (ComSAR) algorithm, a variant of phase linking detailed in @HoTongMinh2022CompressedSARInterferometry.

While these tools represent significant progress, there remained a need for software capable of handling the heavy computational demands of large-scale InSAR processing. For example, the TomoSAR library currently requires tens of gigabytes of memory to process more than a small area of interest, while FRInGE and MiaplPy are unable to offer speedups to users who want to process data at a coarser output grid than the full SLC resolution. Additionally, both FRInGE and MiaplPy were designed to process single batches of SLC images.

Dolphin was developed to process both historical archives and incrementally handle new data in near-real time. This capability was specifically designed for the Observational Products for End-Users from Remote Sensing Analysis (OPERA) project. OPERA, a Jet Propulsion Laboratory project funded by the Satellite Needs Working Group (SNWG), is tasked with generating a North American Surface Displacement product covering over 10 million square kilometers of land at 30 meter resolution or finer, with under 72 hours of latency.

# Overview of Dolphin

Dolphin processes coregistered single-look complex (SLC) radar images into a time series of surface displacement. The software has an end-to-end surface displacement processing workflow (\autoref{fig:overview}), accessible through a command line tool, which calls core algorithms for PS/DS processing:

- The `shp` subpackage estimates the SAR backscatter distribution to find neighborhoods of statistically homogeneous pixels (SHPs) using the generalized likelihood ratio test from @Parizzi2011AdaptiveInSARStack or the Kolmogorov-Smirnov test from @Ferretti2011NewAlgorithmProcessing.
- The `phase_link` subpackage processes the complex SAR covariance matrix into a time series of wrapped phase using the CAESAR algorithm [@Fornaro2015CAESARApproachBased], the eigenvalue-based maximum likelihood estimator of interferometric phase (EMI) [@Ansari2018EfficientPhaseEstimation], or the combined phase linking (CPL) approach from @Mirzaee2023NonlinearPhaseLinking.
- The `ps` module selects persistent scatterer pixels from the full-resolution SLCs to be integrated into the wrapped interferograms [@Ferretti2001PermanentScattersSAR].
- The `unwrap` subpackage exposes multiple phase unwrapping algorithms, including the Statistical-cost, Network-flow Algorithm for Phase Unwrapping (SNAPHU) [@Chen2001TwodimensionalPhaseUnwrapping], the PHASS algorithm (available in the InSAR Scientific Computing Environment [@Rosen2018InSARScientificComputing]), and the Extended Minimum Cost Flow (EMCF) 3D phase unwrapping algorithm via the `spurt` library. Dolphin has pre- and post-processing options, including Goldstein filtering [@Goldstein1998RadarInterferogramFiltering] or interferogram masking and interpolation [@Chen2015PersistentScattererInterpolation].
- The `timeseries` module contains basic functionality to invert an overdetermined network of unwrapped interferograms into a time series and estimate the average surface velocity. The outputs of Dolphin are also compatible with the Miami INsar Time-series software for users who are already comfortable with MintPy [@Yunjun2019SmallBaselineInSAR].

To meet the computational demands of large-scale InSAR processing, Dolphin leverages Just-in-time (JIT) compilation, maintaining the readability of Python while matching the speed of compiled languages. The software's compute-intensive routines use the XLA compiler within JAX [@Bradbury2018JAXComposableTransformations] for efficient CPU or GPU processing. Users with compatible GPUs can see 5-20x speedups by simply installing additional packages. Dolphin manages memory efficiently through batch processing and multi-threaded I/O, allowing it to handle datasets larger than available memory while typically using a few gigabytes for most processing stages. These optimizations enable Dolphin to process hundreds of full-frame Sentinel-1 images with minimal configuration, making it well-suited for large-scale projects such as OPERA.

![Overview of main workflow to generate surface displacement. Rectangular stacks indicate input or intermediate raster images. Arrows show the flow of data through the configurable submodules of Dolphin.\label{fig:overview}](figures/dolphin-modules.pdf)

The Dolphin command line tool provides an interface for running the end-to-end displacement workflow. To illustrate, if a user has created a stack of coregistered SLCs in a `data/` directory, they only need to follow two steps to run the full workflow with all default parameters:

1. Configure the workflow with the `config` command, indicating the location of the SLCs, which dumps the output to a YAML file:

```python
dolphin config --slc-files data/*
```

2. Run the workflow saved in the YAML configuration file with the `run` command:

```python
dolphin run dolphin_config.yaml
```

The full set of configuration options can be viewed with the `dolphin config --print-empty` command.

\autoref{fig:mojave} shows an example result of the final average surface velocity map created by Dolphin. The inputs were OPERA Coregistered Single-Look Complex (CSLC) geocoded images from Sentinel-1 data between February 2017 - December 2020 over the Mojave Desert.

# Acknowledgements

Copyright © 2024, California Institute of Technology ("Caltech"). U.S. Government sponsorship acknowledged.
The research was carried out at the Jet Propulsion Laboratory, California Institute of Technology, under a contract with the National Aeronautics and Space Administration (80NM0018D0004). OPERA, managed by the Jet Propulsion Laboratory and funded by the Satellite Needs Working Group, is creating remote sensing products to address Earth observation needs across U.S. civilian federal agencies.

# References
