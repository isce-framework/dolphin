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

<!-- JOSS welcomes submissions from broadly diverse research areas. For this reason, we require that authors include in the paper some sentences that explain the software functionality and domain of use to a non-specialist reader. We also require that authors explain the research applications of the software. The paper should be between 250-1000 words. Authors submitting papers significantly longer than 1000 words may be asked to reduce the length of their paper. -->

<!-- A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience. -->
`dolphin` is a Python library for creating maps of land surface displacement using the remote sensing technique called Interferometric Synthetic Aperture Radar (InSAR). Conventional InSAR uses pairs of SAR images to get a single map of the relative displacement between the two acquisition times. `dolphin` uses state-of-the-art multi-temporal algorithms to reduce the impact of noise sources and produce long time series of displacement at fine resolution.

![Average surface displacement velocity along the radar line-of-sight between February, 2017 and December, 2020. Red (blue) indicates motion towards (away from) the satellite.\label{fig:mojave}](figures/bristol-velocity-sequential.png)

# Statement of need
<!-- A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work. -->

The Sentinel-1 satellite from the European Space Agency (ESA) has provided free and open access to Synthetic Aperture Radar (SAR) data since 2014. This has led to a rapid increase in the availability of SAR data and has enabled a wide range of applications in Earth and environmental sciences. However, processing InSAR data has remained a challenge for non-experts.
Advanced time series algorithms which combine persistent scatterer (PS) and distributed scatterer (DS) (also known as phase linking) have been developed for over a decade [@Guarnieri2008ExploitationTargetStatistics]; however, existing open source tools have not included these techniques until recently.
<!-- Moreoever, the available tools were generally not designed to run at continental scale in a cloud computing environment. -->

<!-- A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline. -->
<!-- While phase linking algorithms have been known for over a decade until several years ago, there were no open source libraries which could perform .  -->

<!-- The original software used in phase linking literature was not made public, but researchers have since made efforts to open source these advanced algorithms. -->
The first prototype was the [`FRInGE`](https://github.com/isce-framework/fringe) C++ library [@Fattahi2019FRInGEFullResolutionInSAR], which implements algorithms and workflows from [@Ferretti2011NewAlgorithmProcessing] and [@Ansari2018EfficientPhaseEstimation].
<!-- was created as a proof of concept for multiple phase linking algorithms   -->
<!-- `FRInGE`, a C++ library with Python bindings,  -->
The [`Miaplpy`](https://github.com/insarlab/MiaplPy) Python library contains a superset of the features in `FRInGE`, as well as new algorithms developed in [@Mirzaee2023NonlinearPhaseLinking]. The MATLAB [`TomoSAR`](https://github.com/DinhHoTongMinh/TomoSAR) library was also made public in 2022, which implements the "Compressed SAR" (ComSAR) algorithm- a variant of phase linking detailed in [@HoTongMinh2022CompressedSARInterferometry].

`dolphin` has been developed for the Observational Products for End-Users from Remote Sensing Analysis (OPERA) project. OPERA is a Jet Propulsion Laboratory project funded by the Satellite Needs Working Group (SNWG) tasked with generating a North American Surface Displacement product. This product is required to cover over 10 million square kilometers of land at 30 meter resolution or finer with under 72 hours of latency; as such, new software was required which could handle the heavy computational demands of the project.

# Overview of Dolphin

`dolphin` processes stacks of coregistered single-look complex (SLC) radar images into a time series of surface displacement. The software has pre-made workflows accessible through command line tools which call core algorithms for PS/DS processing:

- The `shp` subpackage estimates the SAR backscatter distribution to find neighborhoods of statistically homogeneous pixels (SHPs) using the generalized likelihood ratio test from @Parizzi2011AdaptiveInSARStack or the Kolmogorov-Smirnov test from @Ferretti2011NewAlgorithmProcessing.
- The `phase_link` subpackage processes the complex SAR covariance matrix into a time series of wrapped phase using the CAESAR algorithm [@Fornaro2015CAESARApproachBased], the eigenvalue-based maximum likelihood estimator of interferometric phase (EMI) [@Ansari2018EfficientPhaseEstimation], or the combined phase linking (CPL) approach from [@Mirzaee2023NonlinearPhaseLinking].
- The `unwrap` subpackage exposes multiple phase unwrapping algorithms, including the Statistical-cost, Network-flow Algorithm for Phase Unwrapping (SNAPHU) [@Chen2001TwodimensionalPhaseUnwrapping] and the PHASS algorithm (available in the InSAR Scientific Computing Environment [@Rosen2018InSARScientificComputing]).
- The `timeseries` module contains basic functionality to invert an overdetermined network of unwrapped interferograms into a time series and estimate the average surface velocity.

The outputs of `dolphin` are also compatible with the Miami INsar Time-series software for users who are already comfortable with MintPy [@Yunjun2019SmallBaselineInSAR].

While `dolphin` has been primarily tested on CPU-based workstations and cloud environments, the compute-intensive routines leverage the accelerated linear algebra (XLA) compiler within `jax` library  [@Bradbury2018JAXComposableTransformations]. This means that with only an extra installation, users can accelerate their processing by 5-20x on machines with a compatible GPU.

The `dolphin` command line tool provides an interface for running the end-to-end displacement workflow on large datasets (e.g. hundreds of full-frame Sentinel-1 images) with minimal required configuration.
For example, if a user has created a stack of coregistered SLCs in a `data/` directory, they only need to follow two steps to run the full workflow with all default parameters:

1. Configure the workflow with the `config` command, indicating the location of the SLCs, which dumps the output to a YAML file:

```python
dolphin config --slc-files data/*
```

2. Run the workflow saved in the YAML configuration file with the `run` command:

```python
dolphin run dolphin_config.yaml
```

\autoref{fig:mojave} shows an example result of the final average surface velocity map created by `dolphin`. The inputs were OPERA Coregistered Single-Look Complex (CSLC) geocoded images from Sentinel-1 data between February 2017 - December 2020 over the Mojave Desert.

# Acknowledgements

Copyright © 2024, California Institute of Technology ("Caltech"). U.S. Government sponsorship acknowledged.
The research was carried out at the Jet Propulsion Laboratory, California Institute of Technology, under a contract with the National Aeronautics and Space Administration (80NM0018D0004). The OPERA project has been funded by the Satellite Needs Working Group.

# References
