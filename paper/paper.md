---
title: 'dolphin: A Python phase linking package for large-scale InSAR PS/DS processing'
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
    orcid:
    affiliation: 1
  - name: Geoffrey M. Gunter
    orcid:
    affiliation: 1
  - name: Talib Oliver-Cabrera
    orcid:
    affiliation: 1
  - name: Emre Havazli
    orcid:
    affiliation: 1
  - name: Heresh Fattahi
    orcid:
    affiliation: 1

affiliations:
 - name: NASA Jet Propulsion Laboratory, California Institute of Technology
   index: 1
date: 5 March 2024
bibliography: paper.bib
---

# Summary

<!-- A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience. -->

`dolphin` is an open source Python library for processing Interferometric Synthetic Aperture Radar (InSAR) data using state-of-the-art algorithms.

The software has both a core library containing the InSAR algorithms and a set of command line tools for configuration, processing, and visualization. The core library is designed to be modular and extensible, allowing for easy integration of new algorithms and methods. The command line tools are designed to be user-friendly and to provide a simple interface for processing large sets of images at high resolutions.

<!-- JOSS welcomes submissions from broadly diverse research areas. For this reason, we require that authors include in the paper some sentences that explain the software functionality and domain of use to a non-specialist reader. We also require that authors explain the research applications of the software. The paper should be between 250-1000 words. Authors submitting papers significantly longer than 1000 words may be asked to reduce the length of their paper. -->


# Statement of need

The Sentinel-1 satellite from the European Space Agency (ESA) has been providing free and open access to Synthetic Aperture Radar (SAR) data since 2014. This has led to a rapid increase in the availability of SAR data, and has enabled a wide range of applications in Earth and environmental sciences. InSAR is a powerful technique for measuring surface deformation, and has been used to study a wide range of geophysical processes, including earthquakes, volcanic activity, and land subsidence. However, processing InSAR data can be challenging, particularly for large-scale time series analysis.

`dolphin` is designed to address this need by providing a flexible and extensible platform for processing large-scale InSAR data. The software is designed to be easy to use, and to provide a simple interface for processing large sets of images at high resolutions. The core library contains a range of state-of-the-art algorithms for processing InSAR data, including algorithms for phase linking, time series analysis, and persistent scatterer and distributed scatterer processing.

<!-- A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work. -->

## Related work

<!-- A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline. -->

<!-- Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it. -->

# Acknowledgements

Copyright (c) 2024 California Institute of Technology ("Caltech"). U.S. Government sponsorship acknowledged.

# References
