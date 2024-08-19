# Computational Cognitive Neuroscience Simulations

This repository contains the neural network simulation models for the [CCN Textbook](https://compcogneuro.org). For more information, see the [simulations website](https://compcogneuro.org/simulations).

## Status

* **August 2024**: The sims are now being updated to run on the web as documented on the website.

* **Feb 15, 2023**: Version 1.3.3 release: updated to improved Vulkan driver selection in latest GoGi.

* **Sept 15, 2022**: Version 1.3.2 release: updated to new NetView with raster view and separate weight recording.

* **Sept 9, 2021**: Version 1.3.1 release: bug fixes, deep leabra version of sg, python works on windows.

* **Nov 23, 2020**: Version 1.2.2 release: full set of Python versions and the pvlv model.

* See https://github.com/CompCogNeuro/sims/releases for full history

## Developer notes

*This is not relevant for regular users*

The Makefile contains targets that build all the sims programs and copy the resulting executable into a consolidated directory `~/ccnsimpkg/` which can then be used to make the .zip / .tar files for distribution purposes.  The targets are: `mac`, `linux`, `windows`.

To build all `windows` targets using Makefile's on Windows (i.e., `make windows`), you have to use cygwin with native make installed -- could not get recursive invocation of make to work in powershell.  Also have to `mv /usr/bin/gcc.exe /usr/bin/gcc-cyg.exe` so it will use `TDM-GCC-64` version -- otherwise it won't build.
