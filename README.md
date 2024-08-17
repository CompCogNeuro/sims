# Computational Cognitive Neuroscience Simulations

This repository contains the neural network simulation models for the [CCN Textbook](https://CompCogNeuro.org), managed on a [GitHub Repository](https://github.com/CompCogNeuro/ed4).

To run these simulations on your computer, it is easiest to download the full set of executable programs for the type of computer you are using (Apple Mac, Microsoft Windows, or Linux):

* **Mac Apple Silicon:** (updated to support Sonoma (14) [ccn_sims_v1.3.5_mac-m1.zip](https://github.com/CompCogNeuro/sims/releases/download/v1.3.5/ccn_sims_v1.3.5_mac-m1.zip)
    + **IMPORTANT:** For both of the Mac packages, you must also download and install **version 1.3.261.1** ~~the latest version~~ of Vulkan SDK: https://sdk.lunarg.com/sdk/download/1.3.261.1/mac/vulkansdk-macos-1.3.261.1.dmg (from https://vulkan.lunarg.com/sdk/home, version dated 30-Aug-2023) -- click on `Select All` to select what to install, and otherwise just accept the defaults for all prompts. (latest version 1.3.268.1 has a bug: https://github.com/goki/vgpu/issues/9)  
   
* **Mac Intel:** [ccn_sims_v1.3.5_mac-intel.zip](https://github.com/CompCogNeuro/sims/releases/download/v1.3.5/ccn_sims_v1.3.5_mac-intel.zip) (see note above about installing vulkan-sdk)

* **Windows:** [ccn_sims_v1.3.3_windows.zip](https://github.com/CompCogNeuro/sims/releases/download/v1.3.3/ccn_sims_v1.3.3_windows.zip)

* **Linux:** [ccn_sims_v1.3.3_linux.tar.gz](https://github.com/CompCogNeuro/sims/releases/download/v1.3.3/ccn_sims_v1.3.3_linux.tar.gz)

If you have an older computer that does not work the above releases, try going into the older Releases -- click on the github `Releases` link to see.

Alternatively, you can use the [Python](#python) version -- see instructions at that link for how to install.

# Usage

Each simulation has a `README` button, which directs your browser to open the corresponding `README.md` file on github.  This contains full step-by-step instructions for running the model, and questions to answer for classroom usage of the models.  See your syllabus etc for more info.

Use standard `Ctrl+` and `Ctrl-` key sequences to zoom the display to desired scale, and the GoGi preferences menu has an option to save the zoom (and various other options).

The main actions for running are in the `Toolbar` at the top, while the parameters of most relevance to the model are in the `Control panel` on the left.  Different output displays are selectable in the `Tabbed views` on the right of the window.

The [Go Emergent Wiki](https://github.com/emer/emergent/wiki/Home) contains various help pages for using things like the `NetView` that displays the network.

You can always access more detailed parameters by clicking on the button to the right off `Net` in the control panel (also by clicking on the layer names in the NetView), and custom params for this model are set in the `Params` field.

## Mac notes

If double-clicking on the program doesn't work (error message about unsigned application or verified developer -- google "mac unsigned application" for more information), you may have to do a "right mouse click" (e.g., Ctrl + click) to open the executables in the `.zip` version -- it may be easier to just open the `Terminal` app, `cd` to the directory, and run the files from the command line directly, although apparently more recent mac versions will still complain so you need to navigate your Finder to folder created from the .zip file and click on one of the applications and Open it, and then others should be OK.

# Info

These models are implemented in the *Go* (golang) version of [emergent](https://github.com/emer/emergent), with [Python](#python) versions available as well.  This github repository contains the full source code and you can build and run the models by cloning the repository and building / running the individual projects as described in the [Build From Source](#build-from-source) section below for specific step-by-step instructions.

Older releases can be found in the [Releases](https://github.com/CompCogNeuro/sims/releases) directory -- look under `Assets` for files of the form `ccn_sims_<version>_<platform>.zip` where `<version>` is the version string (higher generally better), and `<platform>` is `mac`, `linux`, or `windows`.

The even older [C++ emergent (cemer)](https://github.com/emer/cemer) sims project files are available here:  [cecn_8_5_2.zip](https://github.com/CompCogNeuro/sims/releases/download/v1.2.2/cecn_8_5_2.zip) (no longer updated or supported -- recommend transitioning to new ones)
    
# Status

* **Feb 15, 2023**: Version 1.3.3 release: updated to improved Vulkan driver selection in latest GoGi.

* **Sept 15, 2022**: Version 1.3.2 release: updated to new NetView with raster view and separate weight recording.

* **Sept 9, 2021**: Version 1.3.1 release: bug fixes, deep leabra version of sg, python works on windows.

* **Nov 23, 2020**: Version 1.2.2 release: full set of Python versions and the pvlv model.

* See https://github.com/CompCogNeuro/sims/releases for full history

# List of Sims and Exercise Questions

Here's a full list of all the simulations and the textbook exercise questions associated with them:

## Chapter 2: Neuron

* `neuron`: Integration, spiking and rate code activation. (Questions **2.1 -- 2.7**)

* `detector`: The neuron as a detector -- demonstrates the critical function of synaptic weights in determining what a neuron detects. (Questions **2.8 -- 2.10**)

## Chapter 3: Networks

* `face_categ`: Face categorization, including bottom-up and top-down processing (used for multiple explorations in Networks chapter) (Questions **3.1 -- 3.3**)

* `cats_dogs`: Constraint satisfaction in the Cats and Dogs model. (Question **3.4**)

* `necker_cube`: Constraint satisfaction and the role of noise and accommodation in the Necker Cube model. (Question **3.5**)

* `inhib`: Inhibitory interactions via inhibitory interneurons, and FFFB approximation. (Questions **3.6 -- 3.8**)

## Chapter 4: Learning

* `self_org`: Self organizing learning using BCM-like dynamic of XCAL (Questions **4.1 -- 4.2**).

* `pat_assoc`: Basic two-layer network learning simple input/output mapping tasks (pattern associator) with Hebbian and Error-driven mechanisms (Questions **4.3 -- 4.6**).

* `err_driven_hidden`: Full error-driven learning with a hidden layer, can solve any input output mapping (Question **4.7**).

* `family_trees`: Learning in a deep (multi-hidden-layer) network, showing advantages of combination of self-organizing and error-driven learning (Questions **4.8 -- 4.9**).

* `hebberr_combo`: Hebbian learning in combination with error-driven facilitates generalization (Questions **4.10 -- 4.12**).

Note: no sims for chapter 5

## Chapter 6: Perception and Attention

* `v1rf`: V1 receptive fields from Hebbian learning, with lateral topography. (Questions **6.1 -- 6.2**)

* `objrec`: Invariant object recognition over hierarchical transforms. (Questions **6.3 -- 6.5**)

* `attn`: Spatial attention interacting with object recognition pathway, in a small-scale model. (Questions **6.6 -- 6.11**)

## Chapter 7: Motor Control and Reinforcement Learning

* `bg`: Action selection / gating and reinforcement learning in the basal ganglia. (Questions **7.1 -- 7.4**)

* `rl_cond`: Pavlovian Conditioning using Temporal Differences Reinforcement Learning. (Questions **7.5 -- 7.6**)

* `pvlv`: Pavlovian Conditioning with the PVLV model (Questions **7.7 -- 7.9**)

* `cereb`: Cerebellum role in motor learning, learning from errors. (Questions **7.10 -- 7.11**) **NOT YET AVAIL!**

## Chapter 8: Learning and Memory

* `abac`: Paired associate AB-AC learning and catastrophic interference. (Questions **8.1 -- 8.3**)

* `hip`: Hippocampus model and overcoming interference. (Questions **8.4 -- 8.6**)

* `priming`: Weight and Activation-based priming. (Questions **8.7 -- 8.8**)

## Chapter 9: Language

* `dyslex`: Normal and disordered reading and the distributed lexicon. (Questions **9.1 -- 9.6**)

* `ss`: Orthography to Phonology mapping and regularity, frequency effects. (Questions **9.7 -- 9.8**)

* `sem`: Semantic Representations from World Co-occurrences and Hebbian Learning. (Questions **9.9 -- 9.11**)

* `sg`:  The Sentence Gestalt model. (Question **9.12**)

## Chapter 10: Executive Function

* `stroop`: The Stroop effect and PFC top-down biasing (Questions **10.1 -- 10.3**)

* `a_not_b`: Development of PFC active maintenance and the A-not-B task (Questions **10.4 -- 10.6**)

* `sir`: Store/Ignore/Recall Task - Updating and Maintenance in more complex PFC model (Questions **10.7 -- 10.8**)

# Python

**Note:** version 1.3.2 does not work with python at this point.  The previous 1.3.1 version should still work.

Running the sims under Python uses a compiled version of the underlying Go-based simulation infrastructure (i.e., all of [emer](https://github.com/emer) and all of [GoGi](https://github.com/goki/gi) ) that links in a specific version of Python, in the form of an executable file named `pyleabra`.  The pyleabra executable is just like a `python3` executable in all other respects.

Because it is built with a specific version of python3 baked in, you may want to build your own version of this executable based on the version of python that you use for your other work, in which case see the instructions at: [leabra python](https://github.com/emer/leabra/tree/master/python).  Also, there can be various library path issues for finding the python library that the executable is linked against -- the install process attempts to ensure that your machine has the same version ours was built from.

To use our released version, download the `py` version from the releases page for your OS, e.g.,:

*  [ccn_py_sims_v1.3.1_mac.zip](https://github.com/CompCogNeuro/sims/releases/download/v1.3.1/ccn_py_sims_v1.3.1_mac.zip)
*  [ccn_py_sims_v1.3.1_linux.tar.gz](https://github.com/CompCogNeuro/sims/releases/download/v1.3.1/ccn_py_sims_v1.3.1_linux.tar.gz)
*  [ccn_py_sims_v1.3.1_windows.zip](https://github.com/CompCogNeuro/sims/releases/download/v1.3.1/ccn_py_sims_v1.3.1_windows.zip)

un-zip / un-tar that file (e.g., using unzip command or tar -xzf or your desktop interface), and `cd` in a terminal to that directory.

The `README.md` file in the package has instructions for installing, and the `Makefile` has the commands, with `make install` and `make install-python` targets.  Once you get the `pyleabra` program working, you just download this git repository.

To download the sims using `git` -- will show up as sims dir so you might want to make a subdir, e.g.:

```sh
$ mkdir ~/ccnsims
$ cd ~/ccnsims
$ git clone https://github.com/CompCogNeuro/sims
```

Then you can go to the location of the sims source, and just run the .py executables, e.g.,

```sh
$ cd ~/ccnsims/sims/ch2/neuron
$ ./neuron.py
```

## Installing other python packages

As noted above, `pyleabra` is built with a specific version of python (e.g., 3.8.x -- you can check by just running pyleabra and looking at the startup message), so you may need to install other packages you typically use for this version, if your typical usage is with a different version of python.  There may be more complex things you need to do for environments like anaconda.  e.g., here's how you would install numpy and pandas:

```bash
$ pyleabra -m pip install numpy pandas
```

# Build From Source

First, you *must* read and follow the [GoGi Install](https://github.com/goki/gi/wiki/Install) instructions, and build the `examples/widgets` example and make sure it runs -- that page has all the details for extra things needed for different operating systems.

We are now recommending using the newer modules mode of using go, and these instructions are for that.

**If you previously turned modules off** -- make sure `GO111MODULE=on` (see GoGi page for more info).

(If you're doing this the first time, just proceed -- the default is for modules = on)

The `#` notes after each line are comments explaining the command -- don't type those!

```bash
$ cd <wherever you want to install>  # change to directory where you want to install
$ git clone https://github.com/CompCogNeuro/sims   # get the code, makes a sims dir
$ cd sims        # go into it
$ cd ch6/objrec  # this has the most dependencies -- test it
$ go build       # this will get all the dependencies and build everything
$ ./objrec &     # this will run the newly build executable
```

All the dependencies (emergent packages, gogi gui packages, etc) will be installed in:
```
~/go/pkg/mod/github.com/
```
where `~` means your home directory (can also be changed by setting `GOPATH` to any directory).

# Dev Notes

*This is not relevant for regular users of the compiled executables or python versions*

The Makefile contains targets that build all the sims programs and copy the resulting executable into a consolidated directory `~/ccnsimpkg/` which can then be used to make the .zip / .tar files for distribution purposes.  The targets are: `mac`, `linux`, `windows`.

To build all `windows` targets using Makefile's on Windows (i.e., `make windows`), you have to use cygwin with native make installed -- could not get recursive invocation of make to work in powershell.  Also have to `mv /usr/bin/gcc.exe /usr/bin/gcc-cyg.exe` so it will use `TDM-GCC-64` version -- otherwise it won't build.

