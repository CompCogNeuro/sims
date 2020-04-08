# Computational Cognitive Neuroscience Simulations

This repository contains the neural network simulation models for the [CCN Textbook](https://github.com/CompCogNeuro/ed4) (now on github!).

These models are implemented in the new *Go* (golang) version of [emergent](https://github.com/emer/emergent), with Python versions available as well (note: not yet!).  This github repository contains the full source code and you can build and run the models by cloning the repository and building / running the individual projects as described in the emergent Wiki help page: [Wiki Install](https://github.com/emer/emergent/wiki/Install) -- see [Build From Source](#build-from-source) section below for specific step-by-step instructions.

The simplest way to run the simulations is by downloading a `zip` (or `tar.gz` for linux) file of all of the built models for your platform.  These are fully self-contained executable files and should "just work" on each platform.

* The full set of files are in the [Releases](https://github.com/CompCogNeuro/sims/releases) directory -- look under `Assets` for files of the form `ccn_sims_<version>_<platform>.zip` where `<version>` is the version string (higher generally better), and `<platform>` is `mac`, `linux`, or `windows`.  Here are links (based on last time this README was updated)

    + [ccn_sims_v1.0.3_mac.zip](https://github.com/CompCogNeuro/sims/releases/download/v1.0.3/ccn_sims_v1.0.3_mac.zip)
    + [ccn_sims_v1.0.3_windows.zip](https://github.com/CompCogNeuro/sims/releases/download/v1.0.3/ccn_sims_v1.0.3_windows.zip)
    + [ccn_sims_v1.0.3_linux.tar.gz](https://github.com/CompCogNeuro/sims/releases/download/v1.0.3/ccn_sims_v1.0.3_linux.tar.gz)

# Usage

Each simulation has a `README` button, which directs your browser to open the corresponding `README.md` file on github.  This contains full step-by-step instructions for running the model, and questions to answer for classroom usage of the models.  See your syllabus etc for more info.

Use standard `Ctrl+` and `Ctrl-` key sequences to zoom the display to desired scale, and the GoGi preferences menu has an option to save the zoom (and various other options).

The main actions for running are in the `Toolbar` at the top, while the parameters of most relevance to the model are in the `Control panel` on the left.  Different output displays are selectable in the `Tabbed views` on the right of the window.

The [Go Emergent Wiki](https://github.com/emer/emergent/wiki/Home) contains various help pages for using things like the `NetView` that displays the network.

You can always access more detailed parameters by clicking on the button to the right off `Net` in the control panel (also by clicking on the layer names in the NetView), and custom params for this model are set in the `Params` field.

## Mac notes

You probably have to do a "right mouse click" (e.g., Ctrl + click) to open the executables in the `.zip` version -- it may be easier to just open the `Terminal` app, `cd` to the directory, and run the files from the command line directly.

# Status

**3/30/2020**: Version 1.0.3 release -- no major changes, just updated to most recent GoGi GUI.

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

* `pvlv`: Pavlovian Conditioning with the PVLV model (Questions **7.7 -- 7.9**)  **NOT YET AVAIL!**

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
$ ./objrec &     # this will run the newly-build executable
```

All the dependencies (emergent packages, gogi gui packages, etc) will be installed in:
```
~/go/pkg/mod/github.com/
```
where `~` means your home directory (can also be changed by setting `GOPATH` to any directory).

