# Computational Cognitive Neuroscience Simulations

This repository contains the neural network simulation models for the [CCN Textbook](https://grey.colorado.edu/CompCogNeuro/index.php/CCNBook/Main).

These models are implemented in the new *Go* (golang) version of [emergent](https://github.com/emer/emergent), with Python versions available as well (note: not yet!).  This github repository contains the full source code and you can build and run the models by cloning the repository and building / running the individual projects as described in the emergent Wiki help page: [Wiki Install](https://github.com/emer/emergent/wiki/Install).

The simplest way to run the simulations is by downloading a `zip` file of all of the built models for your platform.  These are fully self-contained executable files and should "just work" on each platform.

* The full set of files are in the [Releases](https://github.com/CompCogNeuro/sims/releases) directory -- check there for files of the form `ccn_sims_<version>_<platform>.zip` where `<version>` is the version string (higher generally better), and `<platform>` is `mac`, `linux`, or `windows`.

# Usage

Each simulation has a `README` button, which directs your browser to open the corresponding `README.md` file on github.  This contains full step-by-step instructions for running the model, and questions to answer for classroom usage of the models.  See your syllabus etc for more info.

Use standard `Ctrl+` and `Ctrl-` key sequences to zoom the display to desired scale, and the GoGi preferences menu has an option to save the zoom (and various other options).

The main actions for running are in the `Toolbar` at the top, while the parameters of most relevance to the model are in the `Control panel` on the left.  Different output displays are selectable in the `Tabbed views` on the right of the window.

The [Go Emergent Wiki](https://github.com/emer/emergent/wiki/Home) contains various help pages for using things like the `NetView` that displays the network.

You can always access more detailed parameters by clicking on the button to the right off `Net` in the control panel (also by clicking on the layer names in the NetView), and custom params for this model are set in the `Params` field.

## Mac notes

You probably have to do a "right mouse click" (e.g., Ctrl + click) to open the executables in the `.zip` version -- it may be easier to just open the `Terminal` app, `cd` to the directory, and run the files from the command line directly.

# Status

**9/10/2019**: Chapters 2-4 are nearly complete and an initial binary test release is available.  Python versions will be made available pending a program to convert the go files to python more automatically.  Classes may need to depend on the C++ emergent versions for some gaps.



