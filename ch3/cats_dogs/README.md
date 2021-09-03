Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

This project explores a simple **semantic network** intended to represent a (very small) set of relationships among different features used to represent a set of entities in the world.  In our case, we represent some features of cats and dogs: their color, size, favorite food, and favorite toy. The network contains information about a number of individual cats and dogs, and is able to use this information to make *generalizations* about what cats and dogs in general have in common, and what is unique about them. It can also tell you about the *consistency* of a given feature with either cats or dogs -- this is where the harmony function can be useful in assessing the total constraint satisfaction level of the network with a particular configuration of feature inputs. The network can also be used to perform *pattern completion* as a way of retrieving information about a particular individual or individuals. Thus, this simple network summarizes many of the topics covered in this chapter.

Cats and Dogs Semantics:

| Species | Name      | Color         | Size   | Food   | Toy     |
|---------|-----------|---------------|--------|--------|---------|
| Cat     | Morris    | Orange        | Small  | Grass  | String  |
|         | Socks     | Black & White | Small  | Bugs   | Feather |
|         | Sylvester | Black & White | Small  | Grass  | String  |
|         | Garfield  | Orange        | Medium | Scraps | String  |
|         | Fuzzy     | White         | Medium | Grass  | Feather |
| Dog     | Rex       | Black         | Large  | Scraps | Bone    |
|         | Fido      | Brown         | Medium | Shoe   | Shoe    |
|         | Spot      | Black & White | Medium | Scraps | Bone    |
|         | Snoopy    | Black & White | Medium | Scraps | Bone    |
|         | Butch     | Brown         | Large  | Shoe   | Shoe    |

The knowledge embedded in the network is summarized in the above table. This knowledge is encoded by simply setting a weight of 1 between an *instance* (`Identity`) node representing an individual cat or dog and thecorresponding feature value that this individual possesses (c.f., the Jets and Sharks model from McClelland & Rumelhart, 1988). Each of the groups of features (i.e., values within one column of the table) are represented within distinct layers that have their own within-layer inhibition. In additon, all of the identity units and the name units are within their own separate layers as well. We use the `FFFB` inhibitory function here which allows considerable flexibility in the actual number of active units per layer.

# Exploration

* As usual, take some time to examine the weights in the network, and verify that the weights implement the knowledge shown in the table. To do so, select the `r.Wt` value in the `NetView` and then click on individual neurons in the different layers. 

Let's first verify that when we present an individual's name as input, it will recall all of the information about that individual. This is a form of pattern completion with a single unique input cue. 

* Press `Test Trial` to present the default input patterns to the network, which activates the `Morris` name unit.

You should see that the network activates the appropriate features for Morris. You can think about this process as finding the most harmonious activation state given the input constraint of Morris, and the constraints in the network's weights. Equivalently, you can think about it as settling into the Morris attractor.

* Click on the `CatsAndDogPats` button in the left control panel, and double-click on the pattern for the Name layer -- this brings up an edit window where you can edit the values -- zero out the first cell and add a 1 into the second one (Socks).  Do `Init` and `Test Trial` responds to this (should be as expected from the above table).  Go ahead and try a few other name activations (change the appropriate value from 0 to 1).

Now, let's see how this network can give us general information about cats versus dogs, even though at some level it just has information about a set of individuals.

* Set all the Name inputs to 0, and then double-click on the Species inputs, and set the first unit to 1 (Cat), and do `Init` and `Test Trial` again.  Use the VCR `Time` buttons at the bottom-right to go back and replay the cycle-by-cycle activation settling. 

You should see that the network activates features that are typical of cats, and ends up settling on a subset of cat individuals.

> **Question 3.4:** Explain why the subset of cat individuals ended up getting activated, when just `cat` was provided as input -- how might this differential activation of individuals provide useful information about different cats in relation to the general `cat` category?

* Repeat this test but instead activate the `dog` unit instead of `cat`.

# Constraint Satisfaction

Now let's make use of some of the constraint satisfaction ideas. We can view the *harmony* of the network over cycles of settling using a graph view.

* Go back to testing just `cat` and select the `TstCycPlot` tab to view a plot of harmony over cycles of settling.

Notice that, as we expected, this value appears to monotonically increase over settling, indicating that the network is increasingly satisfying the constraints as the activations are updated.

Now, let's make some more *specific* queries for the network.

* Activate the `large` size input (last unit in `Size` column) in addition to `cat`, and test that. 

You should see that the final harmony value is lower than that for just `cat` alone (even though it starts out higher initially). This lower harmony reflects the fact that you provided discordant, inconsistent constraints, which the network was not able to satisfy as well (indeed if you look network, it ended up struggling for a long time and then flipping over to representing a *dog*, Rex).

* Turn off the `large` size input and turn on the `medium` one (middle unit in Size column) in addition to `cat`, and test again.

Although this case starts off earlier, the final harmony is also lower than `cat` alone, because even though there are some medium-sized cats, there are fewer of them, so the constraints are tighter. Put another way, `cat` is an easy constraint to satisfy, so the resulting harmony is large. `cat` plus `medium` is harder to satisfy because it applies to fewer things, so the harmony is lower.

There are a seemingly infinite number of different ways that you can query the network -- go ahead and present different input patterns and see what kinds of responses the network gives you. Most of them will we hope be recognizable as a reasonable response to the set of constraints provided by the input pattern.

Have fun experimenting!

