Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

This simulation illustrates how self-organizing learning in response to natural images produces the oriented edge detector receptive field properties of neurons in primary visual cortex (V1). This provides insight into why the visual system encodes information in the way it does, while also providing an important test of the biological relevance of our computational models.

You will notice that the network has the two input layers, each 12x12 in size, one representing a small patch of on-center LGN neurons (), and the other representing a similar patch of off-center LGN neurons (). Specific input patterns are produced by randomly sampling a patch from a set of four larger (800x600 pixels) images of natural scenes (you can view these scenes by clicking on the images v1rf_img1.png, etc in this directory in your file browser). The single V1 layer is 14x14 in size.

You should observe that the unit is fully, randomly connected with the input layers, and that it has a circular *neighborhood* of lateral excitatory connectivity, which is key for inducing topographic representations. We have also added a lateral inhibitory projection, with 0 initial weights, which learns (as do the excitatory lateral connections) -- these two lateral projections allow the model to develop more complex topographic representations compared to a single fixed excitatory projection.

The hidden units will initially have a somewhat random and sparse pattern of activity in response to the input images.

You should observe that the on- and off-center input patterns have complementary activity patterns. That is, where there is activity in one, there is no activity in the other, and vice versa. This complementarity reflects the fact that an on-center cell will be excited when the image is brighter in the middle than the edges of its receptive field, and an off-center cell will be excited when the image is brighter in the edges than in the middle. Both cannot be true, so only one is active per image location. Keep in mind that the off-center units are active (i.e., with positive activations) to the extent that the image contains a relatively dark region in the location coded by that unit. Thus, they do not actually have negative activations to encode darkness.

You should observe that there tend to be contiguous edges of light-dark transitions, at various angles -- you might recognize close-up segments of trees, mountains etc in the input image shown in the upper right of the network display.

If you pay close attention to the patterns of activity in the Hidden layer, you may also perceive that neighboring neurons have a greater tendency to be co-active with each other -- this is due to the lateral connectivity, which is not strong enough to force only a single bubble of contiguous neurons to become active, but over time it is sufficient to induce a neighborhood topology as we'll see in a moment.

If we let this network run for many, many more image presentations, it will develop a set of representations in V1 that reflect the correlational structure of edges that are present in the input. Because this can take several minutes (depending on your computer), we will just load a pre-trained network at this point.  This loads network weights that were trained for 100 epochs of 100 image presentation, or a total of 10,000 image presentations.

You should see in the weights projected onto the input layers some indication of a left-diagonal orientation coding (i.e., a diagonal bar of stronger weight values), with the on-center () bar in the bottom-left of the input patch, and the off-center () bar just above and right of it. Note that the network always has these on- and off-center bars in adjacent locations, never in the same location, because they are complementary (both are never active in the same place). Also, these on- and off-center bars will always be parallel to each other and perpendicular to the direction of light change, because they encode edges, where the change in illumination is perpendicular to the orientation of the edge.

You should observe subtle changes in the weights. As you click around to other units, you might also see a transition in the *polarity* of the receptive field, with some having a *bipolar* organization with one on-center and one off-center region, while others have a *tripolar* organization with one on-center region and two off-center ones. Further, they may vary in size and location.

# Receptive Field View

Although this examination of the individual unit's weight values reveals both some aspects of the dimensions coded by the units and their topographic organization, it is difficult to get an overall sense of the unit's representations by looking at them one at a time. Instead, we will use a single display of all of the receptive fields at one time.

You will now see a grid view that presents the pattern of receiving weights for each hidden unit. To make it easier to view, this display shows the off-center weights subtracted from the on-center ones, yielding a single plot of the receptive field for each hidden unit. Positive values (in red tones going to a maximum of yellow) indicate more on-center than off-center excitation, and vice versa for negative values (in blue tones going to a maximum negative magnitude of purple). The receptive fields for each hidden unit are arranged to correspond with the layout of the hidden units in the network. To verify this, look at the same 3 units that we examined individually, which are along the bottom right of the grid view. You should see the same features we described above, keeping in mind that this grid log represents the *difference* between the on-center and off-center values. You should clearly see the topographic nature of the receptive fields, and also the full range of variation among the different receptive field properties.

You should observe that the topographic organization of the different features, where neighboring units usually share a value along at least one dimension (or are similar on at least one dimension). Keep in mind that the topography wraps around, so that units on the far right should be similar to those on the far left, and so forth. You should also observe that a range of different values are represented for each dimension, so that the space of possible values (and combinations of values) is reasonably well covered.

Mathematically, the overall shape of these receptive fields can be captured by a **Gabor** function, which is a sine wave times a gaussian, and many vision researchers use such functions to simulate V1-level processing.

# Probing Inputs

We can directly examine the weights of the simulated neurons in our model, but not in the biological system, where more indirect measures must be taken to map the receptive field properties of V1 neurons. One commonly used methodology is to measure the activation of neurons in response to simple visual stimuli that vary in the critical dimensions (e.g., oriented bars of light). Using this technique, experimenters have documented all of the main properties we observe in our simulated V1 neurons -- orientation, polarity, size, and location tuning, and topography. We will simulate this kind of experiment now.  This will bring up a table containing 4 events, each of which represents an edge at a different orientation and position.

Now, let's present these patterns to the network.

You should observe that the units that coded for the orientation and directionality of the probe were activated.

If you are interested, you can draw new patterns into the probe events, and present them by the same procedure just described. In particular, it is interesting to see how the network responds to multiple edges present in a single input event.

# Recurrent Connectivity Effects

Finally, to see that the lateral connectivity is responsible for developing topographic representations, you can load a set of receptive fields generated from a network trained with set to 0.05.

You should see little evidence of a topographic organization in the resulting receptive field grid log, indicating that this strength of lateral connectivity provided insufficient neighborhood constraints.

There appears to be an interaction between the topographic aspect of the representations and the nature of the individual receptive fields themselves, which look somewhat different in this weaker lateral connectivity case compared to the original network. These kinds of interactions have been documented in the brain {{&lt; cite "DasGilbert95,WelikyKandlerKatz95" &gt;}} and make sense computationally given that the lateral connectivity has an important effect on the response properties of the neurons, which is responsible for tuning up their receptive fields in the first place.

If you are interested, you can also try running models without the lateral inhibition (set to 0), and / or without learning in the recurrent cons (turn off) -- you should see that the resulting receptive fields are less complex, and are composed of more monolithic blocks. This suggests an important role for learning in these lateral connections -- both excitatory and inhibitory.

