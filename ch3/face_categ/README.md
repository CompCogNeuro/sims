Back to [All Sims](https://github.com/CompCogNeuro/sims)

# Introduction

This project explores how sensory inputs (in this case simple cartoon faces) can be categorized in multiple different ways, to extract the relevant information and collapse across the irrelevant. It allows you to explore both bottom-up processing from face image to categories, and top-down processing from category values to face images (imagery), including the ability to dynamically iterate both bottom-up and top-down to cleanup partial inputs (partially occluded face images).

If you are following along with the text, then first do Part I in the
section on Categorization, and then come back for Part II after reading
about Bidirectional Excitatory Dynamics and Attractors.

# Part I: Feedforward (bottom-up) Flow of Information from Inputs to Categories

## The Network and Face Inputs

Let's first examine the network, shown in the tab in the right 3D panel.
It has a 16x16 layer for the face images, and three different
categorization output layers:

-   with "Happy" and "Sad" units, categorizes the emotion represented in
    the face into these two different categories.

-   with "Female" and "Male" units, categorizes the face according to
    these two gender categories.

-   with 6 labeled units with the names given to the different faces in
    the input (Alberto, Betty, Lisa, Mark, Wendy, Zane) -- the network
    can categorize the individual despite differences in emotional
    expression. Four additional units are available if you want to
    explore further by adding new faces.

These weights were learned in a way that makes their representations
particularly obvious by looking at these weights, so you can hopefully
see sensible-looking patterns for each unit. To further understand how
this network works, we can look at the input face patterns and
corresponding categorization values that it was trained on (we'll
explain this learning process in the [Learning
Chapter](/CCNBook/Learning "wikilink")).

## Testing the Network

The next step in understanding the basics of the network is to see it
respond to the inputs.

You will see the network process the face input and activate the
appropriate output categories for it (e.g., for the first pattern, it
will activate Happy, Male, and Alberto). Note that we are using the
NOISY_XX1 rate coded activation function, as with most of our
simulations.

You have probably noticed that the pattern of network activations was
recorded in the display next to the network. This allows you to see the
whole picture of network behavior in one glance.

## Using Cluster Plots to Understand the Categorization Process

A provides a convenient way of visualizing the similarity relationships
among a set of items, where multiple different forms of similarity may
be in effect at the same time (i.e., multidimensional similarity
structure). First, we'll look at the cluster plot of the input faces,
and then of the different categorizations performed on them, to see how
the network transforms the similarity structure to extract the relevant
information and collapse across the irrelevant.

You should see the resulting cluster plot in the tab.

Now, let's see how this input similarity structure is transformed by the
different types of categorization.

You should observe that the different ways of categorizing the input
faces each emphasize some differences while collapsing across others.
For example, if you go back and look at the values of the "Happy" and
"Sad" Emotion units, you will clearly see that these units care most
about (i.e., have the largest weights from) the mouth and eye features
associated with each of the different emotions, while having weaker
other weights from the inputs that are common across all faces.

This ability of synaptic weights to drive the detection of specific
features in the input is what drives the categorization process in a
network, and it is critical for extracting the behaviorally-relevant
information from inputs, so it can be used at a higher level to drive
appropriate behavior. For example, if Zane is looking sad, then perhaps
it is not a good time to approach him for help on your homework..

In terms of **localist vs. distributed** representations, the category
units are localist within each layer, having only one unit active,
uniquely representing a specific category value (e.g., Happy vs. Sad).
However, if you aggregate across the set of three category layers, it
actually is a simple kind of distributed representation, where there is
a distributed pattern of activity for each input, and the similarity
structure of that pattern is meaningful. In more completely distributed
representations, the individual units are no longer so clearly
identifiable, but that just makes things more complicated for the
purposes of this simple simulation.

Having multiple different ways of categorizing the same input in effect
at the same time (in parallel) is a critical feature of neural
processing -- all too often researchers assume that one has to choose a
particular level at which the brain is categorizing a given input, when
in fact all evidence suggests that it does massively parallel
categorization along many different dimensions at the same time.

# Part II: Bidirectional (Top-Down and Bottom-Up) Processing

In this section, we use the same face categorization network to explore
bidirectional top-down and bottom-up processing through the
bidirectional connections present in the network. First, let's see these
bidirectional connections.

Thus, as we discussed in the [Networks
Chapter](/CCNBook/Networks "wikilink"), the network has roughly
symmetric bidirectional connectivity, so that information can flow in
both directions and works to develop a consistent overall interpretation
of the inputs that satisfies all the relevant constraints at each level
(*multiple constraint satisfaction*).

## Top-Down Imagery

A simple first step for observing the effects of bidirectional
connectivity is to activate a set of high-level category values and have
that information flow top-down to the input layer to generate an image
that corresponds to the combination of such categories. For example, if
we activate Happy, Female, and Lisa, then we might expect that the
network will be able to "imagine" what Lisa looks like when she's happy.

You should see that the high-level category values for the first face in
the list (Happy, Male, Alberto) were activated at the start, and then
the face image filed in over time based on this top-down information.

## Interactive Top-Down and Bottom-Up and Partial Faces

Next, let's try a more challenging test of bidirectional connectivity,
where we have partially occluded face input images (20 pixels at random
have been turned off from the full face images), and we can test whether
the network will first correctly recognize the face (via bottom-up
processing from input to categories), and then use top-down activation
to fill in or **pattern complete** the missing elements of the input
image, based on the learned knowledge of what each of the individuals
(and their emotions) look like.

You should observe the initial partial activation pattern, followed by
activation of the category-level units, and then the missing elements of
the face image gradually get filled in.

Another way of thinking about the behavior of this network is in terms
of **attractor dynamics**, where each specific face and associated
category values represents a coordinated attractor, and the process of
activation updating over cycles results in the network settling into a
specific attractor from a partial input pattern that neverthelss lies
within its overall attractor basin.

At a technical level, the ability of the network to fill in the missing
parts of the input requires **soft clamping** of the input patterns --
the face pattern comes into each input as an extra contribution to the
excitatory net input, which is then integrated with the other synaptic
inputs coming top-down from the category level.


