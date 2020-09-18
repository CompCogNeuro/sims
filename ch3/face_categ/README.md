Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

This project explores how sensory inputs (in this case simple cartoon faces) can be categorized in multiple different ways, to extract the relevant information and collapse across the irrelevant. It allows you to explore both bottom-up processing from face image to categories, and top-down processing from category values to face images (mental imagery), including the ability to dynamically iterate both bottom-up and top-down to cleanup partial inputs (partially occluded face images).

If you are following along with the text, then first do Part I in the section on *Categorization*, and then come back for Part II after reading about *Bidirectional Excitatory Dynamics and Attractors*.

# Part I: Feedforward (bottom-up) Flow of Information from Inputs to Categories

## The Network and Face Inputs

Let's first examine the network, shown in the tab in the right 3D panel.  It has a 16x16 layer for the face images, and three different categorization output layers:

* `Emotion` with "Happy" and "Sad" units, categorizes the emotion represented in the face into these two different categories.

* `Gender` with "Female" and "Male" units, categorizes the face according to these two gender categories.

* `Identity` with 6 labeled units with the names given to the different faces in the input (Alberto, Betty, Lisa, Mark, Wendy, Zane) -- the network can categorize the individual despite differences in emotional expression. Four additional units are available if you want to explore further by adding new faces.

* Select the `r.Wt` variable to view in the `NetView` network view, and click on each of the different output category neurons in the network. This will display the weight values going into each neuron. 

These weights were learned in a way that makes their representations particularly obvious by looking at these weights, so you can hopefully see sensible-looking patterns for each unit. To further understand how this network works, we can look at the input face patterns and corresponding categorization values that it was trained on (this learning process is explained in the chapter on *Learning* in the textbook).

* Click on the `FacePats` button in the left-hand *control panel* -- you can see the names, faces, and emotional expresions that the network was trained on.

## Testing the Network

The next step in understanding the basics of the network is to see it respond to the inputs.

* Select the `Act` value to view the neuron activities in the NetView, and then do `Test Trial` to see the network respond to the first face, `Alberto_happy`, with the view being updated for each of 20 cycles of neural updating.  You can use the VCR buttons in the lower-right of the NetView, after the `Time` label, to review how the network responded cycle-by-cycle -- use the fast-reverse to skip back to the start and then single-step forward in time to see things unfolding cycle-by-cycle.

You should see the network process the face input and activate the appropriate output categories for it (e.g., for the first pattern, it will activate `happy`, `male`, and `Alberto`). 

* Continue to `Test Trial` through the remainder of the face input patterns, and verify that it correctly categorizes each input.

* You can view a full record of the input and responses by clicking on the `TstTrlLog` button which pulls up a window with each trial recorded.

## Using Cluster Plots to Understand the Categorization Process

A [ClusterPlot](https://github.com/CompCogNeuro/sims/blob/master/ch3/face_categ/ClusterPlot.md) provides a convenient way of visualizing the similarity relationships among a set of items, where multiple different forms of similarity may be in effect at the same time (i.e., multidimensional similarity structure).  If unfamiliar with these, please click that link to read more about how to read a cluster plot.  First, we'll look at the cluster plot of the input faces, and then of the different categorizations performed on them, to see how the network transforms the similarity structure to extract the relevant information and collapse across the irrelevant.

* Press the `Cluster Plots` button in the toolbar, and then click on the `eplot.Plot2D` button next to the `ClustFaces` line in the control panel on the left.  This will pull up a cluster plot run on the face `Input` layer images.

> **Question 3.1:** Given what you know about how a Cluster Plot works (see above link), describe how the three features (gender, emotion, and identity) relate to the clustering of images by similarity.  Specifically, think about where there are the greatest number of overlapping pixels across the different images from each of the different categories (all the happy vs. sad, female vs. male, and within each individual).

Now, let's see how this input similarity structure is transformed by the different types of categorization.

* Click on the plot for `ClustEmote`, which shows the cluster plot run on the `Emotion` layer patterns for each input.

> **Question 3.2:** How does the Emotion categorization transform the overall face input similarity compared to what we saw in the first cluster plot -- ie., what items are now the most similar to each other?


* Click on the plot for `ClustGend`, which shows the cluster plot run on the `Gender` layer patterns for each input, and likewise for the `ClustIdent` plot, which shows the cluster plot run on the `Identity` layer patterns for each input.

You should observe that the different ways of categorizing the input faces each emphasize some differences while collapsing across others.  For example, if you go back and look at the `r.Wt` values of the "happy" and "sad" Emotion units, you will clearly see that these units care most about (i.e., have the largest weights from) the mouth and eye features associated with each of the different emotions, while having weaker other weights from the inputs that are common across all faces.  This enables the emotion layer to extract that emotional signal much more clearly than it is represented in the overall face inputs -- the emotion differences are present in those face inputs, but just not very dominant.

This ability of synaptic weights to drive the detection of specific features in the input is what drives the categorization process in a network, and it is critical for extracting the behaviorally-relevant information from inputs, so it can be used at a higher level to drive appropriate behavior. For example, if Zane is looking sad, then perhaps it is not a good time to approach him for help on your homework..

In terms of **localist vs. distributed** representations, the category units are localist within each layer, having only one unit active, uniquely representing a specific category value (e.g., happy vs. sad). However, if you aggregate across the set of three category layers, it actually is a simple kind of distributed representation, where there is a distributed pattern of activity for each input, and the similarity structure of that pattern is meaningful. In more completely distributed representations, the individual units are no longer so clearly identifiable, but that just makes things more complicated for the purposes of this simple simulation.

Having multiple different ways of categorizing the same input in effect at the same time (in parallel) is a critical feature of neural processing -- all too often researchers assume that one has to choose a particular level at which the brain is categorizing a given input, when in fact all evidence suggests that it does massively parallel categorization along many different dimensions at the same time.

# Part II: Bidirectional (Top-Down and Bottom-Up) Processing

In this section, we use the same face categorization network to explore bidirectional top-down and bottom-up processing through the bidirectional connections present in the network. First, let's see these bidirectional connections.

* In the NetView, select the `s.Wt` (sending weights) variable and click on the various category output units -- you can click back and forth between `r.Wt` and `s.Wt` to compare the receiving and sending weights for a given unit -- in general they should have a similar pattern with somewhat different overall magnitudes. 

Thus, as we discussed in the *Networks* Chapter, the network has roughly symmetric bidirectional connectivity, so that information can flow in both directions and works to develop a consistent overall interpretation of the inputs that satisfies all the relevant constraints at each level (*multiple constraint satisfaction*).

## Top-Down Imagery

A simple first step for observing the effects of bidirectional connectivity is to activate a set of high-level category values and have that information flow top-down to the input layer to generate an image that corresponds to the combination of such categories. For example, if we activate Happy, Female, and Lisa, then we might expect that the network will be able to "imagine" what Lisa looks like when she's happy.

* In the NetView, select the `Act` variable. Next, click the `Set Input` button in the top toolbar, and select `Top Down`, and then do Init and hit `Test Trial` multiple times to see the activation dynamics unfold -- use the Time VCR buttons to examine the settling cycle-by-cycle.

You should see that the high-level category values for the first face in the list (Happy, Male, Alberto) were activated at the start, and then the face image filed in over time based on this top-down information.

## Interactive Top-Down and Bottom-Up and Partial Faces

Next, let's try a more challenging test of bidirectional connectivity, where we have partially occluded face input images (20 pixels at random have been turned off from the full face images), and we can test whether the network will first correctly recognize the face (via bottom-up processing from input to categories), and then use top-down activation to fill in or **pattern complete** the missing elements of the input image, based on the learned knowledge of what each of the individuals (and their emotions) look like.

* Click `Set Input` again and turn `Top Down` off, and then click `Set Pats` and select `Partial` to input partial faces instead of full faces.  Then `Test Trial` through the inputs again.

You should observe the initial partial activation pattern, followed by activation of the category-level units, and then the missing elements of the face image gradually get filled in (definitely use the Time VCR buttons to see this unfold cycle-by-cycle).

> **Question 3.3:** Across multiple different such partial faces, what is the order in which the *correct* category units get active (there may be transient activity in incorrect units)?  For each case also list how this order corresponds to the timing of when the missing features in the input face start to get filled in.

Another way of thinking about the behavior of this network is in terms of **attractor dynamics**, where each specific face and associated category values represents a coordinated attractor, and the process of activation updating over cycles results in the network settling into a specific attractor from a partial input pattern that neverthelss lies within its overall attractor basin.

More generally, being able to fill in missing pieces of associated information is a key benefit of bidirectional connectivity, and it is a highly flexible way to access information -- any subset of elements can trigger the completion of the rest.  This is also known as **content-addressable memory** as opposed to how standard computers typically require a specific memory address to access memory.  One of the great innovations of Google and other such search engines was to recreate the content addressable nature human memory so you can just enter some random words and get to the full relevant information.

At a technical level, the ability of the network to fill in the missing parts of the input requires **soft clamping** of the input patterns -- the face pattern comes into each input as an extra contribution to the excitatory net input, which is then integrated with the other synaptic inputs coming top-down from the category level.

Also, you might be surprised to know that most of the neural networks currently powering modern AI applications do *not* have this bidirectional connectivity, and thus lack the corresponding flexibilty of human knowledge and memory.

