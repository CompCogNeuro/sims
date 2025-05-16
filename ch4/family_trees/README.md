Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

This simulation illustrates the case of learning in a deep network using the family trees task (Hinton, 1986). This task shows how learning can recode inputs that have no similarity structure into a hidden layer that captures the functional similarity structure of the items. In this case, the people at different levels of a family tree get mapped into similar representations. This shows that neural networks are not merely bound by surface similarity structure, but in fact can create their own similarity structure based on functional relationships. Because this network has multiple hidden layers intervening between the input and the output, it also serves as a classic test of error-driven learning in a deep network.

![The Family Trees](fig_family_trees.png?raw=true "The Family Trees")

**Figure 1:** The two isomorphic family trees that are trained, with each pairwise relationship (Husband, Wife, Son, Daughter, Father, Mother, Brother, Sister, Aunt, Uncle, Niece, Nephew) trained as a distinct input/output pattern.

The structure of the environment is shown in Figure 1. The network is trained to produce the correct name in response to questions like "Rob is married to whom?" These questions are presented by activating one of 24 name units in an agent input layer (e.g., "Rob"), in conjunction with one of 12 units in a relation input layer (e.g., "Married'"), and training the network to produce the correct unit activation over the patient output layer.

# Training

First, notice that the network (displayed in [[sim:Network]] tab) has `Agent` and `Relation` input layers, and a `Patient` output layer all at the bottom of the network. These layers have localist representations of the 24 different people and 12 different relationships, which means that there is no overlap, and thus no overt similarity, in these input patterns between any of the people. The `AgentCode`, `RelationCode`, and `PatientCode` hidden layers provide a means for the network to re-represent these localist representations as richer distributed patterns that should facilitate the learning of the mapping by emphasizing relevant distinctions and deemphasizing irrelevant ones. The central `Hidden` layer is responsible for performing the mapping between these recoded representations to produce the correct answers.

* Click on the [[sim:Family Trees]] Patternu on the left to browse through the input / output patterns, which should help you understand how the task is presented to the network. The names of the events in the first column are in the following format: Agent.Relation.Patient, and can be interpreted along the following lines: "Christo's wife is who?" "Penny." 

Now, let's see how this works with the network itself.

* Do [[sim:Init]] and [[sim:Step Trial]] in toolbar.  This will run the network through 4 quarters of processing the input, settling in the minus or expectation phase, and then receiving the plus or outcome activation in the `Patient` layer.  You can use the VCR buttons at the bottom-right of the Network to review these activation states at the end of each quarter (or click on the [[sim:Phase]] / `ActQ1`, `ActQ2`, `ActM` and `ActP` variables).  

You should see that all of the hidden layers change their activation patterns to reflect this additional information, feeding backward through the bidirectional connectivity, at least to some extent. You can click on [[sim:Phase]]/`ActDif` to see the difference between minus and plus phase activation to see this error signal directly.  The default network is using the default combination of BCM Hebbian (self-organizing) and error-driven learning in XCAL. Let's see how long it takes this network to learn the task.

* Click on [[sim:Train Epoch Plot]] to view a plot of performance over epochs of training, and then change the step level (to the right of [[sim:Step]]) from `Trial` to `Run`, then do [[sim:Step]].
 
As the network trains, the graph displays the `PctErr` statistic for training: proportion of trials with a non-zero SSE error. The training times are variable, depending on the random initial weights, but yours should train somewhere between 20 and 50 epochs. During this time, the network has learned to transform the localist input units into systematic internal distributed representations that capture the semantic structure of the relationships between the different members of the two families. To see this, we need to run various analyses of the Hidden layer activations.

# Representational Analysis

To get a sense of how learning has shaped the transformations performed by this network to emphasize relevant similarities, we can do various analyses of the hidden unit activity patterns recorded by testing over all the 100 different input patterns (all the specific Agent-Relation-Patient triples), comparing trained vs. random initial weights to see what specifically has been learned.

* First, set the run mode (in the top left) to `Test` instead of `Train`, and do [[sim:Init]], [[sim:Run]] to collect a recording of all the layer activity patterns (you can see them in the [[sim:Test Trial]] tab), and then press the [[sim:Reps Analysis]] button in the toolbar, which performs various different analyses as described below on these activations from the Hidden and AgentCode layers.

The most direct way to examine relationships among different activation patterns is to compute the pairwise similarities (inverse of distances) between each pattern and all others.  We use the *correlation* similarity measure, which produces a 1 for identical patterns, 0 for completely unrelated patterns, and -1 for completely *anticorrelated* patterns.  (Interestingly, correlation is equivalent to a cosine angle in N dimensional space, using mean-normalized activation patterns, and cosine is equivalent to the simple dot product between the vectors, normalized by the length of the vectors.)

* Click on [[sim:Stats]] in the left panel, then click on the [[sim:Sim Mats]], and then on button next to `HiddenRel`, to bring up the similarity matrix for the Hidden layer, with patterns labeled and sorted according to the type of relationship encoded.  This sorting is key to making the patterns of similarity related to this relationship factor evident in the similarity matrix.

You should observe largish red squares along the diagonal yellow line cutting across from the upper left to the lower right, clearly organized according to the relationship groupings. These squares indicate a moderately high level of similarity among inputs with the same relationship. That is, the internal hidden representations of people in the same functional relationship, like 'aunt', is more similar than it is to other familial positions. In addition, you should observe a smattering of other similar patterns with similar red color, typically also organized according to the relationship groups.

* Next, click on the button next to `HiddenAgent`, which shows the hidden unit activation distances organized by the agent role. Look back and forth between this and the previous relationship-organized plot to compare them.  You can hold down the `Shift` key when you click the button to pull up `Sim Mats` in separate windows.

> **Question 4.8:** What do you think this pattern of results means for how the network is encoding the inputs -- what is the primary component (relationship vs. agent) of the inputs along which the hidden layer is organizing the input patterns?  How might it be beneficial for the model to have organized the hidden layer in this way, in order to perform the task -- i.e., which component is more *central* across all of the patterns?

Although the raw distance matricies show all the data, it can be difficult to see the more complex patterns -- distributed representations encode many different forms of similarity at the same time. Thus, we will use two other plots to see more of this structure: cluster plots and principal-components-analysis (PCA) plots.

* Click on the [[sim:HiddenRelClust]] tab to see the cluster plot of the Hidden activations labeled by relationships.

You can see now that for example Niece and Nephew are grouped together, as are several other related items (e.g., Brother and Sister). However, there are clearly some other dimensions of structure represented here -- the cluster plot can change dramatically based on small differences in distances.

Thus, we turn to the PCA plot as a way to cut through some of the complexity. PCA is a way to distill the information about activities across many units into the strongest simpler components that explain much of the data. Here we focus on just the first two principal dimensions - where the first dimension discovers a component with the most shared information among the patterns, and then once that is taken into account, the second component displays the next remaining largest amount of shared variance.

* Click on the [[sim:HiddenRelPCA]] tab to see the PCA plot for the Hidden layer organized by relations.

The resulting plot shows the first principal component of variation on the X axis, and the second component along the Y axis. The exact patterns will differ depending on network and initial weights etc, but one reasonable way for the network to represent the problem is with one component (e.g. the X axis) representing relative age in the relationships, and the the other component separating male and female (e.g., Father, Son, Brother, Uncle together, vs. Sister, Mother, Daughter).

You can also look at the cluster and PCA plots of the Agent-based labels (the PCA plot here is for components 2 and 3), to see what kind of organization is taking place there. In general, you can see some sensible organization of similar places in the tree being nearby, but given the high-dimensional nature of the relationships and the distributed representations, it is not totally systematic, and because everyone is related in some way, it is difficult to really determine how sensible the organization is.

* Now, let's see how these results compare to the network with random initial weights.  Do `Train` mode [[sim:Init]], then `Test` mode [[sim:Init]], and [[sim:Run]], and [[sim:Reps Analysis]], and then click on [[sim:HiddenRelPCA]] again, and also look at the similarity matrix `Sim Mat` for that case too.

> **Question 4.9:** How do the untrained representations compare to the previous trained ones? Although some of the same larger structure is still present, is the organization as systematic as with the trained weights? Focus specifically on the relational data, as that was more clear in the trained case.

Note there is still a fair amount of structure in the distance matricies present even with random weights, so the differences between the PCA plots of the trained and untrained networks may be subtle. This is due to the similarity structure of the input patterns themselves -- even though each individual input unit is localist, there is structure across the three layers (agent, relation, patient), and the model representations will tend to reflect this structure even without any learning. Learning refines this initial structure, and, most critically, establishes the proper synaptic weights to produce the correct Patient response for each input.

Getting more systematic learned representational structure in the network requires a larger set of training patterns that more strongly constrain and shape the network's internal representations (in the original Hinton (1986) model, a much smaller number of hidden units was used, over a very long training time, to force the model to develop more systematic representations even with this small set of patterns). We'll see examples of larger sets of inputs shaping systematic internal representations in later chapters, for example in the object recognition model in the Perception chapter and the spelling-to-sound model in the Language chapter. In any case, focus on what types of items are more likely to be clustered together before and after training. 

# The Roles of Hebbian Vs. Error-Driven Learning

As a deep, multi-layered network, this model can demonstrate some of the advantages of combining self-organizing (Hebbian) and error-driven learning, although they are fairly weak effects due to the limited structure and size of the input patterns. The `Learn` variable can be changed from `HebbError` to `PureErr` or `PureHebb` -- you have to hit `Init` after changing this setting, to have it affect the relevant parameters.

You can try running the model using the `PureErr` setting, and you may (or may not) find that it takes longer to learn than the `HebbErr` setting. Then try `PureHebb` -- although it does learn something (which is impressive given that there is no explicit error-driven learning), it fails to learn the task beyond around 50% correct at best (typically around 25% correct, 75% errors). Nevertheless, this Hebbian learning is beneficial when combined with error-driven learning, in larger networks -- in current large-scale vision models (e.g., O'Reilly, Wyatte, et al, 2013) it is essential and learning fails entirely without the BCM Hebbian.

Hebbian learning facilitates learning in deep networks, especially as the error signals get weaker once the network has learned much of the task and the remaining error gradients are very small -- the self-organizing constraints provide a continued steady drive on learning (which can also facilitate generalization to new patterns depending on similar structures). Furthermore, if we compare even the error-driven learning network to a pure backpropagation or recurrent GeneRec/CHL error-driven network, both of which lack the inhibitory constraints and learn this family trees problem much more slowly (O'Reilly, 1996), we see that inhibitory competition is also providing a beneficial learning constraint in Leabra networks.

Nevertheless, pure Hebbian learning by itself is clearly incapable of learning tasks such as this (and many many others). One reason is evident in the average learning trajectory: the positive feedback dynamics and "myopic" local perspective of pure Hebbian learning end up creating rich-get-richer representations that result in worse performance as learning proceeds. Thus, error-driven learning must play a dominant role overall to actually learn complex cognitive tasks.


