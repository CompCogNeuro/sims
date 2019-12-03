Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

This network is trained using Hebbian learning on paragraphs from an early draft of the *Computational Explorations..* textbook, allowing it to learn about the overall statistics of when different words co-occur with other words, and thereby learning a surprisingly capable (though clearly imperfecdt) level of semantic knowlege about the topics covered in the textbook. This replicates the key results from the Latent Semantic Analysis research by Landauer and Dumais (1997).

This network takes a while to train, so we will start by loading in pre-trained weights.

# Individual Unit Representations

To start, let's examine the weights of individual units in the network.

You should observe sparse patterns of weights, with different units picking up on different patterns of words in the input. However, because the input units are too small to be labeled, you can't really tell which words a given unit is activated by. The GetWordRF button (get receptive field in terms of words) on the control panel provides a solution to this problem.

You should see an alphabetized list of the words that this hidden unit receives from with a weight above a threshold of .85. One of the most interesting things to notice here is that the unit represents multiple roughly synonymous terms. For example, you might see the words "act," "activation," and "activations." 

This property of the representation is interesting for two reasons. First, it indicates that the representations are doing something sensible, in that semantically related words are represented by the same unit. Second, these synonyms probably do not occur together in the same paragraph very often. Typically, only one version of a given word is used in a given context. For example, "The activity of the unit is..." may appear in one paragraph, while "The unit's activation was..." may appear in another. Thus, for such representations to develop, it must be based on the similarity in the general contexts in which similar words appear (e.g., the co-occurrence of "activity" and "activation" with "unit" in the previous example). This generalization of the semantic similarity structure across paragraphs is essential to enable the network to transcend rote memorization of the text itself, and produce representations that will be effective for processing novel text items.

Although there clearly is sensible semantic structure at a local level within the individual-unit representations, it should also be clear that there is no single, coherent theme relating all of the words represented by a given unit. Thus, individual units participate in representing many different clusters of semantic structure, and it is only in the aggregate patterns of activity across many units that more coherent representations emerge. This network thus provides an excellent example of distributed representation.

# Summarizing Similarity with Cosines

To probe these distributed representations further, we can present words to the input and measure the hidden layer activation patterns that result. Specifically we are interested in the extent to which the hidden representations overlap for different sets of words, which tells us how similar overall the internal semantic representation is. Instead of just eyeballing the pattern overlap, we can compute a numerical measure of similarity using *normalized inner products* or *cosines* between pairs of sending weight patterns -- cosine values go from -1 to +1, with +1 being maximal similarity, 0 being completely unrelated, and -1 being maximal dissimilarity or anti-correlation. The mean activity of each pattern is subtracted first before computing the inner product, so even though all the activations are positive, you can see negative cosines due to this renormalization.

The tab will activate and show you the cosines between the hidden representations of these two words. You should get a number around .32 (note that the similarity of each word with itself is 1).

You should see that attention and spelling are only related by around -0.14 -- because the mean is subtracted, any negative number indicates low similarity. This should match your overall intuition: we talk about attention as being critical for solving the binding problem in several different situations, but we don't talk much about the role of attention in spelling.

# Distributed Representations of Multiple Words

We now present multiple word inputs at the same time, and see how the network chooses a hidden layer representation that best fits this combination of words. Thus, novel semantic representations can be produced as combinations of semantic representations for individual words. This ability is critical for some of the more interesting and powerful applications of these semantic representations (e.g., multiple choice question answering, essay grading, etc.).

One interesting question we can explore is to what extent we can sway an otherwise somewhat ambiguous term to be interpreted in a particular way. For example, the term "binding" is used in two different contexts in our text. One context concerns the binding problem of visual features for object recognition, and another concerns the rapid binding of information into a memory in the hippocampus. Let's begin this exploration by first establishing the baseline association between "binding" and "object recognition."

You should get a cosine of around .47 (interestingly, if you compare vs. just object or just recognition, it is lower). Now, let's see if adding "features" in addition to "binding" increases the hidden layer similarity, to push it more in the sense appropriate for object recognition.

The similarity does indeed increase, producing a cosine of around .53. To make sure that there is an interaction between "binding" and "features" producing this increase, we also need to test with just "features" alone.

The similarity drops back to .36. Thus, there is some extra overlap in the combination of "binding" and "features" together that is not present by using each of them alone. Note also that the direct overlap between features and binding alone is .37. Now if we instead probe with "rapid binding" (still against "invariant object recognition," we should activate a different sense of attention, and get a smaller cosine.

The similarity does now decrease, with a cosine of only around .28. Thus, we can see that the network's activation dynamics can be influenced to emphasize different senses of a word.

You should see that the similarity goes back up to .47. Thus, this is potentially a very powerful and flexible form of semantic representation that combines rich, overlapping distributed representations and activation dynamics that can magnify or diminish the similarities of different word combinations.

# A Multiple-Choice Quiz

Based on your knowledge of the textbook, which of the options following each "question" provides the best match to the meaning?

-   0. neural activation function
    -   A spiking rate code membrane potential point
    -   B interactive bidirectional feedforward
    -   C language generalization nonwords
-   1. transformation
    -   A emphasizing distinctions collapsing differences
    -   B error driven hebbian task model based
    -   C spiking rate code membrane potential point
-   2. bidirectional connectivity
    -   A amplification pattern completion
    -   B competition inhibition selection binding
    -   C language generalization nonwords
-   3. cortex learning
    -   A error driven task based hebbian model
    -   B error driven task based
    -   C gradual feature conjunction spatial invariance
-   4. object recognition
    -   A gradual feature conjunction spatial invariance
    -   B error driven task based hebbian model
    -   C amplification pattern completion
-   5. attention
    -   A competition inhibition selection binding
    -   B gradual feature conjunction spatial invariance
    -   C spiking rate code membrane potential point
-   6. weight based priming
    -   A long term changes learning
    -   B active maintenance short term residual
    -   C fast arbitrary details conjunctive
-   7. hippocampus learning
    -   A fast arbitrary details conjunctive
    -   B slow integration general structure
    -   C error driven hebbian task model based
-   8. dyslexia
    -   A surface deep phonological reading problem damage
    -   B speech output hearing language nonwords
    -   C competition inhibition selection binding
-   9. past tense
    -   A overregularization shaped curve
    -   B speech output hearing language nonwords
    -   C fast arbitrary details conjunctive

We can present this same quiz to the network, and determine how well it does relative to students in the class! The telegraphic form of the quiz is because it contains only the content words that the network was actually trained on. The best answer is always A, and B was designed to be a plausible foil, while C is obviously unrelated (unlike people, the network can't pick up on these regularities across test items). The quiz is presented to the network by first presenting the "question," recording the resulting hidden activation pattern, and then presenting each possible answer and computing the cosine of the resulting hidden activation with that of the question. The answer that has the closest cosine is chosen as the network's answer. 

You should observe that the network does OK, but not exceptionally -- 60-80 percent performance is typical (i.e., .2 to .4 error mean). Usually, the network does a pretty good job of rejecting the obviously unrelated answer C, but it does not always match our sense of A being better than B. In question 6, the B phrase was often mentioned in the context of the question phrase, but as a *contrast* to it, not a similarity. Because the network does not have the syntactic knowledge to pick up on this kind distinction, it considers them to be closely related because they appear together. This probably reflects at least some of what goes on in humans -- we have a strong association between "black" and "white" even though they are opposites. However, we can also use syntactic information to further refine our semantic representations -- a skill that is lacking in this network. The next section describes a model that begins to address this skill.

