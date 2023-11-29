Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

This network is trained using Hebbian learning on paragraphs from an early draft of the *Computational Explorations* textbook, allowing it to learn about the overall statistics of when different words co-occur with other words, and thereby learning a surprisingly capable (though clearly imperfect) level of semantic knowlege about the topics covered in the textbook. This replicates the key results from the *Latent Semantic Analysis* research by [Landauer and Dumais (1997)](#references).

The `Input` layer has one unit for each different word that appeared with a frequency of 5 or higher (and excluding purely function words like "the" etc) -- 1920 words in total.  Each paragraph is presented as a single input pattern during training, with each word in the paragraph activated in the input (if the same word appears multiple times, it still just has the same unit activation).  After each such paragraph, Hebbian learning between input and active `Hidden` layer neurons takes place, using our standard BCM-style learning mechanism, as explored earlier in the [v1rf](https://github.com/CompCogNeuro/sims/blob/master/ch6/v1rf/README.md) and [self_org](https://github.com/CompCogNeuro/sims/blob/master/ch6/self_org/README.md) projects.  This model also includes recurrent lateral excitatory and inhibitory connections just like `v1rf`, which can induce a topological organization of neurons.  Unlike in the visual model, the high-dimensional nature of semantics makes this somewhat harder to understand but nevertheless the same principles are likely at work.

This network takes a while to train, so we will start by loading in pre-trained weights.

* Do `OpenWeights` in the toolbar.

# Individual Unit Representations

To start, let's examine the weights of individual units in the network.

* Select `r.Wt`, and then select various `Hidden` units at random to view.

You should observe sparse patterns of weights, with different units picking up on different patterns of words in the input. However, because the input units are too small to be labeled, you can't really tell which words a given unit is activated by. The `Wt Words` button provides a solution to this problem.

* View the weights for the unit just to the right of the lower-leftmost Hidden unit, and then hit the `Wt Words` button.  This pulls up a list of words with weight values > `WtWordsThr` (0.75).

One of the most interesting things to notice here is that the unit represents multiple roughly synonymous terms. For example, you might see the words "act," "activation," and "activations" and "add", "added", "adding", "additional".

> **Question 9.9:** List some other examples of roughly synonymous terms represented by this unit.

This property of the representation is interesting for two reasons. First, it indicates that the representations are doing something sensible, in that semantically related words are represented by the same unit. Second, these synonyms probably do not occur together in the same paragraph very often. Typically, only one version of a given word is used in a given context. For example, "The activity of the unit is..." may appear in one paragraph, while "The unit's activation was..." may appear in another. Thus, for such representations to develop, it must be based on the similarity in the general contexts in which similar words appear (e.g., the co-occurrence of "activity" and "activation" with "unit" in the previous example). This generalization of the semantic similarity structure across paragraphs is essential to enable the network to transcend rote memorization of the text itself, and produce representations that will be effective for processing novel text items.

* Click on the unit one row behind the first row (still one column over from the left) -- this unit has a relatively strong lateral weight with the first unit we clicked on.  Then click `Wt Words` -- each such window continues to show its original results so you can put them next to each other to compare between these two units.

You should observe a lot of overlap but also many differences between these two words.  This partially-overlapping coarse-coded distributed representation provides both some redundancy and more systematic coverage of a large high-dimensional semantic space.

* View the `Wt Words` representations for several other units further away in the layer to get a better sense of the variety of neuron "word receptive fields".

Although there clearly is sensible semantic structure at a local level within the individual-unit representations, it should also be clear that there is no single, coherent theme relating all of the words represented by a given unit. Thus, individual units participate in representing many different clusters of semantic structure, and it is only in the aggregate patterns of activity across many units that more coherent representations emerge. This network thus provides an excellent example of distributed representation.

# Summarizing Similarity with Cosines

To probe these distributed representations further, we can present words to the input and measure the hidden layer activation patterns that result. Specifically we are interested in the extent to which the hidden representations overlap for different sets of words, which tells us how similar overall the internal semantic representation is. Instead of just eyeballing the pattern overlap, we can compute a numerical measure of similarity using *normalized inner products* or *cosines* between pairs of sending weight patterns -- cosine values go from -1 to +1, with +1 being maximal similarity, 0 being completely unrelated, and -1 being maximal dissimilarity or anti-correlation.  We actually subtract the mean activity of each pattern before computing the inner product, which ends up being equivalent to a *correlation* -- this allows us to see negative correlation values even though all the activations are positive.

* You can see that "attention" is present in `Words1` in the control panel, and "binding" is in `Words2`.  Press `Test All` to test each of these cases in turn. If in the future you get an error message about the word not being found in the list of valid words, then fix and retry.

* Use the `Time` VCR buttons at the bottom right of NetView to replay the two trials.  You can see the actual distributed representations in the Hidden layer for these words, which is what the correlation is based on. You should observe that the two patterns overlap roughly 50%.  You can also click on `TstTrlLog` and see the two patterns there.

The `TstWordsCorrel` value toward the bottom of the control panel shows the correlation between these two patterns, which should be around .38.

* You can also click on `TstEpcPlot` to see a summary plot updated with each result as we go forward here.

* Now replace `Words2` with "spelling" (do not include quotes) and do `Test All` again.

You should see that attention and spelling are only related by around 0.06, indicating low similarity. This should match your overall intuition: we talk about attention as being critical for solving the binding problem in several different situations, but we don't talk much about the role of attention in spelling.

* Compare several other words that the network should know about from reading this textbook (tip: Click `TrainEnv` in the left control panel and then click `Words` in the window that appears to see a list of all the words, and scroll through that to see what words are in the valid list (these are words with frequency greater than 5, and not purely syntactic).

> **Question 9.10:** Report the `TstWordsCorrel` values for several additional sets of Words comparisons, along with how well each matches your intuitive semantics from having read this textbook yourself.

# Distributed Representations of Multiple Words

We now present multiple word inputs at the same time, and see how the network chooses a hidden layer representation that best fits this combination of words. Thus, novel semantic representations can be produced as combinations of semantic representations for individual words. This ability is critical for some of the more interesting and powerful applications of these semantic representations (e.g., multiple choice question answering, essay grading, etc.).

One interesting question we can explore is to what extent we can sway an otherwise somewhat ambiguous term to be interpreted in a particular way. For example, the term "binding" is used in two different contexts in our text. One context concerns the binding problem of visual features for object recognition, and another concerns the rapid binding of information into a memory in the hippocampus. Let's begin this exploration by first establishing the baseline association between "binding" and "object recognition."

* Enter "binding" for `Words1` and "object recognition" for `Words2`, and do `Test All`.

You should get a correlation of around .49 (interestingly, if you compare vs. just object or just recognition, it is lower). Now, let's see if adding "features" in addition to "binding" increases the hidden layer similarity, to push it more in the sense appropriate for object recognition.

* Add "features" to `Words1` and do `Test All`.

The similarity does indeed increase, producing a correlation of around .57. To make sure that there is an interaction between "binding" and "features" producing this increase, we also need to test with just "features" alone.

* Cut out "binding" from `Words1`, so it just has "features", and do `Test All`.

The similarity drops back to .44. Thus, there is some extra overlap in the combination of "binding" and "features" together that is not present by using each of them alone. Note also that the direct overlap between features and binding alone is .43. Now if we instead probe with "rapid binding" (still against "invariant object recognition," we should activate a different sense of attention, and get a smaller correlation.

* Set `Words1` to "rapid binding" and words2 to "object recognition" and `Test All`.

The similarity does now decrease, with a correlation of only around .28. Thus, we can see that the network's activation dynamics can be influenced to emphasize different senses of a word.

* To finish this test, now enter "hippocampus" in `Words2` and `Test All`, to see if that provides a better match to this sense of binding.

You should see that the similarity goes back up to .47. Thus, this is potentially a very powerful and flexible form of semantic representation that combines rich, overlapping distributed representations and activation dynamics that can magnify or diminish the similarities of different word combinations.

> **Question 9.11:** Think of another example of a word that has different senses (that is well represented in this textbook), and perform an experiment similar to the one we just performed to manipulate these different senses. Document and discuss your results.

# A Multiple-Choice Quiz

Based on your knowledge of the textbook, which of the options following each "question" provides the best match to the meaning?

0. neural activation function
    - A. spiking rate code membrane potential point
    - B. interactive bidirectional feedforward
    - C. language generalization nonwords
1. transformation
    - A emphasizing distinctions collapsing differences
    - B error driven hebbian task model based
    - C spiking rate code membrane potential point
2. bidirectional connectivity
    - A amplification pattern completion
    - B competition inhibition selection binding
    - C language generalization nonwords
3. cortex learning
    - A error driven task based hebbian model
    - B error driven task based
    - C gradual feature conjunction spatial invariance
4. object recognition
    - A gradual feature conjunction spatial invariance
    - B error driven task based hebbian model
    - C amplification pattern completion
5. attention
    - A competition inhibition selection binding
    - B gradual feature conjunction spatial invariance
    - C spiking rate code membrane potential point
6. weight based priming
    - A long term changes learning
    - B active maintenance short term residual
    - C fast arbitrary details conjunctive
7. hippocampus learning
    - A fast arbitrary details conjunctive
    - B slow integration general structure
    - C error driven hebbian task model based
8. dyslexia
    - A surface deep phonological reading problem damage
    - B speech output hearing language nonwords
    - C competition inhibition selection binding
9. past tense
    - A overregularization shaped curve
    - B speech output hearing language nonwords
    - C fast arbitrary details conjunctive

We can present this same quiz to the network, and determine how well it does relative to students in the class! The telegraphic form of the quiz is because it contains only the content words that the network was actually trained on. The best answer is always A, and B was designed to be a plausible foil, while C is obviously unrelated (unlike people, the network can't pick up on these regularities across test items). The quiz is presented to the network by first presenting the "question," recording the resulting hidden activation pattern, and then presenting each possible answer and computing the correlation of the resulting hidden activation with that of the question. The answer that has the closest correlation is chosen as the network's answer. 

* Press the `Quiz All` button, and then click on the `TstQuizPlot` tab to see the overall results. You will see a bar plot of the correlation matches for each alternative answer to each question, along with the scoring of that as an error or not. At the end you will see summary statistics for overall performance (`Err` in the last bar is summary Pct correct).  You can click on the `TstQuizLog` table in the left control panel to see the text values in tabular form too.

You should observe that the network does pretty well, but not perfectly, with a `TstQuizPctCor` of .8 = 80 percent correct.  The network does a very good job of rejecting the obviously unrelated answer C, but it does not always match our sense of A being better than B. In question 6, the B phrase was often mentioned in the context of the question phrase, but as a *contrast* to it, not a similarity. Because the network does not have the syntactic knowledge to pick up on this kind distinction, it considers them to be closely related because they appear together. This probably reflects at least some of what goes on in humans -- we have a strong association between "black" and "white" even though they are opposites. However, we can also use syntactic information to further refine our semantic representations -- a skill that is lacking in this network, which is taken up in the final simulation in this chapter.

# References

Landauer, T. K., & Dumais, S. T. (1997). A Solution to Plato’s Problem: The Latent Semantic Analysis Theory Of Acquisition, Induction, and Representation of Knowledge. Psychological Review, 104, 211–240.

