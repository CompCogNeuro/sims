Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

This model simulates normal and disordered (dyslexic) reading performance in terms of a distributed representation of word-level knowledge across Orthography, Semantics, and Phonology. It is based on a model by Plaut and Shallice (1993). Note that this form of dyslexia is *aquired* (via brain lesions such as stroke) and not the more prevalent developmental variety.

Because the network takes some time to train (for 250 epochs), we will just load in a pre-trained network to begin with. Be sure to download the weights from the panel at the right.

# Normal Reading Performance

For our initial exploration, we will just observe the behavior of the network as it "reads" the words presented to the orthographic input layer. Note that the letters in the input are ordered left-to-right, bottom to top.

You will see the activation flow through the network, and it should settle into the correct pronunciation and semantics for the first word, "tart" (the bakery food). In the network data at the bottom of the network view, you can see the output_name showing what the network produced over the phonology output layer. 

The column shows whether this item is concrete (*Con*) or abstract (*Abs*), the column indicates the input pattern, and column shows the word that is closest to what the network produced, and the columns after that indicate what type of error the network makes, if any. All of these should be 0, as the network is intact and well trained. Concrete words have more distinctive features, whereas abstract words have fewer.

# Reading with Complete Pathway Lesions

We next explore the network's ability to read with one of the two pathways to phonology removed from action. This relatively simple manipulation provides some insight into the network's behavior, and can be mapped onto two of the three dyslexias. Specifically, when we remove the semantic pathway, leaving an intact direct pathway, we reproduce the characteristics of surface dyslexia, where words can be read but access to semantic representations is impaired and visual errors are made. When we remove the direct pathway, reading must go through the semantic pathway, and we reproduce the effects of deep dyslexia by finding both semantic and visual errors. Note that phonological dyslexia is a milder form of deep dyslexia, which we explore when we perform incremental amounts of partial damage instead of lesioning entire pathways.

We begin by lesioning the semantic pathway.

This does not actually remove any units or other network structure; it just flips a "lesion" flag that (reversibly) deactivates an entire layer (its color frame changes from emerald to grey to indicate this inactive status). Note that by removing an entire pathway, we make the network rely on the remaining intact pathway. This means that the errors one would expect are those associated with the properties of the *intact* pathway, not the lesioned one. For example, lesioning the direct pathway makes the network rely on semantics, allowing for the possibility of semantic errors to the extent that the semantic pathway doesn't quite get things right without the assistance of the missing direct pathway. Completely lesioning the semantic pathway itself does *not* lead to semantically related errors -- there is no semantic information left for such errors to be based on! 

For each of the errors, compare the word the network produced (closest_name) with the input word (trial_name). If the produced word is very similar orthographically (and phonologically) to the input word, this is called a *visual* error, because the error is based on the visual properties instead of the semantic properties of the word. The simulation automatically scores errors as visual if the input orthography and the response orthography (determined from the response phonology) overlap by two or more letters. You should see this reflected in the vis column in the grid view.

Now, let's try the direct pathway lesion and retest the network

The simulation does automatic coding of semantic errors, but they are somewhat more difficult to code because of the variable amount of activity in each pattern. We use the criterion that if the input and response semantic representations overlap by .4 or more as measured by the *cosine* or *normalized inner product* between the patterns, then errors are scored as semantic. The cosine goes from 0 for totally non-overlapping patterns to 1 for completely overlapping ones. The value of .4 does a good job of including just the nearest neighbors in the cluster plot of semantic relationships (). Nevertheless, because of the limited semantics, the automatically coded semantic errors do not always agree with our intuitions, which you might see if you check out the individual semantic errors in .

To summarize the results so far, we have seen that a lesion to the semantic pathway results in purely visual errors, while a lesion to the direct pathway results in a combination of visual and semantic errors. To a first order of approximation, this pattern is observed in surface and deep dyslexia, respectively. As simulated in the PMSP model, people with surface dyslexia are actually more likely to make errors on low-frequency irregular words, but we cannot examine this aspect of performance because frequency and regularity are not manipulated in our simple corpus of words. Thus, the critical difference for our model is that surface dyslexia does not involve semantic errors, while the deep dyslexia does. Visual errors are made in both cases.

# Reading with Partial Pathway Lesions

We next explore the effects of more realistic types of lesions that involve partial, random damage to the units in the various pathways, where we systematically vary the percentage of units damaged. There are six different lesion types, corresponding to damaging different layers in the semantic and direct pathways. For each type of lesion, one can specify the percent of units removed from the layer in question with the lesion_pct value. 

The first two lesion types damage the semantic pathway hidden layers (OS_Hid and SP_Hid), to simulate the effects of surface dyslexia. The next type damages the direct pathway (OP_Hid), to simulate the effects of phonological dyslexia, and at high levels, deep dyslexia. The next two lesion types damage the semantic pathway hidden layers again (OS_Hid and SP_Hid) but with a simultaneous complete lesion of the direct pathway, which corresponds to the model of deep dyslexia explored by Plaut & Shallice (1993). Finally, the last lesion type damages the direct pathway hidden layer again (OP_Hid) but with a simultaneous complete lesion of the semantic pathway, which should produce something like an extreme form of surface dyslexia. This last condition is included more for completeness than for any particular neuropsychological motivation.

## Semantic Pathway Lesions

You should observe that the network makes almost exclusively visual errors (like the network with a full semantic pathway lesion). Results with 25 samples per lesion level are shown in Figure dd.2 in [Dyslexia Details](/CCNBook/Language/Dyslexia_Details "wikilink") -- corresponding results from this model can be found in the BatchTestOutputData tab.

10 in the corresponding control panel), so to see the other data you have to use the scroll bar at the bottom of the graph -- drag that to the right to explore the full set of data, then return to the OS_HID case.}}

Your results should also show this general pattern of purely visual errors (or perhaps some "other" errors at high lesion levels), which is generally consistent with surface dyslexia, as expected. It is somewhat counterintuitive that semantic errors are not made when lesioning the semantic pathway, but remember that the intact direct pathway provides orthographic input directly to the phonological pathway. This input generally constrains the phonological output to be something related to the orthographic input, and it prevents any visually unrelated semantic errors from creeping in. In other words, any tendency toward semantic errors due to damage to the semantic pathway is preempted by the direct orthographic input. We will see that when this direct input is removed, semantic errors are indeed made.

You should observe lots of visual errors, but interestingly, the network also makes some semantic errors in this case. This is due to being much closer to the phonological output, such that the damage can have a more direct effect where incorrect semantic information influences the output.

Results across 25 repetitions can be found in [Dyslexia Details](/CCNBook/Language/Dyslexia_Details "wikilink") for these same semantic pathway lesions in conjunction with a complete lesion of the direct pathway. This corresponds to the type of lesion studied by Plaut & Shallice (1993) in their model of deep dyslexia. For all levels of semantic pathway lesion, we now see semantic errors, together with visual errors and a relatively large number of "other" (uncategorizable) errors. This pattern of errors is generally consistent with that of deep dyslexia, where all of these kinds of errors are observed. Comparing the effects of these lesions relative to the previous case, we see that the direct pathway was playing an important role in generating correct responses, particularly in overcoming the semantic confusions that the semantic pathway would have otherwise made.  

Figure dd.3 in [Dyslexia Details](/CCNBook/Language/Dyslexia_Details "wikilink") also shows the relative number of semantic errors for the concrete versus abstract words. One characteristic of deep dyslexia is that patients make more semantic errors on abstract words relative to concrete words.

## Direct Pathway Lesions

Figure dd.1 in [Dyslexia Details](/CCNBook/Language/Dyslexia_Details "wikilink") shows the effects of direct pathway lesions, both with and without an intact semantic pathway. Let's focus first on the case with the intact semantic pathway (the Full Sem graphs in the figure).

Notice that for smaller levels of damage more of the errors are visual than semantic. This pattern corresponds well with phonological dyslexia, especially assuming that this damage to the direct pathway interferes with the pronunciation of nonwords, which can presumably only be read via this direct orthography to phonology pathway. Unfortunately, we can't test this aspect of the model because the small number of training words provides an insufficient sampling of the regularities that underlie successful nonword generalization, but the large-scale model of the direct pathway described in the next section produces nonword pronunciation deficits with even relatively small amounts of damage.

Interestingly, as the level of damage increases, the model makes increasingly more semantic errors, such that the profile of performance at high levels of damage provides a good fit to deep dyslexia, which is characterized by the presence of semantic and visual errors, plus the inability to pronounce nonwords. The semantic errors result from the learning-based division of labor effect as described previously (Section 10.3.2 in the text). Furthermore, we see another aspect of deep dyslexia in this data, namely a greater proportion of semantic errors in the abstract words than in the concrete ones (especially when you add together semantic and visual + semantic errors). 

This case of partial direct pathway damage with a completely lesioned semantic pathway produces mostly visual and "other" errors.

