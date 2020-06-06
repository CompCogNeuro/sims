Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

This model simulates normal and disordered (dyslexic) reading performance in terms of a distributed representation of word-level knowledge across Orthography, Semantics, and Phonology. It is based on a model by [Plaut and Shallice (1993)](#references). Note that this form of dyslexia is *acquired* (via brain lesions such as stroke) and not the more prevalent developmental variety.

# Normal Reading Performance

Because the network takes a bit of time to train (for 250 epochs), we will just load in a pre-trained network to begin with.

* Do `Open Trained Wts` in toolbar.

For our initial exploration, we will just observe the behavior of the network as it "reads" the words presented to the orthographic input layer. Note that the letters in the input are ordered left-to-right, bottom to top.

* Press `Test Trial`.

You will see the activation flow through the network, and it should settle into the correct pronunciation and semantics for the first word, "tart" (the bakery food). In the left panel, `TrlPhon` shows the closest phonology pattern compared to what the network produced -- it should match the `TrlName` input (the second part of the name shows the repeated phonology output representation).

* Use the VCR `Time` buttons at the bottom right of the `NetView` to review the settling dynamics over time.

* Continue to `Test Trial` through a few more words, paying particular attention to the timing of when the Phonological layer gets active relative to the Semantic one (you can always review to see this more clearly).  Then, do `Test All` to test the remainder of the inputs.

* Click on the `TstTrlPlot` tab to see a record of the network's performance on the full set of words.  It should perform perfectly, so there isn't much to see.  Click on the `TstTrlLog` button in the left control panel, which pulls up a text table view of the same plot data.

The `ConAbs` column shows whether this item is concrete (*Con*) or abstract (*Abs*), and the columns after that indicate what type of error the network makes: `Vis` = visual errors, `Sem` = semantic errors, `VisSem` = both, `Blend` = not a clearly pronounced word, `Other` = some other hard-to-categorize error.  Concrete words have more distinctive features, whereas abstract words have fewer, which impacts their relative susceptibility to lesions.

> **Question 9.1:** Do you think the initial phonological activation is caused by the "direct" input via orthography or the "indirect" input via semantics? Did you see any cases where the initial phonological pattern is subsequently altered when the later input arrives?  Provide an example word where this happened.

# Reading with Complete Pathway Lesions

We next explore the network's ability to read with one of the two pathways to phonology removed from action. This relatively simple manipulation provides some insight into the network's behavior, and can be mapped onto two of the three dyslexias. Specifically, when we remove the semantic pathway, leaving an intact direct pathway, we reproduce the characteristics of **surface dyslexia**, where words can be read but access to semantic representations is impaired and visual errors are made. When we remove the direct pathway, reading must go through the semantic pathway, and we reproduce the effects of **deep dyslexia** by finding both semantic and visual errors. Note that phonological dyslexia is a milder form of deep dyslexia, which we explore when we perform incremental amounts of partial damage instead of lesioning entire pathways.

We begin by lesioning the semantic pathway.

* Click the `Lesion` button in the toolbar, and select `SemanticsFull` (leave Proportion = 0), and then do `Test Trial` (be sure *not* to hit `Init`, as this will initialize the weights -- just do `Open Trained Weights` and re-lesion if you do).

You should see that only the direct pathway is activated, but likely it will still be able to produce the correct phonology output.  This does not actually remove any units or other network structure; it just flips a "lesion" (`Off`) flag that (reversibly) deactivates an entire layer. Note that by removing an entire pathway, we make the network rely on the remaining intact pathway. This means that the errors one would expect are those associated with the properties of the *intact* pathway, not the lesioned one. For example, lesioning the direct pathway makes the network rely on semantics, allowing for the possibility of semantic errors to the extent that the semantic pathway doesn't quite get things right without the assistance of the missing direct pathway. Completely lesioning the semantic pathway itself does *not* lead to semantically related errors -- there is no semantic information left for such errors to be based on! 

* Do `Test All` to test all items, and look at the `TstTrlPlot` and `TstTrlLog` (just click the `Updt` button to update the table view).  You can see a sum of all the testing results in the `TstEpcLog` which can be clicked in the control panel. This records a new row for each Test All run, along with the lesion and proportion setting.

> **Question 9.2:** How many times did the network with only the direct pathway (SemanticsFull lesion) make a reading mistake overall (you can count the number of 1's in the various error columns, or look at the `TstEpcLog` sums, in the last row)?  Notice that the network does not produce any blend outputs, indicating that the phonological output closely matched a known word.

For each of the errors, compare the word the network produced (`Phon`) with the input word (`TrialName`). If the produced word is very similar orthographically (and phonologically) to the input word, this is called a *visual* error, because the error is based on the visual properties instead of the semantic properties of the word. The simulation automatically scores errors as visual if the input orthography and the response orthography (determined from the response phonology) overlap by two or more letters. You should see this reflected in the Vis column in the Table.

> **Question 9.3:** How many of the semantically lesioned network's errors were visual, broken down by concrete and abstract, and overall?

Now, let's try the direct pathway lesion and retest the network.

* Click `Lesion` and select `DirectFull`, then do `TestAll` again.  Click `Updt` in the tables to see the latest results.

> **Question 9.4:** What was the total number of errors this time, and how many of these errors were visual, how many were semantic, and how many were "other" for the concrete and abstract categories (as reported in `TstEpcLog`)?.

![Semantics Cluster Plot](fig_dyslex_sem_clust.png?raw=true "Cluster Plot of Semantics similarity structure")

**Figure 1:** Cluster plot of semantic similarity for words in the simple triangle model of reading and dyslexia, showing the major split between abstract (top) and concrete (bottom) clusters. Words that are semantically close (e.g., within the same terminal cluster) are sometimes confused for each other in simulated deep dyslexia.

The simulation does automatic coding of semantic errors, but they are somewhat more difficult to code because of the variable amount of activity in each pattern. We use the criterion that if the input and response semantic representations overlap by .4 or more as measured by the *cosine* or *normalized inner product* between the patterns, then errors are scored as semantic. The cosine goes from 0 for totally non-overlapping patterns to 1 for completely overlapping ones. The value of .4 does a good job of including just the nearest neighbors in the cluster plot of semantic relationships (Figure 1). Nevertheless, because of the limited semantics, the automatically coded semantic errors do not always agree with our intuitions, which you might see if you check out the individual semantic errors in the `TstTrlLog`.

To summarize the results so far, we have seen that a lesion to the semantic pathway results in purely visual errors, while a lesion to the direct pathway results in a combination of visual and semantic errors. To a first order of approximation, this pattern is observed in surface and deep dyslexia, respectively. As simulated in the PMSP model, people with surface dyslexia are actually more likely to make errors on low-frequency irregular words, but we cannot examine this aspect of performance because frequency and regularity are not manipulated in our simple corpus of words. Thus, the critical difference for our model is that surface dyslexia does not involve semantic errors, while the deep dyslexia does. Visual errors are made in both cases.

# Reading with Partial Pathway Lesions

We next explore the effects of more realistic types of lesions that involve partial, random damage to the units in the various pathways, where we systematically vary the percentage of units damaged. There are six different lesion types, corresponding to damaging different layers in the semantic and direct pathways. For each type of lesion, one can specify the percent of units removed from the layer in question with the lesion_pct value. 

The first two lesion types damage the semantic pathway hidden layers (OShidden and SPhidden), to simulate the effects of surface dyslexia. The next type damages the direct pathway (OPhidden), to simulate the effects of phonological dyslexia, and at high levels, deep dyslexia. The next two lesion types damage the semantic pathway hidden layers again (OShidden and SPhidden) but with a simultaneous complete lesion of the direct pathway, which corresponds to the model of deep dyslexia explored by [Plaut & Shallice (1993)](#references). Finally, the last lesion type damages the direct pathway hidden layer again (OPhidden) but with a simultaneous complete lesion of the semantic pathway, which should produce something like an extreme form of surface dyslexia. This last condition is included more for completeness than for any particular neuropsychological motivation.

## Semantic Pathway Lesions

* Do `Lesion` of type `OShidden` and `Proportion = .1`, and do `Test All` -- then play with different `Proportion` values.

![Semantics Partial Lesions](fig_dyslex_partles_sem.png?raw=true "Partial Semantics Lesions")

**Figure 2:** Partial semantic pathway lesions in the dyslexia model, with an intact direct pathway. Only visual errors are observed, and damage to the orthography to semantics hidden layer (OShidden) seems to be more impactful than the semantics to phonology (SPhidden) layer.

You should observe that the network makes almost exclusively visual errors (like the network with a full semantic pathway lesion). Results with 25 samples per lesion level are shown in Figure 2.

Your results should also show this general pattern of purely visual errors (or perhaps some "other" errors at high lesion levels), which is generally consistent with surface dyslexia, as expected. It is somewhat counterintuitive that semantic errors are not made when lesioning the semantic pathway, but remember that the intact direct pathway provides orthographic input directly to the phonological pathway. This input generally constrains the phonological output to be something related to the orthographic input, and it prevents any visually unrelated semantic errors from creeping in. In other words, any tendency toward semantic errors due to damage to the semantic pathway is preempted by the direct orthographic input. We will see that when this direct input is removed, semantic errors are indeed made.

* Do some `SPhidden` lesions at various `Proportion` levels, and then look at the `TstEpcLog` results for SPhidden lesions.

You should observe lots of visual errors, but interestingly, the network also makes a very small number of  semantic errors in this case. This is due to being much closer to the phonological output, such that the damage can have a more direct effect where incorrect semantic information influences the output.

* Next, skip ahead to the `OPhiddenDirectFull` and `SPhiddenDirectFull` cases.

![Semantics Partial Lesions + Direct Full](fig_dyslex_partles_semdir.png?raw=true "Partial Semantics + Direct Full Lesions")

**Figure 3:** Partial semantic pathway lesions with complete direct pathway lesions. The 0.0 case shows results for just a pure direct pathway lesion. Relative to this, small levels of additional semantic pathway damage can produce slightly higher rates of semantic errors.

Results across 25 repetitions can be found in Figure 3 for these same semantic pathway lesions in conjunction with a complete lesion of the direct pathway. This corresponds to the type of lesion studied by [Plaut & Shallice (1993)](#references) in their model of deep dyslexia. For all levels of semantic pathway lesion, we now see semantic errors, together with visual errors and a relatively large number of "other" (uncategorizable) errors. This pattern of errors is generally consistent with that of deep dyslexia, where all of these kinds of errors are observed. Comparing the effects of these lesions relative to the previous case, we see that the direct pathway was playing an important role in generating correct responses, particularly in overcoming the semantic confusions that the semantic pathway would have otherwise made.  

> **Question 9.5:** Compare the first bar in each graph of Figure 3 (corresponding to the case with only a direct pathway lesion, and no damage to the semantic pathway) with the subsequent bars, which include increasing amounts of semantic pathway damage: Does additional semantic pathway damage appear to be necessary to produce the semantic error symptoms of deep dyslexia?  Focus specifically on the `Semantic` and `Vis + Sem` errors.

Figure 3 also shows the relative number of semantic errors for the concrete versus abstract words. One characteristic of deep dyslexia is that patients make more semantic errors on abstract words relative to concrete words.

> **Question 9.6:** Is there evidence in Figure 3 that the model also makes a differential number of errors for concrete vs. abstract words?


## Direct Pathway Lesions


![Direct Partial Lesions](fig_dyslex_partles_direct.png?raw=true "Partial Direct Lesions")

**Figure 4:** Partial direct pathway lesions in the dyslexia model, either with or without an intact semantic pathway (Full Sem vs. No Sem, respectively). The highest levels of semantic errors (i.e., deep dyslexia) are shown with Full Sem in the abstract words, consistent with the simpler results showning this pattern with a full lesion of the direct pathway.

Figure 4 shows the effects of direct pathway lesions, both with and without an intact semantic pathway. Let's focus first on the case with the intact semantic pathway (the Full Sem graphs in the figure).

* Do `Lesion` of type `OPhidden` and try some different `Proportion` values (.1, .4, etc)

Notice that for smaller levels of damage more of the errors are visual than semantic. This pattern corresponds well with phonological dyslexia, especially assuming that this damage to the direct pathway interferes with the pronunciation of nonwords, which can presumably only be read via this direct orthography to phonology pathway. Unfortunately, we can't test this aspect of the model because the small number of training words provides an insufficient sampling of the regularities that underlie successful nonword generalization, but the large-scale model of the direct pathway described in the next section produces nonword pronunciation deficits with even relatively small amounts of damage.

Interestingly, as the level of damage increases, the model makes increasingly more semantic errors, such that the profile of performance at high levels of damage provides a good fit to deep dyslexia, which is characterized by the presence of semantic and visual errors, plus the inability to pronounce nonwords. The semantic errors result from the learning-based division of labor effect as described in the text. Furthermore, we see another aspect of deep dyslexia in this data, namely a greater proportion of semantic errors in the abstract words than in the concrete ones (especially when you add together semantic and visual + semantic errors). 

* Do `Lesion` of `OPhiddenSemanticFull`, with various Proportions.

This case of partial direct pathway damage with a completely lesioned semantic pathway produces mostly visual and "other" errors.

# References

Plaut, D. C., & Shallice, T. (1993). Deep dyslexia: A case study of connectionist neuropsychology. Cognitive Neuropsychology, 10(5), 377–500.

