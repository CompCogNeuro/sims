Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

This model simulates normal and disordered (dyslexic) reading performance in terms of a distributed representation of word-level knowledge across Orthography, Semantics, and Phonology. It is based on a model by [Plaut and Shallice (1993)](#references). Note that this form of dyslexia is _acquired_ (via brain lesions such as stroke) and not the more prevalent developmental variety.

# Normal Reading Performance

Because the network takes a bit of time to train (for 250 epochs), we will just load in a pre-trained network to begin with. On each trial, this network was presented with one aspect of a word's representation (e.g., its orthography) and was trained to produce the other two representations (e.g., its phonology and semantics), so that the network ultimately learned to map from each word's visual form, speech output, or meaning to the other representations.  

* Do [[sim:Open Trained Wts]] in toolbar.

For our initial exploration, we will just observe the behavior of the network as it "reads" the words presented to the orthographic input layer. Note that the letters in the input are ordered left-to-right, bottom to top.

* Click the run mode to `Test` instead of `Train`, press [[sim:Init]] and [[sim:Step]] `Trial`.

You will see the activation flow through the network, and it should settle into the correct pronunciation and semantics for the first word, "tart" (the bakery food). You can click on the [[sim:Act]] -> [[sim:Act/Targ]] variable in the [[sim:Network]] to see the target outputs, and see that it made the correct response in the `Phonology` and `Semantics` layers based on the `Othography` input.

* Use the VCR `Time` buttons at the bottom right of the `Network` to review the settling dynamics over time.

* Continue to [[sim:Step]] `Trial` through a few more words, paying particular attention to the timing of when the Phonological layer gets active relative to the Semantic one within a trial (you can always review to see this more clearly).  Then, do [[sim:Run]] to test the remainder of the inputs.

> **Question 10.6:** Do you think the initial phonological activation for each word is caused by the "direct" input via orthography or the "indirect" input via semantics? Did you see any cases where the initial phonological pattern is subsequently altered when the later input arrives?  Provide an example word where this happened.

<sim-question id="10.6">

* Click on the [[sim:Test Trial Plot]] tab to see a record of the network's performance on the full set of words. It should only make one "Other" error for the word "flag", which it pronounces as "flaw".

[[sim:ConAbs]] shows whether this item is concrete (*Con*) or abstract (*Abs*) (`ConAbs`=0 for concrete, 1 for abstract). You can click the checkbox next to `ConAbs` to view it; you'll see that the second half of the list contains abstract words. Most of the checkboxes after `ConAbs` indicate what type of error the network makes: [[sim:Vis]] = visual errors, [[sim:Sem]] = semantic errors, [[sim:VisSem]] = both, [[sim:Blend]] = not a clearly pronounced word, [[sim:Other]] = some other hard-to-categorize error. (You may need to expand your browser to see the colors of the bars, or you can hover at the top of a bar to see what it represents.) Concrete words have more distinctive features, whereas abstract words have fewer, which impacts their relative susceptibility to lesions, as we'll see.

# Reading with Complete Pathway Lesions

We next explore the network's ability to read with one of the two pathways to phonology removed from action. This relatively simple manipulation provides some insight into the network's behavior, and can be mapped onto two of the three dyslexias. Specifically, when we remove the semantic pathway, leaving an intact direct pathway, we reproduce the characteristics of **surface dyslexia**, where words can be read but access to semantic representations is impaired and visual errors are made. When we remove the direct pathway, reading must go through the semantic pathway, and we reproduce the effects of **deep dyslexia** by finding both semantic and visual errors. Note that phonological dyslexia is a milder form of deep dyslexia, which we explore when we perform incremental amounts of partial damage instead of lesioning entire pathways.

We begin by lesioning the semantic pathway.

* Go back to the [[sim:Network]] tab and then click the [[sim:Lesion/Lesion]] button in the toolbar at the top (*not* in the control panel on the left, which just reports which lesion was performed) and select `Semantics full` (leave Proportion = 0), and then do [[sim:Init]] and [[sim:Step]] `Trial`.

You should see that only the direct pathway is activated, but likely it will still be able to produce the correct phonology output.  This does not actually remove any units or other network structure; it just flips a "lesion" (`Off`) flag that (reversibly) deactivates an entire layer. Note that by removing an entire pathway, we make the network rely on the remaining intact pathway. This means that the errors one would expect are those associated with the properties of the *intact* pathway, not the lesioned one. For example, lesioning the direct pathway makes the network rely on semantics, allowing for the possibility of semantic errors to the extent that the semantic pathway doesn't quite get things right without the assistance of the missing direct pathway. Completely lesioning the semantic pathway itself does *not* lead to semantically related errors -- there is no semantic information left for such errors to be based on! 

* Do [[sim:Run]] to test all items, and look at the [[sim:Test Trial Plot]] (you can click off all the bars and toggle each one on in turn to see them more clearly). You can see a sum of all the testing results in the [[sim:Test Epoch Plot]] tab. This records a new row for each Test Run run, along with the lesion and proportion setting.

> **Question 10.7:** How many times did the network with only the direct pathway (SemanticsFull lesion) make a reading mistake overall? (You can count the number of 1's in the [[sim:Test Trial Plot]] if you are viewing only the 5 types of errors, or you can look at the [[sim:Test Epoch Plot]] which shows the counts of different types of errors, and add those for the SemanticsFull Lesion case.)  Notice that the network does not produce any blend outputs, indicating that the phonological output closely matched a known word.

<sim-question id="10.7">

To understand the different types of errors, click on the [[sim:Test Trial]] tab, which shows the table of testing results per trial. For each of the categorized errors (i.e., where [[sim:Vis]] or [[sim:Other]] is 1; the SSE through Err columns show raw error and can be ignored here), compare the word the network produced (under [[sim:Phon]]) with the input word (under [[sim:TrialName]]). If the produced word is very similar orthographically (and phonologically) to the input word, this is called a *visual* error, because the error is based on the visual properties instead of the semantic properties of the word. The simulation automatically scores errors as visual if the input orthography and the response orthography (determined from the response phonology) overlap by two or more letters. You should see this reflected in the Vis column in the Table.

> **Question 10.8:** How many of the semantically lesioned network's errors were visual, broken down by concrete and abstract, and overall?

<sim-question id="10.8">

Now, let's try the direct pathway lesion and retest the network.

* Click [[sim:Lesion/Lesion]] in the toolbar at the top and select `Direct full`, then do [[sim:Init]] and [[sim:Run]] again. 

> **Question 10.9:** What was the total number of errors this time, and how many of these errors were visual, semantic, visual semantic, blend, and "other" for the concrete versus abstract categories (as reported in [[sim:Test Epoch Plot]])?.  You may need to click [[sim:Unfilter]] in the plot toolbar to get it to update.

<sim-question id="10.9">

![Semantics Cluster Plot](fig_dyslex_sem_clust.png?raw=true "Cluster Plot of Semantics similarity structure")

**Figure 1:** Cluster plot of semantic similarity for words in the simple triangle model of reading and dyslexia, showing the major split between abstract (top) and concrete (bottom) clusters. Words that are semantically close (e.g., within the same terminal cluster) are sometimes confused for each other in simulated deep dyslexia.

The simulation does automatic coding of semantic errors, but they are somewhat more difficult to code because of the variable amount of activity in each pattern. We use the criterion that if the input and response semantic representations overlap by .4 or more as measured by the *cosine* or *normalized inner product* between the patterns, then errors are scored as semantic. The cosine goes from 0 for totally non-overlapping patterns to 1 for completely overlapping ones. The value of .4 does a good job of including just the nearest neighbors in the cluster plot of semantic relationships (Figure 1). Nevertheless, because of the limited semantics, the automatically coded semantic errors do not always agree with our intuitions, which you might see if you check out the individual semantic errors in the [[sim:Test Trial]].

To summarize the results so far, we have seen that a lesion to the semantic pathway results in purely visual errors, while a lesion to the direct pathway results in a combination of visual and semantic errors. To a first order of approximation, this pattern is observed in surface and deep dyslexia, respectively. As simulated in the PMSP model, people with surface dyslexia are actually more likely to make errors on low-frequency irregular words, but we cannot examine this aspect of performance because frequency and regularity are not manipulated in our simple corpus of words. Thus, the critical difference for our model is that surface dyslexia does not involve semantic errors, while the deep dyslexia does. Visual errors are made in both cases.

# Reading with Partial Pathway Lesions

We next explore the effects of more realistic types of lesions that involve partial, random damage to the units in the various pathways, where we systematically vary the percentage of units damaged. There are six different lesion types, corresponding to damaging different layers in the semantic and direct pathways. For each type of lesion, one can specify the percent of units removed from the layer in question with the `proportion` value. 

The first two lesion types damage the semantic pathway hidden layers (OShidden and SPhidden), to simulate the effects of surface dyslexia. The next type damages the direct pathway (OPhidden), to simulate the effects of phonological dyslexia, and at high levels, deep dyslexia. The next two lesion types damage the semantic pathway hidden layers again (OShidden and SPhidden) but with a simultaneous complete lesion of the direct pathway, which corresponds to the model of deep dyslexia explored by [Plaut & Shallice (1993)](#references). Finally, the last lesion type damages the direct pathway hidden layer again (OPhidden) but with a simultaneous complete lesion of the semantic pathway, which should produce something like an extreme form of surface dyslexia. This last condition is included more for completeness than for any particular neuropsychological motivation.

## Semantic Pathway Lesions

* Do [[sim:Lesion/Lesion]] (in the toolbar at the top) of type `OShidden` and `Proportion = .1`, and do [[sim:Run]] -- then play with different `Proportion` values.

![Semantics Partial Lesions](fig_dyslex_partles_sem.png?raw=true "Partial Semantics Lesions")

**Figure 2:** Partial semantic pathway lesions in the dyslexia model, with an intact direct pathway. Only visual errors are observed, and damage to the orthography to semantics hidden layer (OShidden) seems to be more impactful than the semantics to phonology (SPhidden) layer.

You should observe that the network makes almost exclusively visual errors (like the network with a full semantic pathway lesion). Results with 25 samples per lesion level are shown in Figure 2.

Your results should also show this general pattern of purely visual errors (or perhaps some "other" errors at high lesion levels), which is generally consistent with surface dyslexia, as expected. It is somewhat counterintuitive that semantic errors are not made when lesioning the semantic pathway, but remember that the intact direct pathway provides orthographic input directly to the phonological pathway. This input generally constrains the phonological output to be something related to the orthographic input, and it prevents any visually unrelated semantic errors from creeping in. In other words, any tendency toward semantic errors due to damage to the semantic pathway is preempted by the direct orthographic input. We will see that when this direct input is removed, semantic errors are indeed made.

* Do some `SPhidden` lesions at various `Proportion` levels, and then look at the [[sim:Test Epoch Plot]] results for SPhidden lesions.

You should observe lots of visual errors, but interestingly, the network also makes a very small number of  semantic errors in this case. This is due to being much closer to the phonological output, such that the damage can have a more direct effect where incorrect semantic information influences the output.

* Next, skip ahead to the `OPhiddenDirectFull` and `SPhiddenDirectFull` cases.

![Semantics Partial Lesions + Direct Full](fig_dyslex_partles_semdir.png?raw=true "Partial Semantics + Direct Full Lesions")

**Figure 3:** Partial semantic pathway lesions with complete direct pathway lesions. The 0.0 case shows results for just a pure direct pathway lesion. Relative to this, small levels of additional semantic pathway damage can produce slightly higher rates of semantic errors.

Results across 25 repetitions can be found in Figure 3 for these same semantic pathway lesions in conjunction with a complete lesion of the direct pathway. This corresponds to the type of lesion studied by [Plaut & Shallice (1993)](#references) in their model of deep dyslexia. For all levels of semantic pathway lesion, we now see semantic errors, together with visual errors and a relatively large number of "other" (uncategorizable) errors. This pattern of errors is generally consistent with that of deep dyslexia, where all of these kinds of errors are observed. Comparing the effects of these lesions relative to the previous case, we see that the direct pathway was playing an important role in generating correct responses, particularly in overcoming the semantic confusions that the semantic pathway would have otherwise made.  

> **Question 10.10:** Compare the first bar in each graph of Figure 3 (corresponding to the case with only a direct pathway lesion, and no damage to the semantic pathway) with the subsequent bars, which include increasing amounts of semantic pathway damage: Does additional semantic pathway damage appear to be necessary to produce the semantic error symptoms of deep dyslexia?  Focus specifically on the `Semantic` and `Vis + Sem` errors.

<sim-question id="10.10">

Figure 3 also shows the relative number of semantic errors for the concrete versus abstract words. One characteristic of deep dyslexia is that patients make more semantic errors on abstract words relative to concrete words.

> **Question 10.11:** Is there evidence in Figure 3 that the model also makes a differential number of errors for concrete vs. abstract words?

<sim-question id="10.11">


## Direct Pathway Lesions


![Direct Partial Lesions](fig_dyslex_partles_direct.png?raw=true "Partial Direct Lesions")

**Figure 4:** Partial direct pathway lesions in the dyslexia model, either with or without an intact semantic pathway (Full Sem vs. No Sem, respectively). The highest levels of semantic errors (i.e., deep dyslexia) are shown with Full Sem in the abstract words, consistent with the simpler results showning this pattern with a full lesion of the direct pathway.

Figure 4 shows the effects of direct pathway lesions, both with and without an intact semantic pathway. Let's focus first on the case with the intact semantic pathway (the Full Sem graphs in the figure).

* Do [[sim:Lesion/Lesion]] of type `OPhidden` and try some different `Proportion` values (.1, .4, etc)

Notice that for smaller levels of damage more of the errors are visual than semantic. This pattern corresponds well with phonological dyslexia, especially assuming that this damage to the direct pathway interferes with the pronunciation of nonwords, which can presumably only be read via this direct orthography to phonology pathway. Unfortunately, we can't test this aspect of the model because the small number of training words provides an insufficient sampling of the regularities that underlie successful nonword generalization, but the large-scale model of the direct pathway described in the next section produces nonword pronunciation deficits with even relatively small amounts of damage.

Interestingly, as the level of damage increases, the model makes increasingly more semantic errors, such that the profile of performance at high levels of damage provides a good fit to deep dyslexia, which is characterized by the presence of semantic and visual errors, plus the inability to pronounce nonwords. The semantic errors result from the learning-based division of labor effect as described in the text. Furthermore, we see another aspect of deep dyslexia in this data, namely a greater proportion of semantic errors in the abstract words than in the concrete ones (especially when you add together semantic and visual + semantic errors). 

* Do [[sim:Lesion/Lesion]] of `OPhiddenSemanticFull`, with various Proportions.

This case of partial direct pathway damage with a completely lesioned semantic pathway produces mostly visual and "other" errors.

# References

* Plaut, D. C., & Shallice, T. (1993). Deep dyslexia: A case study of connectionist neuropsychology. Cognitive Neuropsychology, 10(5), 377–500.

