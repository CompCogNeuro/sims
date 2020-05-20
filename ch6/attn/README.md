Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

This simulation illustrates how object recognition (ventral, what) and spatial (dorsal, where) pathways interact to produce spatial attention effects, and accurately capture the effects of brain damage to the spatial pathway.

The model is based on [Cohen, et al. (1994)](#references) with several improvements in terms of its overall biologically-based structure and the scope of data which it can explain.

# Network Structure

Let's step through the network structure and connectivity, which was completely pre-specified (i.e., the network was not trained, and no learning takes place, because it was easier to hand-construct this simple architecture). As you can see, the network has mutually interconnected *Spatial* and *Object* pathways feeding off of a V1-like layer that contains a spatially mapped feature array. In this simple case, we're assuming that each "object" is represented by a single distinct feature in this array, and also that space is organized along a single dimension. Thus, the first row of units represents the first object's feature (which serves as the cue stimulus) in each of 7 locations, and the second row represents the second object's feature (which serves as the target) in these same 7 locations.

* Now select `r.Wt` in the NetView and click on the object and spatial units to see how they function via their connectivity patterns.

The object processing pathway has a sequence of 3 increasingly spatially invariant layers of representations, with each unit collapsing over 3 adjacent spatial locations of the object-defining feature in the layer below. Note that the highest, fully spatially invariant level of the object pathway plays the role of the output layer, and is used for measuring the reaction time to detect objects. This happens by stopping settling whenever the *target* output (object 2) gets above an activity of .6 (if this doesn't happen, settling stops after 220 cycles).

The spatial processing pathway has a sequence of two layers of spatial representations, differing in the level of spatial resolution. As in the object pathway, each unit in the spatial pathway represents 3 adjacent spatial locations, but unlike the object pathway, these units are not sensitive to particular features. Two units per location provide distributed representations in both layers of the spatial pathway. This redundancy will be useful for demonstrating the effects of partial damage to this pathway.

# Perceiving Multiple Objects

Although much of the detailed behavioral data we will explore with the model concerns the Posner spatial cueing task, we think the more basic functional motivation for visual attention is to facilitate object recognition when multiple objects are presented simultaneously.  Therefore, we will start with a quick exploration of the network's object recognition capacities as a function of the spatial distribution of two objects. This will provide an introduction to the kinds of interactions between spatial and object processing that can happen using this relatively simple model. Let's begin by viewing the events that we will present to the network.

* Click on the `MultiObjs` button to view the events (the one that is below the `Test` line that also says `MultiObjs`)a.

You should see 3 events. The first event has two different objects (features) present in different spatial locations. Note that the target object has slightly higher activation (i.e., it is more salient), which will result in the reliable selection of this object over the other. The next event has two of the same objects (targets) presented in different locations. Finally, the last event has the two different objects in the same spatial location. As the figure makes clear, recognizing objects when they overlap in the same location is considerably more difficult than when they appear in different locations. Although it is clearly easier to recognize objects if only copies of the same object are present as opposed to different objects, this difference is not likely to be significant for small numbers of presented objects.

Now, let's test these predictions in the model.

* Switch back to viewing `Act` in the `NetView`. Do `Init` and `Test Trial`.

This will present the first event to the Network, which will stop settling (i.e., updating the network's activations a cycle at a time) when the target unit's activation exceeds the threshold of .5 in the Output layer.  You should see that the network relatively quickly focuses its spatial attention on the more active input on the right side, and the Object pathway represents that target item, causing the Output activity to get over threshold relatively quickly.

* Then `Test Trial` through the remaining events.

You should have seen that while the network settled relatively quickly for the first two events, it was slowed on the third event where the objects overlap in the same region of space (occasionally not so slow on the last one).

* Click on the `TstTrlPlot` to see a plot of the settling times (number of cycles to reach threshold) for each event.

* You can also set the `ViewUpdt` setting to `Cycle` instead of `FastSpike` and, click back on the `NetView`, and `Test Trial` back through the items, to see a cycle-by-cycle update of the network.  You can use the VCR rewind buttons at the lower right of the NetView to rewind through and see exactly how the network settling played out.

* To get a better sense of the overall data pattern, click back on `TstTrlPlot` and do `Test All` a few times.  There is a small amount of noise so the results should be a little bit different each time, but overall quite consistent.

You should see that overall the network has more difficulty with the objects appearing in the same spatial location, where spatial attention cannot help focus on one object. The overall average cycles by condition are reported in the `TstStats` table -- click the `etable.Table` button there to answer this question:

> **Question 6.6:** Report the `Cycle:Mean` values for each condition from the `TstStats` table.

You should have observed that spatial representations can facilitate the processing of objects by allocating attention to one object over another. The key contrast condition is when both objects lie in the same location, so that spatial attention can no longer separate them out, leaving the object pathway to try to process both objects simultaneously.

# The Posner Spatial Cuing Task

![Posner Task](fig_posner_task.png?raw=true "Posner Task")

**Figure 1:** The Posner spatial cueing task, widely used to explore spatial attention effects. The participant is shown a display with two boxes and a central fixation cross -- on some trials, one of the boxes is cued (e.g., the lines get transiently thicker), and then a target appears in one of the boxes (or not at all on catch trials). The participant just presses a key when they first detect the target. Reaction time is quicker for valid cues vs. invalid ones, suggesting that spatial attention was drawn to that side of space. Patients with hemispatial neglect exhibit slowing for targets that appear in the neglected side of space, particularly when invalidly cued.

Now, let's see how this model does on the Posner spatial cuing task, diagrammed in Figure 1.

* Set `Test` to `StdPosner` and click `Init`.  Then click the `StdPosner` table button to see the events.

There are three *groups* of events shown here, which correspond to a *Neutral* cue (no cue), a *Valid* cue, and an *Invalid* cue. There is just one event for the neutral case, which is the presentation of the target object in the left location. For the valid case, the first event is the cue presented on the left, followed by a target event with the target also on the left. The invalid case has the same cue event, but the target shows up on the right (opposite) side of space. The network's activations are not reset between the cue and target events within a group, but they are reset at the end of a group (after the target).  Thus, residual activation from the cuing event can persist and affect processing for the target event, but the activation is cleared between different trial types.

* Do `Test Trial` to process the *Neutral* trial, the the Cue / Target sequence for the Valid and Invalid trials.  You should set the `ViewUpdt` to `Cycle` to see the detailed settling dynamics on the probe trials (The Cue is not shown in detail). Go back through the history and note how the network responds to each of the three conditions of the Posner task, as you continue to Step through these cases.

* Then, switch to `TstTrlPlot` and do several `Test All` runs to collect some statistics.  Then click on  `TstStats` table (close any existing such windows and be sure to get the new one -- it regenerates a new table every time so the existing ones will not update).

> **Question 6.7:** How does the influence of the spatial cue affect subsequent processing of the target, in terms of the settling times on each condition? Report average data per condition/group from the `TstStats` table.

Typical reaction times for young adults (i.e., college students) on this task are roughly: neutral, 370 ms; valid 350 ms; invalid 390 ms, showing about 20 ms on either side of the neutral condition for the effects of attentional focus. These data should agree in general with the pattern of results you obtained (the invalid effect is a bit higher), but to fit the data more closely you would have to add a constant offset of roughly 310 ms to the number of cycles of settling for each trial type. This constant offset can be thought of as the time needed to perform all the other aspects of the task that are not included in the simulation (e.g., generating a response). Note also that one cycle of settling in the network corresponds with one millisecond of processing in humans. This relationship is not automatic -- we adjusted the time constant for activation updating (`Act.Dt.VmTau` = 7 instead of the default of 3.3) so that the two were in agreement in this particular model.

## Effects of Spatial Pathway Strength

Now let's explore the effects of the parameters in the on the network's performance, which helps to illuminate how the network captures normal performance in this task. First, let's try reducing `SpatToObj` from 2 to 1.5, then 1, which reduces the influence of the two spatial pathway layers on the corresponding object pathway layers.

* Do a few `Test All` runs with `SpatToObj` set to 1.5, then reduce to 1.  The parameter changes take effect when you hit the `Test All` button so no need to hit `Init` (which resets the plot, if you want).

You should find that there is less of an effect for both the valid and invalid conditions. This is just what one would expect -- because there is less of a spatial attention effect on the object system, the invalid cue does not slow it down as much, and similarly it has less effect for the valid case.

* Set `SpatToObj` back to 2, and reduce the influence from the V1 layer to the spatial representations by reducing the value of `V1ToSpat1` from 0.6 to 0.55, then to 0.5, doing a several `Test All` runs for each and noting the cycle averages.

Here the most interesting effect is in the invalid trials, where the network takes an increasing number of cycles as `V1ToSpat1` values drop, ultimately not being able to settle on the right answer with in the 220 max cycle limit. As V1 has less effect on the spatial pathway, it becomes more and more difficult for input in a novel location (e.g., the target presented on the opposite side of the cue) to overcome the residual spatial activation in the cued location. This shows that the spatial pathway needs to have a balance between sensitivity to bottom-up inputs and ability to retain a focus of spatial attention over time.  This network allows this balance to be set separately from that of the influence of the spatial pathway on the object pathway (controlled by the `SpatToObj` parameter), which is not the case with the [Cohen, et al. (1994)](#references) model.

* Set `V1ToSpat1` back to 0.6 (or hit the `Defaults` button) before continuing.


## Close Posner and Retinal Eccentricity

One additional manipulation we can make is to the eccentricity (visual distance) of the cue and target. If they are presented closer together, then one might expect to get less of an attentional effect, or even a facilitation if the nearby location was partially activated by the cue.

* Set `Test` to `ClosePosner`, and do `Init`, `Test Trial` while watching the `NetView`, and then `Test All` a few times while looking at the `TstTrlPlot`.

You will see that an overlapping set of spatial representations is activated, and the plot reveals that there is no longer a very reliable slowing for the invalid case relative to the neutral case.

* Set `Test` back to `StdPosner` before continuing.

## Effects of Spatial Pathway Lesions

As we mentioned earlier, [Posner et al. (1984)](#references) showed that patients who had suffered lesions in one hemisphere of the parietal cortex exhibit differentially impaired performance on the invalid trials of the Posner spatial cuing task. Specifically, they are slower when the cue is presented on the side of space processed by the intact hemisphere (i.e., ipsilateral to the lesion), and the target is then processed by the lesioned hemisphere. The patients showed a 120 ms difference between invalid and valid cases, with a validly cued RT of roughly 640 ms, and an invalidly cued RT of 760 ms. These data can be compared to matched (elderly) control subjects who showed a roughly 60 ms invalid-valid difference, with a validly cued RT of 540 ms and an invalidly cued RT 600 ms.

You should notice that, as one might expect, older people are slower than the young adult normals, and older people with some kind of brain damage are still slower yet, due to generalized effects of the damage. In this case, we are interested in the specific involvement of the parietal lobes in these attentional phenomena, and so we have to be careful to dissociate the specific from the generalized effects. To determine if the patients have a specific attentional problem, we must first find a way of *normalizing* the data so that we can make useful comparisons among these different groups (including the model). We normalize by dividing the elderly control's data by a constant factor to get the same basic numbers reported for the adult normals (or the model). If there is a specific effect of the brain damage, we should find that the pattern of reaction times is different from the adult normals even when it is appropriately normalized.

To find the appropriate scaling factors, we use the ratio between the valid RT's for the different groups. Ideally, we would want to use the neutral case, which should be a good measure of the overall slowing, but only the valid and invalid trial data are available for the patients and elderly controls. So, to compare the elderly controls with the adult normals, we take the adult valid RT's of 350 ms, and divide that by the elderly control subjects valid RT's of 540 ms, giving a ratio of .65.  Now, we multiply the elderly controls invalid RT's (600 ms) by this factor, and we should get something close to the adult normals invalid RT's (390 ms). Indeed, the fit is perfect -- 600 \* .65 = 390. The elderly controls thus appear to behave just like the adult normals, but with a constant slowing factor.

However, when we apply this normalizing procedure to the patient's data, the results do not fit well. Thus, we again divide 350 ms by the the 640 ms valid RT's, giving a ratio of .55. Then, we do 760 \* .55 = 418, which is substantially slower than the 390 ms invalid times for the adult normals. This makes it clear that the patients are specifically slower in the invalid trials even when their overall slowing has been taken into account by the normalizing procedure. This differential slowing is what led to the hypothesis that these patients have difficulty disengaging attention.

Now, we lesion the model, and see if it simulates the patient's data. However, because the model will not suffer the generalized effects of brain damage (which are probably caused by swelling and other such factors), and because it will not "age," we expect it to behave just like a adult subject that has only the specific effect of the lesion.  Thus, we compare the model's performance to the normalized patient values. Although we can add in the 310 ms constant to the models' settling time to get a comparable RT measure, it is somewhat easier to just compare the difference between the invalid and valid cases, which subtracts away any constant factors.

* To lesion the model, click the `Lesion` button and select `LesionSpat12` for the `Layers`, and leave the `Locations` and `Units` at `LesionHalf` -- this will lesion only the right half of the spatial layers, and only 1 out of the 2 spatial units in each location (i.e., a partial lesion). Select `r.Wt` in the network view and confirm that these units (the back 2 units on the right for `Spat1`, and the back right unit for `Spat2`) have their weights zeroed out. `Init`, `Test Trial` while watching the `NetView`, and then `Test All` while watching the `TstTrlPlot`.  Then click on  `TstStats` table (close any existing such windows and be sure to get the new one -- it regenerates a new table every time so the existing ones will not update -- also it reports stats over everything shown in the Plot, so make sure you did `Init` to only get results from the lesioned model).

> **Question 6.8:** Report the resulting averages from the `TstStats` table.

> **Question 6.9:** Compute the invalid-valid difference, and compare it first with that same difference in the intact network, and then with the patient's data as discussed above.

You should have found that you can simulate the apparent disengage deficit without having a specific "disengager" mechanism (at least qualitatively).

## Reverse Posner

One additional source of support for this model comes from the pattern of patient data for the opposite configuration of the cuing task, where the cue is presented in the lesioned side of space, and the invalid target is thus presented in the intact side. Interestingly, data from [Posner et al. (1984)](#references) clearly show that there is a very reduced invalid-valid reaction time difference for this condition in the patients. Thus, it appears that it is easier for the patients to switch attention to the intact side of space, and therefore less of an invalid cost, relative to the normal control data.  Furthermore, there appears to be less of a valid cuing effect for the patients when the cue and target are presented on the damaged side as compared to the intact side. Let's see what the model has to say about this.

* Set `Test` to `ReversePosner`, and do `Init`, `Test Trial` while watching the `NetView`, and then `Test All` a few times while looking at the `TstTrlPlot`.


You should see that the network shows a reduced difference between the valid and invalid trials compared to the intact network. Thus, the cue has less of an effect -- less facilitation on valid trials and less interference on invalid ones. This is exactly the pattern seen in the [Posner et al. (1984)](#references) data. In the model, it occurs simply because the stronger intact side of space where the target is presented has less difficulty competing with the damaged side of space where the cue was presented. In contrast, the disengage theory would predict that the lesioned network on the reverse Posner task should perform like the intact network on the standard Posner task.  Under these conditions, any required disengaging abilities should be intact (either because the network has not been lesioned, or because the cue is presented on the side of space that the lesioned network should
be able to disengage from).

# Balint's Syndrome

As mentioned previously, additional lesion data comes from *Balint's syndrome* patients, who suffered from *bilateral* parietal lesions. The most striking feature of these patients is that they have *simultanagnosia* -- the inability to recognize multiple objects presented simultaneously (see Farah, 1990 for a review). Interestingly, when such subjects were tested on the Posner task [(Coslett & Saffran, 1991)](#references), they exhibited a *decreased* level of attentional effects (i.e., a smaller invalid-valid difference). As emphasized by [Cohen et al. (1994)](#references), these data provide an important argument against the *disengage* explanation of parietal function offered by Posner and colleagues, which would instead predict bilateral slowing for invalid trials (i.e., difficulty disengaging). The observed pattern of data falls naturally out of the model we have been exploring.

* To simulate this condition, first set `Test` back to `StdPosner`, and then do `Lesion` with `Locations` set to `LesionFull` instead of Half (keep Units at `LesionHalf`).  Do `Init`, `Test Trial` while watching the `NetView`, and then `Test All` a few times while looking at the `TstTrlPlot`.  Then click on  `TstStats` table.

> **Question 6.10:** Report the results of the `TstStats` for the bilaterally lesioned network.

Finally, we can explore the effects of a more severe lesion to the parietal spatial representations, which might provide a better model of the syndrome known as *hemispatial neglect* (typically referred to as just *neglect*). As described previously, neglect results from unilateral lesions of the parietal cortex (usually in the right hemisphere), which cause patients to generally neglect the lesioned side of space. We simulate neglect by doing a similar lesion to the unilateral one we did before, but by doing FULL for to lesion both of the units in each location.

* Do `Lesion` with `Locations` at `LesionHalf` but `Units` at `LesionFull`.  Run a few `Test All` with `StdPosner`.

You will see a strong *neglect* phenomenon, which makes the network completely incapable of switching attention into the damaged side of space to detect the target (resulting in the full 220 cycles of settling for the invalid case).

* Now change `Test` to `MultiObj` and do `Init`, `Test Trial` while watching the `NetView`.

Observe that even when the more salient object is in the lesioned side of space, the network still focuses attention on the intact side. Thus, it is specifically neglecting this lesioned side. In the first case, this causes the network to activate the cue object representation, which does not stop the network settling, resulting in a full 220 cycles of settling.

Interestingly, if one does the ReversePosner case, then all attentional effects are completely eliminated, so that the settling times are relatively similar in all three conditions. This is not because the network is incapable of processing stimuli in the damaged side of space -- by looking at the activations in the network you can see that it does process the cue. Instead, the target presented to the good side of space has no difficulty competing with the weak residual representation of the cue in the damaged side. Competition can explain the general tendency for neglect on the grounds that it is very rare to actually have no other competing stimuli (which can be relatively weak and still win the competition), coming into the intact side of space, so that attention is usually focused on the intact side.

The smaller level of damage that produces the slowed target detection times in the Posner task may be more closely associated with the phenomenon of *extinction*, in which patients with unilateral parietal lesions show neglect only when there is a relatively strong competing visual stimulus presented to the good side of space (e.g., the cue in the invalid trials of the Posner task). Thus, the model may be able to account for a wide range of different spatial processing deficits associated with parietal damage, depending on both the severity and location of damage.

# Temporal Dynamics and Inhibition of Return

Another interesting aspect of the Posner spatial cuing task has to do with the temporal dynamics of the attentional cuing effect. To this point, we have ignored these aspects of the task by assuming that the cue activation persists to the point of target onset. This corresponds to experimental conditions when the target follows the cue after a relatively short delay (e.g., around 100 ms). However, the Posner task has also been run with longer delays between the cue and the target (e.g., 500 ms), with some interesting results. Instead of a facilitation effect for the valid trials relative to the invalid ones, the ordering of valid and invalid trials actually reverses at the long delays [(Maylor, 1985)](#references). This phenomenon has been labeled *inhibition of return*, to denote the idea that there is something that inhibits the system from returning attention to the cued location after a sufficient delay.

Our model can be used to simulate at least the qualitative patterns of behavior on the Posner task over different delays. This is done by varying the length of cue presentation (a variable delay event could have been inserted, but residual activation would persist anyway, and varying the cue length is simpler), and turning on the sodium-gated potassium (KNa) adaptation current, which causes neurons that have been active for a while to "fatigue". Thus, if the cue activation persists for long enough, those spatial representations will become fatigued, and if attention is subsequently directed there, the network will actually be slower to respond.  Also, because the spatial activations have fatigued, they no longer compete with the activation of the other location for the invalidly cued trials, eliminating the slowing.

Now, let's see this in the model.

* First, un-lesion the network by running `Lesion` and selecting `NoLesion` for `Layers`. Next, set `Test` to `StdPosner`. Then, click the `KNaAdapt` toggle to On to turn on adaptation. Next, let's choose a cue duration that, even with the accommodation channels active, still produces the original pattern of results. Set `CueDur` to 50. Now, do `Init`, `Test Trial` and `Test All` as usual.

You should observe the now-familiar pattern of a valid facilitation and an invalid slowing (although a bit weaker).

* `Test All` with increasing durations (change using the `CueDur` field) in increments of 50 from 50 to 300.

You should see that the valid-invalid difference decreases progressively with increasing duration, and ultimately, the validly cued condition can actually be a bit *slower* than the invalidly cued one, which is the hallmark of the inhibition of return phenomenon (Figure 8.28). The effect sizes here are fairly small because the form of adaptation here is relatively weak -- a more significant GABA-B like delayed inhibition effect (which is not currently implemented) would be needed to produce more substantial effects.

* Switch to the `NetView`, set `ViewUpdt` to `Cycle` and `Test Trial` through the running of the network with `CueDur` at 300.

> **Question 6.11:** Report in detail what happens on the valid and invalid trials that produces the inhibition of return effect.


# Object-Based Attentional Effects

So far, we have explored spatially mediated attentional effects. However, the very same mechanisms (and model) can be used to understand object-based attentional effects. For example, instead of cuing one region of space, we can cue one object, and then present a display containing the cue object and another different object, and determine which of the two objects is processed more readily. By analogy with the Posner spatial cuing paradigm, we would expect that the cued object would be processed more readily than the non-cued one. Of course, one would have to use different, but similar cue and target objects to rule out a target detection response based on the cue itself.

Because a simple object recognition task is problematic, the object-based attention studies that have been run experimentally typically involve a comparison between two operations (e.g., detecting a small visual target) on one object versus two operations on two different objects. If the spatial distances associated with these operations are the same in the two conditions, any difference in reaction time would indicate a cost for switching attention between the two objects. Such an object cost has been found in a number of studies [(Duncan, 1984; Vecera & Farah, 1994; Mozer et al., 1992)](#references).

In the simulator, we can run the simpler cuing experiment analogous to the Posner task because we have transparent access to the internal representations, and can measure the object processing facilitation directly.

* Set `Test` to `ObjAttn`, do `Init`, and click on `ObjAttn` to see these patterns.

The first event is a control condition, where we present two objects without any prior cuing. Note that, as in the MultiObj case, the target object is more strongly activated than the cue object, so the network will process the target object. The next event is a cuing event, where the cue object is presented in the central location. Then, within the same group so that activations persist, the next event presents the two objects just as in the first event. Thus, if the prior object cue is effective, it should be able to overcome the relatively small difference in bottom-up salience between the two objects, so that the network processes the cue object and not the target. Finally, the next two events are for the case where the two objects appear in the same location. Recall that before, the network was unable to select either object for processing in this case, because they are spatially overlapping. Perhaps now, with the object-based attentional cue, the network will be able to focus on the cue object.

* Make sure `KNaAdapt` is off and `CueDur` is back to 100 (or hit `Defaults`), and then do `Init`, `Test All`, then rewind through time to see how the network responds in detail to each task condition.

You should observe that the prior object cue is indeed capable of influencing subsequent processing in favor of the same object. Note also that the spatial system responds to this in the appropriate manner -- it activates the spatial location associated with the cued object. Finally, note that the top-down object cue is sufficient to enable the system to select one object (even the less active one) when the two objects are presented overlapping in the same location.


# References

Cohen, J. D., Romero, R. D., Farah, M. J., & Servan-Schreiber, D. (1994). Mechanisms of Spatial Attention: The Relation of Macrostructure To Microstructure in Parietal Neglect. *Journal Of Cognitive Neuroscience, 6(4),* 377–387.

Coslett, H. B., & Saffran, E. (1991). Simultanagnosia. To see but not two see. Brain, 114, 1523-1545.

Duncan, J. (1984). Selective attention and the organization of visual information. *Journal of Experimental Psychology: General, 113,* 501–517.

Farah, M. J. (1990). Visual Agnosia. Cambridge, MA: MIT Press.

Maylor, E. (1985). Facilitatory and inhibitory components of orienting in visual space. In M. I. Posner & Marin, O. S. M. (Eds.), Attention and Performance XI. Hillsdale, NJ: Lawrence Erlbaum Associates.

Mozer, M. C., Zemel, R. S., Behrmann, M., & Williams, C. K. I. (1992). Learning to segment images using dynamic feature binding. *Neural Computation, 4,* 650–665.

<a name="PosnerEtAl84"></a>
Posner, M. I., Walker, J. A., Friedrich, F. J., & Rafal, R. D. (1984). Effects of Parietal Lobe Injury on Covert Orienting of Visual Attention. *Journal of Neuroscience, 4,* 1863–1874.

Vecera, S. P., & Farah, M. J. (1994). Does visual attention select objects or locations? *Journal of Experimental Psychology: General, 123,* 146–160.
