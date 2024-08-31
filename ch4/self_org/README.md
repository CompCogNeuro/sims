Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

This model illustrates how self-organizing learning emerges from the interactions between the following factors (as discussed in the *Learning* Chapter of the  [CCN Textbook](https://github.com/CompCogNeuro/book) ):

* **Inhibitory competition** -- only the most strongly driven neurons -- the ones whose synaptic weights best fit ("detect") the current input pattern -- get active. Other neurons may have some positive weights to the input but if they are less than the "winning" unit(s) they will get suppressed due to strong inhibition. 

* **Rich get richer** positive feedback loop -- the Hebbian learning rule ensures that only those neurons that actually get active are capable of learning (when receiver activity y = 0, then xy = 0 too, and the XCAL dWt function is 0 at 0). Thus, the neurons that already detect the current input sufficiently to get active  are the ones that get to further strengthen their ability to detect these inputs. This is the essential insight that Hebb had with why the Hebbian learning function should strengthen an "engram".

* **homeostasis** to balance the positive feedback loop and allow for weaker units to learn about other inputs. 
If left unchecked, the rich-get-richer dynamic ends up with a few units dominating everything (they get active, increase their weights, allowing them to get active for other somewhat overlapping patterns, which causes them to get more active for those, and so forth until all the inputs get categorized into one useless, overly broad category ("everything")). The homeostatic mechanism in our learning rule (review text on "XCAL" and "BCM") helps fight against this: the floating threshold is raised for neurons having high average activity (i.e., over multiple input patterns). This means that it will be more difficult for their weights to further increase, and more likely that their weights will decrease, restoring a balance. Conversely, neurons with very low average activity will obtain a lower floating threshold, making it easier for them to experience net weight increases that get them participating in representing other input patterns and competing more effectively.

The net result is the development of a set of neural detectors that relatively evenly cover the space of different inputs patterns, with systematic categories that encompass the statistical regularities. For example, cats like milk, and dogs like bones, and we can learn this just by observing the reliable co-occurrence of cats with milk and dogs with bones. This kind of reliable co-ocurrence is what we mean by "statistical regularity", and the homeostatic mechanism allows us to learn both sorts of assocations separately (otherwise we might get a representation that just links cats and dogs together with some weak connection to both milk and bones).  See *Hebbian Learning* Appendix in the [CCN Textbook](https://github.com/CompCogNeuro/book) for a very simple illustration of why Hebbian-style learning mechanisms capture patterns of co-occurrence. It is really just a variant on the basic maxim that "things that fire together, wire together".

In this exploration, the network learns about a simple world that consists purely of horizontal and vertical lines. Importantly the network never sees individual lines: they always appear in combination with som other lines (but which other line varies over experience). The clear objective of self-organizing learning in this case is to extract the underlying statistical regularity, so that the network can detect that the world is nevertheless made up of individual lines, which exist as reliable collections of pixels that come on and off together. It would be much more efficient to encode this world in terms of the lines, instead of individual pixels or instead of 'memorizing' the specific combinations of lines that it sees (we will see in a subsequent simulation why this is useful for generalization). 

# Learning

Let's take a look at the network. The 5x5 input projects to a hidden layer of 20 units, which are all fully connected to the input with random initial weights, and have standard FFFB inhibitory competition dynamics operating amongst them.

* As usual, select `Wts / r.Wt` and view the initialized weights for these units by clicking on several hidden units. You will see that the initial wts have been randomized to different values, with no systematicity. 

Let's see the environment the network will be experiencing.

* Click the `Lines2` button in the left control panel.

This will bring up a display showing the training items, which are composed of the elemental horizontal and vertical lines (to see just the lines themselves, click on `Lines1`). You can scroll through all 45 of the patterns. These 45 input patterns represent all unique pairwise combinations of vertical and horizontal lines. Thus, there are no real correlations between the lines (since every line appears with every other line equally frequently). The only reliable correlations are between the pixels that make up a particular line. To put this another way, each line can be thought of as appearing in a number of different randomly related contexts (i.e., with other lines).

It should be clear that if we computed the correlations between individual pixels across all of these images, everything would be equally (weakly) correlated with everything else. Thus, learning must be conditional on the particular type of line for any meaningful correlations to be extracted. We will see that this conditionality will simply self-organize through the interactions of the learning rule and the FFFB inhibitory competition. Note also that because two lines are present in every image, the network will require at least two active hidden units per input, if  each unit is to represent a particular line.

* Return to viewing `Act` in the `Network`. Then, do `Init` and `Step Trial` in the toolbar, to present a single pattern to the network. 

You should see one of the event patterns containing two lines in the input of the network, and a pattern of roughly two or more active hidden units (the FFFB inhibition is approximate in determining how many units are active).

* You can `Step Trial` some more. When you tire of single stepping, just switch from `Trial` to `Run` and do `Step Run`.  You should switch to viewing the `Train Epoch Plot` tab so it runs faster.

After 30 *epochs* (passes through all 45 different events in the environment) of learning, the network will stop. 

* Now select `Wts / r.Wt` again view the trained weights for these units by clicking on several hidden units. You should now see that some of the wts that were previously random have now learned to specialize on individual lines!
 
* To get a bigger picture view of all the weights, click the `Weights` tab which will display a grid view of all of the synaptic weights that were learned in the projection from the Input to the Hidden layer.

The larger-scale 5x4 grid is topographically arranged in the same layout as the `Hidden` layer of the network. Within each of these 20 grid elements is a smaller 5x5 grid representing the input units, showing the weights for each unit. By clicking on the hidden units in the network window with the `r.Wt` variable selected, you should be able to verify this correspondence.  You should see that the network has extracted the individual line elements from these input patterns.

* Go back to the `Weights` tab, set the `Step` to `Epoch`, and `Step` to see how these weight patterns evolved over the course of training, epoch-by-epoch.

As training proceeded, the weights came to more and more clearly reflect the lines present in the environment. Thus, individual units developed *selective* representations of the correlations present within individual lines, or two lines in some cases. The BCM-based XCAL learning algorithm does not alter weights from inactive inputs, so it tends to accumulate a bit of "cruft" (a historical trace of the learning process) in the weights, but the weights to the dominant inputs for each unit get very strong and stand out. This lack of learning to inactive inputs (which differs significantly from more standard forms of Hebbian learning) is not only biologically supported, but also significantly increases the overall storage capacity of the network, by reducing interference from prior learning.

These line representations developed as a result of the interaction between learning and inhibitory competition as follows. Early on, the units that won the inhibitory competition were those that happened to have larger random weights for the input pattern. Because those units were active, learning then tuned these weights to be more selective for that input pattern (not just a couple random units they happened to like in the first place, but now all the active input units), causing them to be more likely to respond to that pattern as a whole (and thus other patterns that also have  sharing one of the two lines). To the extent that the weights are stronger for one of the two lines in the input, the unit will be more likely to respond to inputs having this line, and the weights will continue to increase. If a unit gets over active, then its long-term average activity level, which sets the floating threshold for the BCM-style learning, will make it harder to keep learning about new patterns and it will decrease its weights, allowing other units to become active.  

The dynamics of the inhibitory competition are critical for the self-organizing effect, causing different units to specialize on different lines. Just as in Darwinian evolution, competition drives the pressure to specialize..

The net result of this self-organizing learning is a *combinatorial* distributed representation, where each input pattern is represented as the combination of the two line features present therein. This is the "obvious" way to represent such inputs, but you should appreciate that the network nevertheless had to discover this representation through the somewhat complex self-organizing learning procedure.

Another thing to notice in the weights shown in the grid view is that some units are obviously not selective for anything. These "loser" units (also known as "dead" units) were never reliably activated by any input feature, and thus did not experience much learning. It is typically quite important to have such units lying around, because self-organization requires some "elbow room" during learning to sort out the allocation of units to stable correlational features. Having more hidden units also increases the chances of having a large enough range of initial random selectivities to seed the self-organization process. The consequence is that you need to have more units than is minimally necessary, and that you will often end up with leftovers (plus the redundant units mentioned previously).

From a biological perspective, we know that the cortex does not produce a lot of new cortical neurons in adults, so we conclude that in general there is probably an excess of neural capacity relative to the demands of any given learning context. Thus, it is useful to have these leftover and redundant units, because they constitute a "reserve" that could presumably get activated if new features were later presented to the network (e.g., diagonal lines). We are much more suspicious ofrecisely tuned quantities  of hidden units to work properly (more on this later).

# Unique Pattern Statistic

Although looking at the weights is informative, we could use a more concise measure of how well the network's internal model matches the underlying structure of the environment. We one such measure is plotted in the `Train Epoch Plot` as the network learns.

This shows the results of the **unique pattern statistic** (`UniqPats`), which records the number of unique hidden unit activity patterns that were produced as a result of probing the network with all 10 different types of horizontal and vertical lines (presented individually). Thus, there is a separate testing process which, after each epoch of learning, tests the network on all 10 lines, records the resulting hidden unit activity patterns (with the FFFB inhibition cranked up to 3 so that typically 1 unit is active, so we can find the *most active* response to each input), and then counts up the number of unique such patterns (subject to thresholding so we only care about binary patterns of activity).

The logic behind this measure is that if each line is encoded by (at least) one distinct hidden unit, then this will show up as a unique pattern. If, however, there are units that encode two or more lines together (which is not as efficient of a model of this environment), then this will not result in a unique representation for these lines, and the resulting measure will be lower. Thus, to the extent that this statistic is less than 10, the internal model produced by the network does not fully capture the underlying independence of each line from the other lines. Note, however, that the unique pattern statistic does not care if *multiple* hidden units encode the same line (i.e., if there is redundancy across different hidden units) -- it only cares that the *same* hidden unit not encode *two different* lines.

Also, for most runs, if you use the lower level of inhibition used during training, there will always be a unique pattern of activity -- in the brain, distributed representations are much more efficient for encoding unique patterns via different patterns of active units -- so this `UniqPats` statistic is really a strict, simple measure of learning performance.

The performance of the model on any given run can be quite variable, depending on the random initial weights. Almost always the `UniqPats` statistic is above 5, and often it is a perfect 10, and typically it climbs up over training. Because of this variability, we need to run multiple runs of training to get a better sense of how well the network learns in general.

* Switch `Step` to `Run` and `Step` through multiple runs, each of which starts with different random initial weights.  Switch to viewing the `Train Run Plot` to see a record of the `UniqPats` statistic for each of the 8 runs. 

> **Question 4.1:** How many times across the 8 runs was there a less-than-10 result for the UniqPats number of uniquely represented lines, with the default parameters?

# Parameter Manipulations

Now, let's explore the effects of some of the parameters in the control panel.

One of the most important parameters for BCM-style Hebbian learning is how high the `AvgL` long-term running average threshold rises with increased neural activity (this is denoted by theta in the textbook figures).  If this value is 0, then no matter how high the long run average is, there will be no homeostatic effect on learning. The higher this value goes, the stronger the homeostasis force is that balances against the Hebbian rich-get-richer positive feedback loop, which can result in neurons that are more finely tuned to distinctive patterns, versus more broadly tuned neurons that respond to many different things.

* The parameter that controls how high `AvgL` goes is `AvgLGain`, with a default value of 2.5.  To see how important this factor is, reduce it to 1.5 and 1.  Press `Init` to have the parameter change take effect and restart the train process, and `Train Run` will run through all 8 runs.  Note the effects on the `UniqPats` and also on the learned synaptic weights.

It is also entertaining and informative to watch the `Learn / AvgL` value in the `Network`, to see how this updates over time and is affected by these parameter changes. When `AvgLGain` is low, you should see that only a few units will tend to dominate activity over trials, but when the homeostatic force is working well, the activity will be more evenly spread out.

One thing that is a bit unrealistic about this model is the lack of any activity at all in the units that are off. In the real brain, inactive neurons always have some low level of activity. This can affect the extent to which weights decrease to the less active inputs, potentially leading to cleaner overall patterns of weights.

* To add some noise activity in the input, set the InputNoise to .2, and `Init` and `Train Run` (you can `Step Trial` to see the noise in the input).  

> **Question 4.2:** Now how many sub-10 `UniqPats` stats did you get? Is this an improvement over the no-noise case? (This effect should be easier to see if you leave the AvgLGain to be reduced at 1.5 or 1 compared to default of 2.5). Describe what difference you observe in the weights of the no-noise and noise simulations. Why do you see this difference?

In conclusion, this exercise should give you a feel for the dynamics that underlie self-organizing learning, and also for the importance of how the floating threshold level and homeostasis dynamic plays a key role in this form of learning.


