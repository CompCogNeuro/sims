Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

This project explores the classic paired associates learning task in a cortical-like network, which exhibits catastrophic levels of interference as discovered by [McCloskey & Cohen (1989)](#references).

![AB-AC Data](fig_ab_ac_data_catinf.png?raw=true "AB-AC Data")

**Figure 1:** Data from people learning AB-AC paired associates, and comparable data from McCloskey & Cohen (1989) showing *catastrophic interference* of learning AC on AB.

The basic framework for implementing the AB--AC task is to have two input patterns, one that represents the "A" stimulus, and the other that represents the "list context" (see the Network). Thus, we assume that the subject develops some internal representation that identifies the two different lists, and that this serves as a means of disambiguating which of the two associates should be produced. These input patterns feed into a hidden layer, which then produces an output pattern corresponding to the "B" or "C" associate. We use a distributed representation of random bit patterns to represent the word stimuli.

# Training

First, let's look at the training environment.

* Press the [[sim:AB]] button next to the [[sim:AB patterns]] on the left panel, and then [[sim:AC patterns]] below it.

The `Input` column for each event shows the A item and the `Output` column is the associate (response). The `Context` column provides a "list context" for each event. Note that the first item in the AB list (`b0`) has the same A pattern as the first item in the AC list (`c0`). This is true for each corresponding pair of items in the two lists. The `Context` patterns for all the items on the AB list are all similar random variations of a common underlying pattern, and likewise for the AC items. Thus, these patterns are not identical for each item on the same list, just very similar (this reflects the fact that even if the external environmental context is constant, the internal perception of it fluctuates, and other internal context factors like a sense of time passing change as well).

Now, let's see how well the network performs on this task.

* You can start by doing [[sim:Init]] and then [[sim:Step]]: `Trial` for a few iterations of training to see how the network is trained (very standard minus-phase plus-phase dynamics -- use Time VCR buttons to rewind as usual). Then, click on the [[sim:Test Epoch Plot]] tab to see a plot of training performance, and change the Step to `Run` and [[sim:Step]] for the full Run.

You will see the plot updated as the network is trained, initially on the AB list, and then it automatically switches over to the AC list once error goes to 0. The `ABErr` and `ACErr` lines show the error as the number of list items incorrectly produced, for the AB and AC lists, with the criterion for correct performance being all units on the right side of .5. These are the results from testing on the full set of AB and AC patterns (without any learning during testing, which is not possible in people), after each epoch of training on the current training items (AB or AC). You can click on [[sim:Test Epoch Plot/PctErr]] to see the training error on the current training set.

* You should see the AB testing error go down, but then jump quickly back up, at the point when training switched over to AC, in general agreement with the [McCloskey & Cohen (1989)](#references) results. This  indicates that learning on the AC list (which gradually gets better) has interfered catastrophically with the prior learning on the AB list. Due to the use of inhibition and sparse representations, and a reasonably sized hidden layer, this model sometimes manages to retain some amount of the AB list, but its performance is highly variable and on average not as good as the human data.  Let's collect some statistics by running a batch of several training runs.

* Hit [[sim:Run]] to complete a run of 10 "subjects", and click on the [[sim:Train Run Plot]] to monitor results.

The statistics taken at the end of the AC list training for each "subject" will appear in the [[sim:Train Run Plot]].

* View the [[sim:RunStats Plot]] tab to see summary statistics across the 10 runs, including the mean and min final err for the AB list. This AB data is the overall measure of how much interference there was from training on the AC list.

> **Question 7.1:** Report the `ABErr:Mean` and `Min` in the [[sim:RunStats Plot]] for your batch run of 10 simulated subjects (hover over the points in the plot and report the final, y-axis value listed, or click on the `Table` to see a table of numerical values). Also do another [[sim:Init]] and [[sim:Step]]: `Run` while looking at the [[sim:Test Epoch Plot]] and report the general relationship between AC learning and AB interference across runs -- does AC generally show any significant learning before AB performance has mostly evaporated?

<sim-question id="7.1">

# Reducing Interference

Having replicated the basic catastrophic interference phenomenon, let's see if we can do anything to reduce the level of interference. Our strategy will be to retain the same basic architecture and learning mechanisms while manipulating certain key parameters. The intention here is to illuminate some principles that will prove important for understanding the origin of these interference effects, and how they could potentially be reduced -- though we will see that they have relatively small effects in this particular context.

* Click back on the [[sim:Network]] tab, and do [[sim:Init]], then [[sim:Step]]  `Trial` and pay attention to the overlap of patterns in the `Hidden` layer as you step.  Then do [[sim:Step]]: `Run` to retrain, and click to the [[sim:Test Epoch Plot]] tab to speed training, and then do [[sim:Reps Analysis]] which computes the similarity matrix, PCA plot, and cluster plot as we did with the Family Trees network in Chapter 4. Click on [[sim:Stats]] in the left panel, then the button on the `SimMats` `Hidden` row.

This brings up a similarity matrix of the Hidden layer activation patterns for each of the different AB and AC items.  Although you should see an overall increased similarity for AB items with themselves, and likewise for AC items, there is still considerable "red" similarity across lists, and in particular the items with the same A item (e.g., b0 and c0) tend to be quite similar. You can also see this in the `HiddenClust` cluster plot tab, where `b` and `c` items are grouped together in clusters.

This overlap seems obviously problematic from an interference perspective, because the AC list will activate and reuse the same units from the AB list, altering their weights to support the `C` associate instead of the `B`. Thus, by reducing the extent to which the hidden unit representations overlap (i.e., by making them *sparser*), we might be able to encourage the network to use separate representations for learning these two lists of items. 

* Let's test this idea by increasing the inhibition in the hidden layer using the [[sim:Hidden Inhib Gi]] parameter in the  control panel -- set it to 2.2 instead of 1.8.  Do an [[sim:Init]] and `Run` of 10 runs with these parameters.

This increased inhibition will make each activity pattern in the hidden layer smaller (fewer neurons active), which could result in less overlapping distributed representations.

> **Question 7.2:** Click the [[sim:RunStats Plot]] and report the resulting `AB Err:Mean` and `Min` statistics -- did this reduce the amount of AB interference?

<sim-question id="7.2">

Another thing we can do to improve performance is to enhance the contribution of the `Context` layer inputs relative to the A stimulus, because this list context disambiguates the two different associates.

* Do this by changing the `FmContext` parameter from 1 to 1.5.

This increases the weight scaling for Context inputs (i.e., by increasing the `WtScale.Rel` parameter (see [Input Scaling](https://github.com/emer/leabra#input-scaling) for details). We might imagine that strategic focusing of attention by the subject accomplishes something like this.

Finally, increased amounts of self-organizing learning might contribute to better performance because of the strong correlation of all items on a given list with the associated list context representation, which should be emphasized by Hebbian learning. This could lead to different subsets of hidden units representing the items on the two lists because of the different context representations.

* Do this by increasing `XCalLLrn` to .0005 instead of .0003.

Now let's see if performance is improved by making these three parameter adjustments.

* Do [[sim:Init]] and `Run` -- you can watch the `Train Run Plot` for results as they come in.

> **Question 7.3:** Click the [[sim:RunStats Plot]] and report the resulting `AB Err:Mean` and `Min` statistics -- did these parameters reduce the amount of AB interference?  Informal testing has shown that this is close to the best performance that can be obtained in this network with these parameters -- is it now a good model of human performance?

<sim-question id="7.3">

The final level of interference on the AB list tends to be quite variable.  You can go back and observe the [[sim:Test Epoch Plot]] during training to see that these manipulations have also slowed the onset of the interference somewhat. Thus, we have some indication that these manipulations are having an effect in the right direction, providing some support for the principle of using sparse, non-overlapping representations to avoid interference. 

Another parameter that theortically should be important, is the variance of the initial weights, which can encourage it to use *different* sets of units to represent the different associates.  You can increase this variance to explore its effects:

* Change [[sim:Wt Init Var]] from .25 to .4.

While this might produce some good results, it also does increase variability overall, and may not produce a net benefit.

<!---
A final, highly impactful manipulation is to increase the Hidden layer size significantly, and increase the inhibition a bit more, which gives a much sparser representation and also gives it more "room" to spread out across the larger population of neurons.  We have already made the Hidden layer relatively large (150 neurons) compared to the other layer sizes, but increasing it even further has increasing benefits.

* In the Network, click on the `Hidden` layer name, which pulls up an editor of the layer properties -- change the `Shp` size from 10 x 15 to 20 x 20 or 20 x 30, etc.  Then hit the `Build Net` button in the toolbar, increase `Hidden Inhib Gi` to 2.6 (you can't get away with this high of inhibition in a smaller hidden layer -- too few neurons are active and it fails to learn), and do `Init`, `Run` (switch back to `Train Run Plot`).  You should observe that there are finally cases where the model retains around 40% or so of its AB knowledge after training on AC. 
-->

One important dimension that we have not yet emphasized is the speed with which the network learns -- it is clearly not learning as fast (in terms of number of exposures to the list items) as human subjects do. Further, the manipulations we have made to improve interference performance have resulted in even longer training times (you can see this if you don't clear the graph view between runs with default and these new parameters). Thus, we could play with the [[sim:Lrate]] parameter to see if we can speed up learning in the network. 

* Keeping the same "optimal" parameters, increase the [[sim:Lrate]] to .1 (from .04), and do another [[sim:Init]] and `Run`.

Although the increase in learning rate successfully speeded up the learning process, it is still significantly slower than human learning on this kind of material.  However, in this network, we can push the rate higher and get fairly fast learning before interference starts to increase again -- at some point the higher learning rate results in higher interference as weights change too much.

* You can also run the [[sim:Reps Analysis]] again and compare the results with the better-performing model, to see that the patterns are indeed more separated.

We will see next that these same kinds of specializations we've been making in this model are exactly what the *hippocampus* uses to achieve very high levels of pattern separation, to enable rapid learning of new information.

# References

McCloskey, M., & Cohen, N. J. (1989). Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem. In G. H. Bower (Ed.), The Psychology of Learning and Motivation, Vol. 24 (pp. 109–164). San Diego, CA: Academic Press.

