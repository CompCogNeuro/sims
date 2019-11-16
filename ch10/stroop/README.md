Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

This simulation illustrates how the prefrontal cortex (PFC) can produce top-down biasing for executive control, in the context of the widely-studied Stroop task.


* Let's begin by exploring the connectivity of the StroopNet network using `r.Wt`

You will notice that all of the units for green (`g` = color, `G` = word) versus red (`r` = color, `R` = word) are connected in the way you would expect, with the exception of the connections between the hidden and output units (`gr` = green output, `rd` = red output). Although we assume that people enter the Stroop task with more meaningful connections than the random ones we start with here (e.g., they are able to say "Red" and not "Green" when they perceive red in the environment), we did not bother to preset these connections here because they become meaningful during the course of training on the task.

# Training

Next, let's look at the training environment patterns.

* Click on `Stroop Train` next to `TrainPats` view the patterns.

There are 4 different inputs: `Gw` = reading the word "green", `Rw` = reading the word "red", `gc` = naming the green color, `rc` = naming the red color.  The frequency of these events is controlled by the last column in the table. You can see that the word reading events have a frequency of 6, while the color naming events are at 2. This frequency difference causes word reading to be stronger than color naming. Note that by using training to establish the strength of the different pathways, the model very naturally accounts for the benchmark MacLeod & Dunbar, 1988 training experiments.  Also, we never train the conflict conditions -- just the pure cases, assuming Stroop stimuli are rare in the real world.

Now, let's train the network.

* Do `Init` and `Step Trial` to see a few training events -- standard error-driven learning is used to train the correct output for each case.  Click on `TrnEpcPlot` and then `Train` to train for 55 epochs.

You can see that the network gradually learns to strengthen the connections to the correct output response for naming words *or* colors (it never sees both together during this pretraining).

We next explore the differential weight strengths for the two pathways that develop as a result of training. Although the observed weight differences are not huge, they are enough to produce the behavioral effects of word reading being dominant over color naming. 

* View the network, select `r.Wt`, and click on the `g` and `G` Hidden units.  You can hover the mouse over a given sending unit to see its weight value.

> **Question 10.1:** Report the weights for the `g` and `G` hidden units from their respective input, PFC, and output units (you need only report the `gr` output weights).

You should have observed that the `G` (word reading) hidden unit has stronger weights overall, due to the higher frequency of training.

# Basic Stroop Task

Now, let's test the network on the Stroop task. First, we will view the testing events.

* Click on `Stroop Test` next to `TestPats` to see the testing events.

You should see 6 rows, 3 for word reading and 3 for color naming. In the first 3 rows, you should see the *control, conflict*, and *congruent* conditions for word reading, all of which have the word reading PFC task unit clamped (the "wr" unit). All patterns have the "R" word unit active. *Control* does not have any active color units, *conflict* adds the `g` (green) color unit active, and *congruent* adds the "r" (red) color unit. In rows 4-6, you should see the color naming events. You should observe a similar pattern of inputs for color naming.

Now, we can actually test the network.

* Click on `TstTrlPlot` and do `Test All`.

You will see the response times (cycles of settling) plotted in the graph, labeled with the Control (Ctrl), Conflict (Conf), and Congruent (Cong) conditions, respectively.

If you compare this with the human data shown in the Figure in the Executive Chapter, you will see that the model reproduces all of the important characteristics of the human data as described previously: interference in the conflict condition of color naming, the imperviousness of word reading to different conditions, and the overall slowing of color naming.

Now, we can single-step through the testing events to get a better sense of what is going on.

* First, click back to viewing the `NetView`, and click on `Act` to view activations. Then, do `Test Trial`.  Review the activation settling using the `Time` VCR buttons on the lower right of the NetView.

Each Trial will advance one step through the three conditions of word reading in order (control, conflict, congruent) followed by the same for color naming. For the word reading conditions, you should observe that the corresponding word reading hidden unit is rapidly activated, and that this then activates the corresponding output unit, with little effect of the color pathway inputs. The critical condition of interest is the conflict color naming condition.

> **Question 10.2:** Describe what happens in the network during the conflict color naming condition, paying particular attention to the activations of the hidden units, and how this leads to the observed slowing of response time (settling).

# Effects of Frontal Damage

Now that we have seen that the model accounts for several important aspects of the normal data, we can assess the importance of the prefrontal (PFC) task units in the model by weakening their contribution to biasing the posterior processing pathways (i.e., the hidden layer units in the model). The strength of this contribution can be manipulated using a weight scaling parameter for the connections from the PFC to the Hidden layer. This parameter is shown as the first row of the overall ControlPanel, with a default value of .45 -- we will reduce this to .4 to see an effect. Note that this reduction in the impact of the PFC units is functionally equivalent to the gain manipulation performed by Cohen & Servan-Schreiber (1992).

You should see that the model is now much slower for the conflict color naming condition. This is the same pattern of data observed in frontal and schizophrenic patient populations Cohen & Servan-Schreiber (1992). Thus, we can see that the top-down activation coming from the PFC task units is specifically important for the controlled-processing necessary to overcome the prepotent word reading response. If you reduce the top-down strength even further, the network will start making errors in the color naming conflict condition (which you can see by turning on the line on the graph view). Note that to fit the model to the actual patient response times, one must adjust for overall slowing effects that are not present in the model.

Although we have shown that reducing the PFC gain can produce the characteristic behavior of frontal patients and schizophrenics, it is still possible that other manipulations could cause this same pattern of behavior without specifically affecting the PFC. In other words, the observed behavior may not be particularly diagnostic of PFC deficits. For example, one typical side effect of neurological damage is that overall processing is slower -- what if this overall slowing had a differential impact on the color naming conflict condition? To test this possibility in the model, let's increase the time constant parameter in the , which determines the overall rate of settling in the model.

# SOA Timing Data

\[\[/File:fig_stroop_soa_data.png\|400px\|thumb\|right\|

<caption>
SOA Timing data Glaser and Glaser (1982).

</caption>
\]\]

Another important set of data for the model to account for are the effects of differential stimulus onset times as reported by Glaser and Glaser (1982) (see adjacent figure). To implement this test in the model, we simply present one stimulus for a specified number of cycles, and then add the other stimulus and measure the final response time (relative to the onset of the second stimulus). We use five different SOA (stimulus onset asynchrony) values covering a range of 20 cycles on either side of the simultaneous condition. For word reading, color starts out preceding the word by 20 cycles, then 16, 12, 8, and 4 cycles (all indicated by negative SOA), then color and word are presented simultaneously as in standard Stroop (0 SOA), and finally word precedes color by 4, 8, 12, 16, and 20 cycles (positive SOA). Similarly, for color naming, word initially precedes color (negative SOA), then word and color are presented simultaneously (0 SOA), and finally color precedes word (positive SOA). To simplify the simulation, we run only the most important conditions -- conflict and congruent.

The graph in the right panel should display the response time as a function of SOA on the X axis. The lines are as follows:

-   blue: color naming conflict
-   green: color naming congruent
-   red: word reading (both conflict and congruent are superimposed)

By comparing the simulation data with the human data shown in the adjacent figure, you can see that the model's performance shows both commonalities and contrasts with the behavioral data. We first consider the commonalities. The model simulates several important features of the behavioral data. Most importantly, the model shows that word reading is relatively impervious to color conditions (conflict vs. congruent), even when the colors precede the word inputs, as indicated by the similarity of the two dotted lines in the graph. Thus, the dominant effect in the model is a strength-based competition -- the word reading pathway is sufficiently strong that even when it comes on later, it is relatively unaffected by competition from the weaker color naming pathway.

Another important feature of the human data captured by the model is the elimination of the interference effect of words on color naming when the color precedes the word by a relatively long time (right hand side of the graph). Thus, if the color pathway is given enough time to build up activation, it can drive the response without being affected by the word.

There are two important differences between the model and the human data, however. One difference is that processing is relatively slowed across all conditions as the two inputs get closer to being presented simultaneously. This is particularly evident in the two word reading conditions and in the congruent color naming condition, in the upward slope from -20 to 0 SOA, followed by a downward slope from 0 to 20. This effect can be attributed to the effects of competition -- when inputs are presented together, they compete with one another and thus slow processing. Given that we are using a fairly realistic form of inhibition, with the FFFB inhibition function, this suggests that some other mechanism may be at work to counteract these effects in the brain.

Another difference, which was present in that model, is the increasingly large interference effect for earlier word SOA's on color naming in the model, but not in people. It appears that people are somehow able to reduce the word activation if it appears sufficiently early, thereby minimizing its interfering effects. Cohen, Dunbar, and McClelland (1990) suggested that people might be habituating to the word when it is presented early, reducing its influence. However, this explanation appears unlikely given that the effects of the early word presentation are minimal even when the word is presented only 100 msec early, allowing little time for habituation. Further, this model and other models still fail to replicate the minimal effects of early word presentation even when habituation (accommodation) is added to the models.

An alternative possibility is that the minimal effects of early word presentation reflect a strategic use of perceptual (spatially mediated?) attentional mechanisms that can be engaged after identifying the stimulus as a word. According to this account, once the word has been identified as such, it can be actively ignored, reducing its impact. Such mechanisms would not work when both stimuli are presented together because there would not be enough time to isolate the word without also processing its color.

