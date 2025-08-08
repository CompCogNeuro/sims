*[Back to simulations](https://compcogneuro.org/simulations)*

# Introduction

This simulation illustrates how the prefrontal cortex (PFC) can produce top-down biasing for executive control, in the context of the widely studied Stroop task.  This model is similar to the [Cohen et al., (1994)](#references) version of the classic [Cohen et al, (1990)](#references) Stroop model, with bidirectional excitatory connections and inhibitory competition.

* Let's begin by exploring the connectivity of the StroopNet network using [[sim:Wts]] / [[sim:Wts/r.Wt]]

You will notice that all of the units for green (`g` = color, `G` = word) versus red (`r` = color, `R` = word) are connected in the way you would expect, with the exception of the connections between the hidden and output units (`gr` = green output, `rd` = red output). Although we assume that people enter the Stroop task with more meaningful connections than the random ones we start with here (e.g., they are able to say "Red" and not "Green" when they perceive red in the environment), we did not bother to preset these connections here because they become meaningful during the course of training on the task.

# Training

Next, let's look at the training environment patterns.

* Click on [[sim:sim-form/Train]] in the left control panel to view the patterns.

There are 4 different training input patterns shown in the window that opens up: `Gw` = reading the word "green", `Rw` = reading the word "red", `gc` = naming the green color, `rc` = naming the red color.  The frequency of these events is controlled by the last column in the table. You can see that the word reading events have a frequency of 6, while the color naming events are at 2. This frequency difference causes word reading to be stronger than color naming. Note that by using training to establish the strength of the different pathways, the model very naturally accounts for the benchmark [MacLeod & Dunbar (1988)](#references) training experiments.  Also, we never train the conflict conditions -- just the pure cases, assuming Stroop stimuli are rare in the real world.

Now, let's train the network.

* Do [[sim:Init]] and [[sim:Step]]: `Trial` (go back to viewing [[sim:Act]] / [[sim:Act/Act]]) to see a few training events -- standard error-driven learning is used to train the correct output for each case.  Click on [[sim:Train Epoch Plot]] and then [[sim:Run]] to train for 55 epochs. The network learns to produce the correct responses and [[sim:Train Epoch Plot/PctErr]] goes to zero.

* Go back to the `Network` view, select [[sim:Wts]] / [[sim:Wts/r.Wt]], and click on the `gr` and `rd` Output units. You can see that the network gradually learns to strengthen the connections to the correct output response for naming words *or* colors (it never sees both together during this pretraining).

We next explore the differential weight strengths for the two pathways that develop as a result of training. Although the observed weight differences are not huge, they are enough to produce the behavioral effects of word reading being dominant over color naming. 

* Click on the `g` and `G` Hidden units in turn.  You can hover the mouse over a given sending unit to see its weight value. (To confirm that you are hovering over the correct unit, you can check that the weight value that appears is consistent with the height and color of the bar shown, and look at the value in the brackets: [0,0] indicates the first unit in the layer, and [1,0] indicates the second unit in the layer.)

> **Question 9.4:** Report the weight values for the `g` and `G` hidden units from their respective input, PFC, and `gr` output units. This will be 6 weight values total. (Do not report the weights from the `rd` output units.)

<sim-question id="9.4">

You should have observed that the `G` (word reading) hidden unit has stronger weights overall, due to the higher frequency of training.

# Basic Stroop Task

Now, let's test the network on the Stroop task. First, we will view the testing events.

* Click on [[sim:Test]] in the left control panel to see the testing events.

You should see 6 rows, 3 for word reading and 3 for color naming. In the first 3 rows, you should see the *control, conflict*, and *congruent* conditions for word reading, all of which have the word reading PFC task unit clamped (the "wr" unit). All patterns have the "R" word unit active. *Control* does not have any active color units, *conflict* adds the `g` (green) color unit active, and *congruent* adds the "r" (red) color unit. In rows 4-6, you should see the color naming events. You should observe a similar pattern of inputs for color naming.

Now, we can actually test the network.

* Click on [[sim:Test Trial Plot]], switch the run mode to `Test` instead of `Train`, then [[sim:Init]], and then [[sim:Run]] to run all of the testing inputs.

You will see the response times `RT` plotted, which measures the number of cycles for the Output unit to get above the activation threshold of .5, labeled with the Control (Ctrl), Conflict (Conf), and Congruent (Cong) conditions, respectively.

![Stroop Data](fig_stroop_data.png?raw=true "Stroop Data")

**Figure 1:** Stroop data from [Dunbar & MacLeod (1984)](#references), showing the differential effect of Conflict on color naming compared to word reading.

If you compare this with the human data shown in the Figure 1, you will see that the model reproduces all of the important characteristics of the human data as described previously: interference in the conflict condition of color naming, the imperviousness of word reading to different conditions, and the overall slowing of color naming.

Now, we can single-step through the testing events to get a better sense of what is going on.

* First, click back to viewing the `Network`, and click on [[sim:Act]] / [[sim:Act/Act]] to view activations. Then, do [[sim:Init]] and then [[sim:Step]]: `Trial` (still in `Test` mode).  Review the activation settling using the `Time` VCR buttons on the lower right of the Network. Click [[sim:Step]]: `Trial` to go through each type of trial in the `Test` patterns, and review activation settling for each trial. 

Each Trial will advance one step through the three conditions of word reading in order (control, conflict, congruent) followed by the same for color naming. For the word reading conditions, you should observe that the corresponding word reading hidden unit is rapidly activated, and that this then activates the corresponding output unit, with little effect of the color pathway inputs. The critical condition of interest is the conflict color naming condition.

> **Question 9.5:** Describe what happens in the network during the conflict color naming condition, paying particular attention to the activations of the hidden units, and how this leads to the observed slowing of response time (settling).

<sim-question id="9.5">

# Effects of Frontal Damage

Now that we have seen that the model accounts for several important aspects of the normal data, we can assess the importance of the prefrontal (PFC) task units in the model by weakening their contribution to biasing the posterior processing pathways (i.e., the hidden layer units in the model). The strength of this contribution can be manipulated using a weight scaling parameter for the connections from the PFC to the Hidden layer. This parameter set by [[sim:From PFC]] in the control panel, and defaults to .3 -- we will reduce this to .25 to see an effect. Note that this reduction in the impact of the PFC units is functionally equivalent to the gain manipulation performed by [Cohen & Servan-Schreiber (1992)](#references).

* Reduce the [[sim:From PFC]] parameter in the ControlPanel from 0.3 to 0.25 and then do [[sim:Init]] (critical to apply the change and restart the testing) and [[sim:Run]] and look at the [[sim:Test Trial Plot]] to see the results.

You should see that the model is now significantly slower for the conflict color naming condition. This is the same pattern of data observed in frontal and schizophrenic patient populations [Cohen & Servan-Schreiber (1992)](#references). Thus, we can see that the top-down activation coming from the PFC task units is specifically important for the controlled-processing necessary to overcome the prepotent word reading response. Note that to fit the model to the actual patient response times, one must adjust for overall slowing effects that are not present in the model, and the perceptual / motor offsets. If you reduce the top-down strength even further (e.g., around 0.1), the network will start making errors in the color naming conflict condition (which you can see in the `Err` line in the plot -- you may need to unclick `RT` so the y-axis scales so you can view errors).  

Although we have shown that reducing the PFC gain can produce the characteristic behavior of frontal patients and people with schizophrenia, it is still possible that other manipulations could cause this same pattern of behavior without specifically affecting the PFC. In other words, the observed behavior may not be particularly diagnostic of PFC deficits. For example, one typical side effect of neurological damage is that overall processing is slower -- what if this overall slowing had a differential impact on the color naming conflict condition? To test this possibility in the model, let's increase the time constant parameter [[sim:Dt Vm Tau]] in the control panel, which determines the overall rate of activation updating in the model.

* Restore [[sim:From PFC]] to .3, and then increase [[sim:Dt Vm Tau]] to 40 (from 30). Do `Test` [[sim:Init]] and `Test` [[sim:Run]] again.

> **Question 9.6:** Compare the results of this overall slowing manipulation to the PFC gain manipulation performed previously. Does slowing also produce the characteristic behavior seen in frontal and schizophrenic patients? (To assess whether the conflict color naming condition is specifically affected, you should adjust for overall slowing effects by dividing the conflict color response time by the control color response time--the resulting value is 1.4 in the intact model and 1.6 in the model with the weakened contributions from the prefrontal units.)

# SOA Timing Data

![Stroop SOA Data](fig_stroop_soa_data.png?raw=true "Stroop SOA Data")

**Figure 2:** Stroop SOA timing data from Glaser & Glaser (1983).

Another important set of data for the model to account for are the effects of differential stimulus onset times as reported by [Glaser and Glaser (1983)](#references) (see Figure 2). To implement this test in the model, we simply present one stimulus for a specified number of cycles, and then add the other stimulus and measure the final response time (relative to the onset of the second stimulus). We use five different SOA (_stimulus onset asynchrony_) values covering a range of 20 cycles on either side of the simultaneous condition. For word reading, color starts out preceding the word by 20 cycles, then 16, 12, 8, and 4 cycles (all indicated by negative SOA), then color and word are presented simultaneously as in standard Stroop (0 SOA), and finally word precedes color by 4, 8, 12, 16, and 20 cycles (positive SOA). Similarly, for color naming, word initially precedes color (negative SOA), then word and color are presented simultaneously (0 SOA), and finally color precedes word (positive SOA). To simplify the simulation, we run only the most important conditions -- conflict and congruent.

* To run the SOA test, first view [[sim:Validate Trial Plot]] and then click on `Validate` for the run mode (we are co-opting the `Validate` mode for SOA testing here), and do [[sim:Init]] followed by [[sim:Run]].

The plot should display the response time (`RT`) as a function of SOA on the X axis, for the different conditions as shown.  Note that both conflict and congruent are superimposed for the Word reading case -- there is no perceptible difference.

By comparing the simulation data with the human data shown in the adjacent figure, you can see that the model's performance shows both commonalities and contrasts with the behavioral data. We first consider the commonalities. The model simulates several important features of the behavioral data. Most importantly, the model shows that word reading is relatively impervious to color conditions (conflict vs. congruent), even when the colors precede the word inputs, as indicated by the overlap of those lines. Thus, the dominant effect in the model is a strength-based competition -- the word reading pathway is sufficiently strong that even when it comes on later, it is relatively unaffected by competition from the weaker color naming pathway.

Another important feature of the human data captured by the model is the elimination of the interference effect of words on color naming when the color precedes the word by a relatively long time (right hand side of the graph). Thus, if the color pathway is given enough time to build up activation, it can drive the response without being affected by the word.

There are two important differences between the model and the human data, however. One difference is that, in humans, processing is relatively slowed across all conditions as the two inputs get closer to being presented simultaneously. This is particularly evident in the two word reading conditions and in the congruent color naming condition, in the upward slope from -20 to 0 SOA, followed by a downward slope from 0 to 20. This effect can be attributed to the effects of competition: when inputs are presented together, they compete with one another and thus slow processing. Given that we are using a fairly realistic form of inhibition, with the FFFB inhibition function, this suggests that some other mechanism may be at work to counteract these effects in the brain.

Another difference, which was present in that model, is the increasingly large interference effect for earlier word SOA's on color naming in the model, but not in people. It appears that people are somehow able to reduce the word activation if it appears sufficiently early, thereby minimizing its interfering effects. [Cohen, Dunbar, and McClelland (1990)](#references) suggested that people might be habituating to the word when it is presented early, reducing its influence. However, this explanation appears unlikely given that the effects of the early word presentation are minimal even when the word is presented only 100 msec early, allowing little time for habituation. Further, this model and other models still fail to replicate the minimal effects of early word presentation even when habituation (accommodation) is added to the models.

An alternative possibility is that the minimal effects of early word presentation reflect a strategic use of perceptual (spatially mediated?) attentional mechanisms that can be engaged after identifying the stimulus as a word. According to this account, once the word has been identified as such, it can be actively ignored, reducing its impact. Such mechanisms would not work when both stimuli are presented together because there would not be enough time to isolate the word without also processing its color.

# References

* Cohen, J. D., Dunbar, K., & McClelland, J. L. (1990). On the Control of Automatic Processes: A Parallel Distributed Processing Model of the Stroop Effect. Psychological Review, 97(3), 332–361.

* Cohen, J. D., & Huston, T. A. (1994). Progress in the use of interactive models for understanding attention and performance. In C. Umilta & M. Moscovitch (Eds.), Attention and Performance XV (pp. 1–19). Cambridge, MA: MIT Press.

* Cohen, J. D., & Servan-Schreiber, D. (1992). Context, cortex, and dopamine: A connectionist approach to behavior and biology in schizophrenia. Psychological Review, 99(1), 45–77.

* Dunbar, K., & MacLeod, C. M. (1984). A horse race of a different color: Stroop interference patterns with transformed words. Journal of Experimental Psychology. Human Perception and Performance, 10, 622–639.

* Glaser, M. O., & Glaser, W. R. (1983). Time course analysis of the Stroop phenomenon. Journal of Experimental Psychology. Human Perception and Performance, 8, 875–894.

* MacLeod, C. M., & Dunbar, K. (1988). Training and Stroop-like interference: Evidence for a continuum of automaticity. Journal of Experimental Psychology. Learning, Memory, and Cognition, 14, 126–135.

