Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

This simulation explores how the development of PFC active maintenance abilities can help to make behavior more flexible, in the sense that it can rapidly shift with changes in the environment. The development of flexibility has been extensively explored in the context of Piaget's famous A-not-B task, where a toy is first hidden several times in one hiding location (A), and then hidden in a new location (B). Depending on various task parameters, young kids reliably reach back at A instead of updating to B.

# Network and Inputs

Let's examine the network first.  It has `Location`, `Cover`, and `Toy` inputs, and `GazeExpect` (expectation) and `Reach` outputs. The internal `Hidden` layer maintains information over the delay while the toy is hidden using recurrent self-connections, and represents the prefrontal cortex (PFC) in this model.

Notice that there are three location units corresponding to locations A, B, and C, represented in the location-based units in the network. Also, there are two cover (lid) input units corresponding to C1, the default cover type, and C2, a different cover type, and two toy units corresponding to T1, the default toy, and T2, a different toy type.

* Click on `r.Wt` so we can examine the connectivity -- click on each of the three hidden units first, then the output units.

Each of the three input layers is fully connected to the hidden layer, and the hidden layer is fully connected to each of the two output layers. You can see that there is an initial bias for the same locations to be more strongly activating, with weights of .7, while other locations have only a .3 initial connection weight. Connections from the toy and cover units are relatively weak at .3. The Hidden and GazeExpect layers have self-recurrent excitatory connections back to each unit, which are initially of magnitude .4, but we can change this with the parameter `RecurrentWt`.  Stronger weights here will improve the network's ability to maintain active representations. We are starting out with relatively weak ones to simulate a young infant that has poor active maintenance abilities.

Now, let's examine the input data patterns that will be presented to the network.

* Click on the `AnotB Delay=3` (next to `Delay3Pats`).

Note that there are three types of trials (you likely need to scroll to see them all), indicated by the three names used in the `Group` column:

*  `ptrial`: (pre-trial or "practice" trials): these are presented at the start of an experiment to induce infants to reach to the A location -- they only have a single time step delay between presentation of the item and the choice test.

* `Atrial`: test trials to the A location, which only differ from ptrials in that the delay is longer.

* `Btrial`: this is the key test condition, which is identical to the Atrials except the toy input is in the B location.

Each trial consists of five segments (rows in the table) corresponding to the experimental segments of an A-not-B trial as listed below. During each segment, patterns of activity are presented to the input units corresponding to the visible/salient aspects of the stimulus event; all other input units have no activity. The levels of input activity represent the salience of aspects of the stimulus, with more salient aspects (e.g. a toy that the experimenter waves) producing more activity.

1. *start*: covers sit in place on the apparatus, out of reach for the infant, and before the experimenter draws infant's attention to a particular location (weak equal activation on locations and cover inputs).

2. *toy presentation* (toy-pres): experimenter draws the infant's attention to a toy (e.g., waves it) and places it into one location in apparatus (one location more strongly active, and toy T1 active)

3. *lid presentation* (lid-pres): experimenter further draws the infant's attention to the location while placing the lid over the toy location (toy fading out in activation while cover is more active, and location less active)

4. *delay*: the apparatus sits (out of reach) with all covers in place (equal weak location and toy activation)

5. *choice*: the experimenter makes the apparatus accessible (with covers in place) for the infant's response (reaching is possible/permitted only during this segment) (inputs more active than delay but same pattern; reach layer is disinhibited) 

Each of these trial types can be repeated multiple times, as can the events within the trial. In the version we will be running, the task will consist of four *pre-trials*, two *A* trials, and one *B* trial (in that exact sequence). In addition, the toy-pres inputs are presented three times during each trial, and the number of delay events is varied for 'A' and 'B' trials in order to vary the working memory "load."

# Effects of Delay and RecurrentWt

Now, let's run the network. It will be much easier to tell what is going on in the network by looking at a grid display, rather than trying to watch each trial's activation as the network runs (but you are welcome to do so by stepping through the inputs).

* Click on the `TrnTrlTable` tab. Then press the `Init`, `Train` buttons in the toolbar.

When you do this, the network will run through an entire A-not-B experiment, and record the activations in the table. The `TrialName` column tells you which event is being presented, and the remaining columns show the activations in each layer of the network after each event.

* Let's focus on the pre-trials first, at the top of the table.

Notice that when the toy is presented during the `p-toy-pres` events, the corresponding hidden 'A' location also has become activated due to spreading activations, and that the network also "looks" toward this location in the `GazeExpect` layer.

Furthermore, because Hebbian learning is taking place after each trial, those units that are coactive experience weight increases, which in turn increases the propensity of the network to activate the *A* location representations on subsequent trials.

* Next, scroll down past the 4 pre-trials so that the first A-trial event (`A-start`) is the first row displayed. (Just past midway through the table.)

You will now see the `A` testing trials, where the network's tendency to reach to the `A` location is assessed. Note that as a result of the Hebbian learning, the hidden and GazeExpect units are even more active here than in the pretrials.

* Now scroll down to make the first B-trial event the first one displayed.  You can also click on the `TrnTrlPlot` tab to see a plot of the three Reach activations across time, to get a better quantitative sense of the relative activation levels (`Reach_00` is the A unit, `Reach_01` is the B unit).

> **Question 10.4:** Describe what happens to the network's internal representations and output (gaze, reach) responses over the delay and choice trials for the B trials, and how this relates to Piaget's A-not-B data in infants.

* Now increase the `RecurrentWt` parameter in the ControlPanel to .7 from the default of .4, and `Init`, `Train`.

> **Question 10.5:** Describe how the network responds (i.e., in the gaze and reach outputs) this time, including a discussion of how the increased PFC (hidden) recurrent connection strength affected the network's behavior.

* Next, set `Delay` to `Delay5` and try it with the value of .7 for the recurrent weights.  You will have to scroll down a bit to see the final B test output. You may need to click `UpdtView`.

> **Question 10.6:** What happens on the 'B' trials with those two delays -- why does delay have this effect on the network's behavior?

You can also try to find a recurrent weight value that allows the network to succeed with the longer 5 delay condition.

* Now decrease the `RecurrentWt` parameter to a weaker value of .58 and set `Delay` back to `Delay3`, `Init`, `Train` and examine the B-trial again.

You should observe that the gaze and reach activations are now slightly dissociated on the B trial, reflecting the fact that the Gaze pathway is updated continuously, while the Reach pathway has to wait until the end and is thus more sensitive to small amounts of decay.

* A shorter delay allows infants to typically perform better (making fewer A-not-B errors) -- try going back to the original .4 recurrent weight value with the short delay condition of `Delay1`.

Finally, there is an interesting effect that can occur with very weak recurrent weights, which do not allow the network to maintain the representation of even the *A* location very well on 'A' trials. Because the weight changes toward 'A' depend on such maintained activity of the 'A' units, these weight-based representations will be relatively weak, making the network perseverate less to 'A' than it would with slightly stronger recurrent weights.

* To see this effect, set `Delay` back to `Delay3` and then reduce the `RecurrentWt` parameter to .1.  `Init`, `Train`, and look at the activations of the units in the 'B' choice trial. Then compare this with the case with of .4.  It can be easier to see the difference in the `TrnTrlPlot` -- click on the `Reach` button to pull up a dialog to configure the display of that line, and change the `TensorIdx` to 0 (which will show the `A` unit activation), and then to 1 (which will show the `B` unit activation).

You should see that there is a less strong 'A' response with the weaker recurrent weights (and also some residual activation of the 'B' units), meaning a less strong A-not-B error (and further analysis has confirmed that this is due to the amount of learning on the 'A' trials).


