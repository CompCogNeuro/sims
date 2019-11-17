Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

This simulation explores how the development of PFC active maintenance abilities can help to make behavior more flexible, in the sense that it can rapidly shift with changes in the environment. The development of flexibility has been extensively explored in the context of Piaget's famous A-not-B task, where a toy is first hidden several times in one hiding location (A), and then hidden in a new location (B). Depending on various task parameters, young kids reliably reach back at A instead of updating to B.

Let's examine the network first . It has location, cover, and toy inputs, and gaze/expectation and reach outputs. The internal hidden layer maintains information over the delay while the toy is hidden using recurrent self-connections, and represents the prefrontal cortex (PFC) in this model.

Notice that there are three location units corresponding to locations , , and , represented in the location-based units in the network. Also, there are two cover (lid) input units corresponding to C1, the default cover type, and C2, a different cover type, and two toy units corresponding to T1, the default toy, and T2, a different toy type.

Each of the three input layers is fully connected to the hidden layer, and the hidden layer is fully connected to each of the two output layers. You can see that there is an initial bias for the same locations to be more strongly activating, with weights of .7, while other locations have only a .3 initial connection weight. Connections from the toy and cover units are relatively weak at .3. The hidden and output layers have self-recurrent excitatory connections back to each unit, which are initially of magnitude .25, but we can change this with the parameter in the . Stronger weights here will improve the network's ability to maintain active representations. We are starting out with relatively weak ones to simulate a young infant that has poor active maintenance abilities.

Now, let's examine the input data patterns that will be presented to the network.

Note that there are three types of trials (to see them all, scroll with the purple bar on the right using the red arrow), indicated by the three names used in the Group column:

-   (pre-trial or "practice" trials): these are presented at the start  of an experiment to induce infants to reach to the A location -- they only have a single time step delay between presentation of the item and the choice test.

-   : test trials to the A location, which only differ from ptrials in that the delay is longer.

-   : this is the key test condition, which is identical to the Atrials except the toy input is in the B location.

Each trial consists of five segments (rows in the table) corresponding to the experimental segments of an A-not-B trial as listed below. During each segment, patterns of activity are presented to the input units corresponding to the visible/salient aspects of the stimulus event; all other input units have no activity. The levels of input activity represent the salience of aspects of the stimulus, with more salient aspects (e.g. a toy that the experimenter waves) producing more activity.

1.  *start*: covers sit in place on the apparatus, out of reach for the infant, and before the experimenter draws infant's attention to a particular location (weak equal activation on locations and cover inputs).

2.  *toy presentation* (toy-pres): experimenter draws the infant's attention to a toy (e.g., waves it) and places it into one location in apparatus (one location more strongly active, and toy T1 active)

3.  *lid presentation* (lid-pres): experimenter further draws the infant's attention to the location while placing the lid over the toy location (toy fading out in activation while cover is more active, and location less active)

4.  *delay*: the apparatus sits (out of reach) with all covers in place (equal weak location and toy activation)

5.  *choice*: the experimenter makes the apparatus accessible (with covers in place) for the infant's response (reaching is possible/permitted only during this segment) (inputs more active than delay but same pattern; reach layer is disinhibited) 

Each of these trial types can be repeated multiple times, as can the events within the trial. In the version we will be running, the task will consist of four *pre-trials*, two *A* trials, and one *B* trial (in that exact sequence). In addition, the toy-pres inputs are presented three times during each trial, and the number of delay events is varied for 'A' and 'B' trials in order to vary the working memory "load."

Now, let's run the network. It will be much easier to tell what is going on in the network by looking at a grid display, rather than trying to watch each trial's activation as the network runs (but you are welcome to do so by stepping through the inputs).

When you do this, the network will run through an entire A-not-B experiment, and record the activations in the grid. The column tells you which event is being presented, and the remaining columns show the activations in each layer of the network after each event.

Notice that when the toy is presented during the events, the corresponding hidden 'A' location also has become activated due to spreading activations, and that the network also "looks" toward this location in the layer.

Furthermore, because Hebbian learning is taking place after each trial, those units that are coactive experience weight increases, which in turn increases the propensity of the network to activate the *A* location representations on subsequent trials.

You will now see the *A* testing trials, where the network's tendency to reach to the *A* location is assessed. Note that as a result of the Hebbian learning, the hidden and output units are even more active here than in the pretrials.

You can also try to find a recurrent weight value that allows the network to succeed with the longer 5 delay condition.

You should observe that the gaze and reach activations are now slightly dissociated on the B trial, reflecting the fact that the Gaze pathway is updated continuously, while the Reaching pathway has to wait until the end and is thus more sensitive to small amounts of decay.

Finally, there is an interesting effect that can occur with very weak recurrent weights, which do not allow the network to maintain the representation of even the *A* location very well on 'A' trials. Because the weight changes toward 'A' depend on such maintained activity of the 'A' units, these weight-based representations will be relatively weak, making the network perseverate less to 'A' than it would with slightly stronger recurrent weights.

You should see that there is a less strong 'A' response with the weaker recurrent weights (and also some residual activation of the 'B' units), meaning a less strong A-not-B error (and further analysis has confirmed that this is due to the amount of learning on the 'A' trials).

# References

Cohen, J. D., Dunbar, K., & McClelland, J. L. (1990). On the Control of Automatic Processes: A Parallel Distributed Processing Model of the Stroop Effect. Psychological Review, 97(3), 332–361.

Cohen, J. D., & Huston, T. A. (1994). Progress in the use of interactive models for understanding attention and performance. In C. Umilta & M. Moscovitch (Eds.), Attention and Performance XV (pp. 1–19). Cambridge, MA: MIT Press.

Cohen, J. D., & Servan-Schreiber, D. (1992). Context, cortex, and dopamine: A connectionist approach to behavior and biology in schizophrenia. Psychological Review, 99(1), 45–77.

Dunbar, K., & MacLeod, C. M. (1984). A horse race of a different color: Stroop interference patterns with transformed words. Journal of Experimental Psychology. Human Perception and Performance, 10, 622–639.

Glaser, M. O., & Glaser, W. R. (1983). Time course analysis of the Stroop phenomenon. Journal of Experimental Psychology. Human Perception and Performance, 8, 875–894.

MacLeod, C. M., & Dunbar, K. (1988). Training and Stroop-like interference: Evidence for a continuum of automaticity. Journal of Experimental Psychology. Learning, Memory, and Cognition, 14, 126–135.

