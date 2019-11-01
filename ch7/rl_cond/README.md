Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

This simulation explores the temporal differences (TD) reinforcement learning algorithm under some basic Pavlovian conditioning environments.

To explore the TD learning rule, we use a simple classical (Pavlovian) conditioning task, where the network learns that a stimulus (tone) reliably predicts a subsequent reward. Then, we extend the task to a *second order conditioning* case, where another stimulus (light) reliably predicts that tone. First, we need to justify the use of the TD algorithm in this context, and motivate the nature of the stimulus representations used in the network.

You might recall that we said that the delta rule (aka the Rescorla-Wagner rule) provides a good model of classical conditioning, and thus wonder why TD is needed. It all has to do with the issue of *timing*. If one ignores the timing of the stimulus relative to the response, then in fact the TD rule becomes equivalent to the delta rule when everything happens at one time step (it just trains V(t) to match r(t)). However, animals are sensitive to the timing relationship, and, more importantly for our purposes, modeling this timing provides a particularly clear and simple demonstration of the basic properties of TD learning.

The only problem is that this simple demonstration involves a somewhat unrealistic representation of timing. Basically, the stimulus representation has a distinct unit for each stimulus for each point in time, so that there is something unique for the TD system to learn from. This representation is the **complete serial compound** (CSC) proposed by {{&lt; incite "SuttonBarto90" &gt;}}, and we will see exactly how it works when we look at the model. As we have noted, we explore a more plausible alternative in the [Executive Function](/CCNBook/Executive "wikilink") chapter, where the TD error signal controls the updating of a context representation that maintains the stimulus over time.

# Mapping TD to the Network

There are four separate TD layers that each compute a part of the overall TD algorithm, and the actual TD computations are performed by the simulation code, instead of using network-level interactions among the layers themselves. Although it is possible to get units to compute the necessary additions and subtractions required by the TD algorithm, it is much simpler and more robust to perform these calculations directly using the values represented by the units. The critical network-level computation is learning about the reward value of stimuli, and this is done using a special dopamine-like learning rule in the `TDRewPred` layer.

Here is a description of what each layer does:

*  **ExtRew** -- just represents the external reward input -- it doesn't do any computation, but just provides a way to display in the network what the current reward value is. It gets input from the input data table, representing the US (unconditioned stimulus). The left unit represents a 0 reward, and the right unit represents a 1 reward, while the middle unit represents no reward info avail.

*  **TDRewPred** -- this is the key learning layer, which learns to predict the reward value on the next time step based on the current stimulus inputs: V(t+1). This prediction is generated in the *plus phase* of Leabra settling based on its current weights from the `Input` layer (a linear unbounded activation rule is used, so this unit can represent arbitrary values), whereas in the minus phase the layer's state is clamped to the prediction made on the previous trial: V(t).

*  **TDRewInteg** -- this layer integrates the reward prediction and external reward layer values, and the difference in its plus-minus phase activation states are what drive the TD delta (dopamine-like) signal. Specifically, its minus phase activation is V(t) -- the expectation of reward computed by the rew pred layer *on the previous trial*, and its plus phase activation is the expectation of reward *on the next trial* plus any actual rewards being received at the present time. Thus, its plus phase state is the sum of the ExtRew and TDRewPred values, and this sum is directly clamped as an activation state on the unit.

* **TD** -- this unit computes the plus - minus values from the rew integ layer, which reflects the TD delta value and is thought to act like the dopamine signal in the brain.

# The Network

Let's start by examining the network. The `Input` layer (located at the top, to capture the relative anatomical locations of this cortical area relative to the midbrain dopamine system represented by the TD layers below it) contains three rows of 20 units each. This is the CSC, where the rows each represent a different stimulus (A, B, C), and the columns represent points in time: each unit has a stimulus and a time encoding (e.g., A_10 = time step 10 of the A stimulus). The TD layers are as
described above.

# The Basic TD Learning Mechanism

Let's see how the CSC works in action.

Nothing should happen in the `Input` layer, because no stimulus is present at time step 0. The various TD layers will remain at 0 as well, and the `TD` layer also has a zero activation. Thus, no reward was either expected or obtained, and there is no deviation from expectation. Note the `trial_name:` field shown above the network -- it indicates the time step (e.g., t=0).

This input activation represents the fact that the conditioned stimulus (CS) A (i.e., the "tone" stimulus) came on at t=10. There should be no effect of this on the TD layers, because they have not associated this CS with reward yet.

You will see that this stimulus remains active for 6 more time steps (through t=15), and at the end of this time period, the `ExtRew` layer represents a value of 1 instead of 0, indicating that an external reward was delivered to the network. Because the `TDRewPred` layer has not learned to expect this reward, the TD delta value is positive, as reflected by the activity of the TD unit, and as plotted on the graph above the network, which shows a spike at this time step 15. This TD spike is also drove learning in the `TDRewPred` layer, as we'll see the next time we go through this sequence of trials.

You should see that the `TDRewPred` layer now gets weakly activated at time step 14, signaling a prediction of reward that will come on the next time step. This expectation of reward, even in the absence of a delivered reward on the `ExtRew` layer (which still shows a 0 value representation), is sufficient to drive the TD "dopamine spike" as shown on the graph.

Now the situation is reversed: the `ExtRew` layer shows that the reward has been presented, but the TD value is 0. This is because the `TDRewPred` layer accurately predicted this reward on the prior time step, and thus the reward prediction error, which TD signals, is zero! In terms of the overall graph display, you can see that the "dopamine spike" of TD delta has moved forward one step in time. This is the critical feature of the TD algorithm: by learning to anticipate rewards one time step later, it ends up moving the dopamine spike earlier in time.

You should see that the spike moves "forward" in time with each training step, but can't move any further than the onset of the CS at time step 10.

We can also examine the weights to see what the network has learned.

# Extinction and Second Order Conditioning

At this point, there are many standard phenomena in classical conditioning that can be explored with this model. We will look at two: *extinction* and *second order conditioning*. Extinction occurs when the stimulus is no longer predictive of reward -- it then loses its ability to predict this reward (which is appropriate). Second order conditioning, as we discussed earlier, is where a conditioned stimulus can serve as the unconditioned stimulus for another stimulus -- in other words, one can extend the prediction of reward backward across two separate stimuli.

We can simulate extinction by simply turning off the reward that appears at t=15. To do this, we need to alter the parameters on the control panel that determine the nature of the stimulus input and reward. The first parameter is the `env_type`, which determines which stimuli are being presented (CS A, CS B, and US). Currently, we are presenting the CSA and US. To turn off the US, select . For this to take effect, you need to (or `Gen Inputs` button at the bottom of the ), which uses the parameters to generate an environment. The other parameters shown determine when the stimuli are presented, if they are selected by the parameter. We can leave these in their current state.

Now, let's explore second order conditioning. We must first retrain the network on the stimulus 1 association.

Now, we will turn on the CS B stimulus, which starts at t=2 and lasts until time step 10.

Essentially, the CSA stimulus *acts just like a reward* by triggering a positive delta value, and thus allows the CSB stimulus to learn to predict this first stimulus.

You will see that the early anticipation of reward gets carried out to the onset of the CS B stimulus (which comes first in time).

Finally, we can present some of the limitations of the CSC representation. One obvious problem is capacity -- each stimulus requires a different set of units for all possible time intervals that can be represented. Also, the CSC begs the question of how time is initialized to zero at the right point so every trial is properly synchronized. Finally, the CSC requires that the stimulus stay on (or some trace of it) up to the point of reward, which is unrealistic. This last problem points to an important issue with the TD algorithm, which is that although it can learn to bridge temporal gaps, it requires some suitable representation to support this bridging (which we explore in [Executive Function](/CCNBook/Executive "wikilink")).

# Advanced Explorations

More advanced explorations can be performed by manipulating the extra input patterns in the program found under `.programs` in the left browser panel. Here you can manipulate the probabilities of stimuli being presented, and introduce randomness in the timings. Generally speaking, these manipulations tend to highlight the limitations of the CSC input representation, and of TD more generally, but many of these are addressed by more advanced approaches (e.g., representing the sensory state in more realistic ways; Ludvig, Sutton & Kehoe 2008) and/or using hidden markov models of 'hidden states' to allow for variable timing (Daw, Courville & Touretzky, 2006). In the main motor chapter we consider a different alternative to TD, called PVLV (and the simulation exploration: [PVLV](/CCNBook/Sims/Motor/PVLV "wikilink")) which focuses much less on timing per se, and attempts to address some of the neural mechanisms upstream of the dopamine system that allow it to represent reward expectations.


