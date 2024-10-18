Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

This simulation explores the temporal differences (TD) reinforcement learning algorithm under some basic Pavlovian conditioning environments.

To explore the TD learning rule, we use a simple classical (Pavlovian) conditioning task, where the network learns that a stimulus (tone) reliably predicts a subsequent reward. Then, we extend the task to a _second order conditioning_ case, where another stimulus (light) reliably predicts that tone. First, we need to justify the use of the TD algorithm in this context, and motivate the nature of the stimulus representations used in the network.

You might recall that we said that the delta rule (aka the Rescorla-Wagner rule) provides a good model of classical conditioning, and thus wonder why TD is needed. It all has to do with the issue of *timing*. If one ignores the timing of the stimulus relative to the response, then in fact the TD rule becomes equivalent to the delta rule when everything happens at one time step (it just trains V(t) to match r(t)). However, animals are sensitive to the timing relationship, and, more importantly for our purposes, modeling this timing provides a particularly clear and simple demonstration of the basic properties of TD learning.

The only problem is that this simple demonstration involves a somewhat unrealistic representation of timing. Basically, the stimulus representation has a distinct unit for each stimulus for each point in time, so that there is something unique for the TD system to learn from. This representation is the _complete serial compound_ (CSC) proposed by [Sutton & Barto (1990)](#references), and we will see exactly how it works when we look at the model. As we have noted, we explore a more plausible alternative in the Executive Function chapter, where the TD error signal controls the updating of a context representation that maintains the stimulus over time.

# Mapping TD to the Network

There are four separate TD layers that each compute a part of the overall TD algorithm, and the actual TD computations are performed by the simulation code, instead of using network-level interactions among the layers themselves. Although it is possible to get units to compute the necessary additions and subtractions required by the TD algorithm, it is much simpler and more robust to perform these calculations directly using the values represented by the units. The critical network-level computation is learning about the reward value of stimuli, and this is done using a special dopamine-like learning rule in the `Pred` layer.

Here is a description of what each layer does:

*  `Rew`: Represents the reward input, i.e., the US (unconditioned stimulus), as an activation state (0 = no reward, 1 = reward).

*  `Pred`: The key learning layer, which learns to predict the reward value on the next time step `V(t+1)` based on the current stimulus inputs. This prediction is generated in the *plus phase* of Leabra settling based on its current weights from the `Input` layer (a linear unbounded activation rule is used, so this unit can represent arbitrary values), whereas in the minus phase the layer's state is clamped to the prediction made on the previous trial: `V(t)`.

*  `Integ`: Integrates the reward prediction and external reward layer values, and the difference in its plus-minus phase activation states are what drive the TD delta (dopamine-like) signal. Specifically, its minus phase activation is `V(t)` -- the expectation of reward computed by the rew pred layer *on the previous trial*, and its plus phase activation is the expectation of reward *on the next trial* (from Pred) plus any actual rewards being received at the present time (via a direct synaptic input from Rew layer). Thus, its plus phase state is the sum of the `Rew` and `Pred` values, and this sum is directly clamped as an activation state on the unit.

* `TD`: Computes the plus -- minus values from the Integ layer, which reflects the TD delta value and is thought to act like the dopamine signal in the brain. This TD value sets the DA value in the layer, which is broadcast to all other layers, including critically the Pred layer, where it drives learning.

# The Network

Let's start by examining the network. The `Input` layer (located at the top, to capture the relative anatomical locations of this cortical area relative to the midbrain dopamine system represented by the TD layers below it) contains three rows of 20 units each. This is the CSC, where the rows each represent a different stimulus (A, B, C), and the columns represent points in time: each unit has a stimulus and a time encoding (e.g., A_10 = time step 10 of the A stimulus). The TD layers are as described above.

* Click on `r.Wt` in the `Network` tab and then on the `Pred` layer unit -- you will see that it starts with a uniform weight of 0. Then, click back to viewing `Act`.

# The Basic TD Learning Mechanism

Let's see how the CSC works in action.

* Click `Init` and then the `Step Trial` button on the toolbar to step the input along.

Nothing should happen in the `Input` layer, because no stimulus is present at time step 0. The various TD layers will remain at 0 as well, and the `TD` layer also has a zero activation. Thus, no reward was either expected or obtained, and there is no deviation from expectation. 

* Continue to press the `Step Trial` button until you see an activation in the Input layer (should be 10 more steps).

This input activation represents the fact that the conditioned stimulus (CS) `A` (i.e., the "tone" stimulus) came on at t=10. There should be no effect of this on the TD layers, because they have not associated this CS with reward yet.

* Continue to `Step Trial` some more.

You will see that this stimulus remains active for 6 more time steps (through t=15), and at the end of this time period, the `Rew` layer represents a value of 1 instead of 0, indicating that an external reward was delivered to the network. Because the `Pred` layer has not learned to expect this reward, the TD delta value is positive, as reflected by the activity of the TD unit. This TD dopamine spike also drove learning in the `Pred` layer, as we'll see the next time we go through this sequence of trials.

* Click on the `Train Trial Plot` tab, and then do `Step Trial` to finish out the trial of 20 time steps.  This will show you the TD dopamine spike at t=15.

* Click back on the `Network` and do `Step Trial` through the next trial until time step 14.

You should see that the `Pred` layer now gets weakly activated at time step 14, signaling a prediction of reward that will come on the next time step. This expectation of reward, even in the absence of a delivered reward on the `Rew` layer (which still shows a 0 value representation), is sufficient to drive the TD "dopamine spike" -- click to `Train Trial Plot` to see it on the plot.

* Do Step: Trial one more time (t=15).

Now the situation is reversed: the `Rew` layer shows that the reward has been presented, but the TD value is reduced from the previous 1. This is because the `Pred` layer accurately predicted this reward on the prior time step, and thus the reward prediction error, which TD signals, has been reduced by the amount of the prediction! 

* Do `Step Trial` to process the rest of the trial, and switch to viewing `Train Trial Plot`.

The plot shows that the "dopamine spike" of TD delta has moved forward one step in time. This is the critical feature of the TD algorithm: by learning to anticipate rewards one time step later, it ends up moving the dopamine spike earlier in time.

* Keep doing more `Step Trial` (or just `Train`).

You should see that the spike moves "forward" in time with each `Step Trial`, but can't move any further than the onset of the CS at time step 10.

We can also examine the weights to see what the network has learned.

* Click on `r.Wt` and then on the `Pred` layer unit -- you should see that there are increased weights from the A stimulus for time steps 10-14.  You can also click the `Weights` tab to see the weights in a grid view.

# Extinction and Second Order Conditioning

At this point, there are many standard phenomena in classical conditioning that can be explored with the model. We will look at two: *extinction* and *second order conditioning*. Extinction occurs when the stimulus is no longer predictive of reward -- it then loses its ability to predict this reward (which is appropriate). Second order conditioning, as we discussed earlier, is where a conditioned stimulus can serve as the unconditioned stimulus for another stimulus -- in other words, one can extend the prediction of reward backward across two separate stimuli.

We can simulate extinction by simply turning off the US reward that appears at t=15. 

* Click on the `Envs` and `Train`, and then click off the `Act` toggle in the `US` field -- this will de-activate the US.

* Now, hit `Init` (which does not initialize the weights), and `Reset Trial Log` in toolbar to clear out the `Train Trial Plot` (click on that to view it), and then `Step Trial`.  It can also be useful to click on the `Pred` line in the plot, which shows the predicted reward value.  You can also `Step Trial` and watch the Network to see how the network behaves.

> **Question 8.5:** What happened at the point where the reward was supposed to occur? Explain why this happened using the TD mechanisms of reward expectations compared to actual reward values received.

> **Question 8.6:** Run the network and describe what occurs over the next several epochs of extinction training in terms of the TD error signals plotted in the graph view, and explain why TD does this. After the network is done learning again, does the stimulus still evoke an expectation of reward?

Now, let's explore second order conditioning. We must first retrain the network on the stimulus 1 association.

* In the `Train` env window, turn `US` `Act` back on, then do `Init` and `Train`. You can `Stop` when it is trained.

Now, we will turn on the CS B stimulus, which starts at t=2 and lasts until time step 10.

* Click the `CSB` `Act` on in `Train`, and go back to viewing `Act` in the `Network` if you aren't already.  Hit `Reset Trial Log` to clear the plot, then do `Step Trial` to see the B stimulus followed by the A, then the US (you might need to go through twice to get a full trial, depending on where it stopped last time).

Essentially, the CSA stimulus *acts just like a reward* by triggering a positive delta value, and thus allows the CSB stimulus to learn to predict this first stimulus.

* Push `Train`, and switch to `Train Trial Plot`, then `Stop` when the plot stops changing.

You will see that the early anticipation of reward gets carried out to the onset of the CS B stimulus (which comes first in time).

Finally, we present some of the limitations of the CSC representation. One obvious problem is capacity -- each stimulus requires a different set of units for all possible time intervals that can be represented. Also, the CSC begs the question of how time is initialized to zero at the right point so every trial is properly synchronized. Finally, the CSC requires that the stimulus stay on (or some trace of it) up to the point of reward, which is unrealistic. This last problem points to an important issue with the TD algorithm, which is that although it can learn to bridge temporal gaps, it requires some suitable representation to support this bridging (which we explore in the Executive Function Chapter).

# Advanced Explorations

More advanced explorations can be performed by experimenting with different settings of the `Train` environment. Here you can manipulate the probabilities of stimuli being presented, and introduce randomness in the timings. Generally speaking, these manipulations tend to highlight the limitations of the CSC input representation, and of TD more generally, but many of these are addressed by more advanced approaches (e.g., representing the sensory state in more realistic ways; Ludvig, Sutton & Kehoe 2008) and/or using hidden markov models of 'hidden states' to allow for variable timing (Daw, Courville & Touretzky, 2006). In the main motor chapter we consider a different alternative to TD, called PVLV (and the simulation exploration: PVLV) which focuses much less on timing per se, and attempts to address some of the neural mechanisms upstream of the dopamine system that allow it to represent reward expectations.

# References

* Daw, N. D., Courville, A. C., & Touretzky, D. S. (2006). Representation and timing in theories of the dopamine system. Neural Computation, 18(7), 1637–1677. https://doi.org/10.1162/neco.2006.18.7.1637

* Ludvig, E. A., Sutton, R. S., & Kehoe, E. J. (2008). Stimulus representation and the timing of reward-prediction errors in models of the dopamine system. Neural Computation, 20(12), 3034–3054.

* Sutton, R. S., & Barto, A. G. (1990). Time-Derivative Models of Pavlovian Reinforcement. In J. W. Moore & M. Gabriel (Eds.), Learning and Computational Neuroscience (pp. 497–537). Cambridge, MA: MIT Press.


