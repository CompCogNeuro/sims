Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

This simulation explores the neural basis of _priming_ -- the often surprisingly strong impact of residual traces from prior experience, which can be either _weight-based_ (small changes in synapses) or _activation-based_ (residual neural activity). In the first part, we see how small weight changes caused by the standard slow cortical learning rate can produce significant behavioral priming, causing the network to favor one output pattern over another. Likewise, residual activation can bias subsequent processing, but this is short-lived and transient compared to the long-lasting effects of weight-based priming.

Modeling these phenomena requires training a given input pattern to have two different possible outputs, like two different meanings of the same word (e.g., "bank" can mean "river bank" or "money bank") (i.e., a *one-to-many mapping*).  

Notice that the network has a standard three layer structure, with the `Input` presented to the bottom and `Output` produced at the top -- this is just to keep everything as simple and standard as possible.

* Click `OnlyA` and `OnlyB` in the control panel to see the two different sets of patterns that the network is trained on.

You should see that the `Input` patterns are identical for each set of patterns, but the `A` and `B` have different `Output` patterns paired with this same input. The `Alt AB` patterns are just the combination of both of these A and B sets, alternating the A and B output for each input, and it is what we use to do initial training of the network.

# Network Training

First, we simulate the learning of the semantic background knowledge of these two different associated outputs for each input.

* Press `Init` and `Step Trial` in the toolbar. After observing the standard minus / plus phase error-driven learning of the patterns, click on the `Train Epoch Plot` tab to view the plot of training progress, and do `Run`.

The graph shows two statistics of training: `PctErr` and `Correl`.  Because there are two possible correct outputs for any given input, cannot simply use a standard summed squared error (SSE) measure -- which target would you use when the network can validly produce either one?  Instead, we use a _closest pattern_ statistic to find which output pattern is the closest to the one that the network produced. The `PctErr` stat is based on whether the closest pattern is associated with the same input pattern (i.e., either the `A` or `B` version of the current input).  The `Correl` stat shows the correlation between the network's activity and the closest pattern (1 is a perfect match).  

You should see that the `PctErr` goes quickly down and bounces around near 0, while the `Correl` goes to 0.9 and above, indicating that the model learns to produce one of the two correct outputs for each input.

As something of an aside, it should be noted that the ability to learn this one-to-many mapping task depends critically on the presence of the kWTA inhibition in the network -- standard backpropagation networks will only learn to produce a _blend_ of both output patterns instead of learning to produce one output or the other (cf. [Movellan & McClelland, 1993](#references)). Inhibition helps by forcing the network to choose one output or the other, because both cannot be active at the same time under the inhibitory constraints.

Also, the bidirectional connectivity produces attractor dynamics that enable the network to exhibit a more nonlinear response to the input, by settling into one attractor or the other (you can reduce the strength of the top-down feedback connection, `.BackPath`, in Params to see that the Correl value is significantly reduced, showing that it isn't clearly representing either output, and is instead blending).  Finally, Hebbian learning also appears to be important here, because the network learns the task better with Hebbian learning than in a purely error driven manner. Hebbian learning can help to produce more distinctive representations of the two output cases by virtue of different correlations that exist in these two cases. [O'Reilly & Hoeffner (2000)](#references) provides a more systematic exploration of the contributions of these different mechanisms in this priming task. 

# Weight-Based Priming

**IMPORTANT** after this point, try to remember not to press `Init` in `Train` mode, which will initialize the weights. If you do, you can just press `Open Trained Wts` to open a set of pretrained weights.  Also, because the network won't run any more training epochs once it is trained, we do need to start out by doing `Init` followed by `Open Trained Wts`, just this once.

Having trained the network with the appropriate _semantic_ background knowledge, we are now ready to assess its performance on the priming task. 

First we need to establish a baseline measure of the extent to which the network at the end of training responds with either the `A` or `B` output.

* Select the `Test Trial Plot` to view the testing results, where the network is tested on the `OnlyA` patterns to determine how many times it responds with an A or B to the 12 input patterns. Then set the `Step` to `Epoch` and click `Step` to train one epoch with both `A` and `B` items as targets (i.e,. the `Alt AB` environment, that was used during training). It will automatically run the test after each epoch, so you will see the plot update.

This plot shows for each input (0-12) whether it responded with the A output (`IsA`) in the minus phase (i.e., `ActM`) and also gives the name of the closest output (`Closest`) for visual concreteness.  You should see that it responds with the `a` output on roughly half of the trials (there is no bias in the training -- it can randomly vary quite a bit though).

Next, we will change the training so that it only presents `B` output items, to see whether we can prime the `B` outputs by a single training trial on each of them, using the same slow learning rate that is used in all of our cortical simulations -- e.g., the `objrec` model which learned to recognize objects from visual inputs.

* Click the `Set Env` button in the toolbar, and select `TrainB` -- this will configure to train on the `B` items only. Then do `Step Epoch` again.

The extent of priming is indicated by the *reduction* in `IsA` responses -- the network should respond `b` more frequently, based on a single learning experience with a relatively small learning rate. If you do another `Step Epoch`, almost all of the responses should now be `b`.

To make sure this was not just a fluke, let's try to go the other way, and see how many of the current `b` responses can be flipped back over to `a` with a single exposure to the `a` outputs.

* Click `Set Env` and this time select the `TrainA`.  Then `Step Epoch`.

You should see that some of the `b` responses have now flipped back to `a`, again based on a single exposure.

You can repeat this experiment a couple more times, flipping the `a`'s back to `b`'s, and then back to `a`'s again.

* Click the `Test Epoch Plot` tab see a plot of all the testing results in summary form. 

> **Question 7.7:** Report the IsA results for each of the 3 data points, corresponding to TrainAltAB, TrainB, and TrainA (hover the mouse over the points to get the numbers, or click the `Table` button to see a table of the numbers).

You can optionally explore turning the `Lrate` parameter down to .01 or even lower. (We are applying the parameters every trial so you don't need to do `Init` to get the parameter to take effect.) You should see that although the number of items that flip is reduced, even relatively low `Lrate`s can produce flips.

# Activation-Based Priming

Next, we can see to what extent residual activation from one trial to the next can bias processing. We'll start over with the trained weights.

* Click `Init` while still in `Train` mode, and then do `Open Trained Wts`.

Next, we will use the `AltAB` patterns for testing, because they alternate between the `a` and `b` versions of each input when presented sequentially, so we can see the effect of the the `a` input on the first trial on the second trial of the same input pattern. We will test the extent to which the residual activation from the first `a` item can bias processing on the subsequent `b` case.  Note that we are recording the response of the network in the _minus_ phase, and then the specific `Output` is clamped in the plus phase (even during testing), so we can observe the effect of e.g., the `0_a` `Output` activation (with the `a` pattern) on the network's tendency to produce an `a` response again for the second 0 input. Therefore, we are looking specifically at the response on the second presentation of the same input in these alternating A, B patterns -- if there is activation priming, this second trial should be more likely to be an `a`.

* Click `Set Env` and select `Test alt AB` to use this full set of alternating `AltAB` patterns during _testing_, and then switch to `Test` instead of `Train` mode, and do `Init` (which will not initialize the weights because we are in `Test` mode), `Run` to see the baseline level of responding, while looking at the `Test Trial Plot`.

This is a baseline, because we are still clearing all of the activation out of the network between each input, due to the `Decay` parameter being set to the default of 1. You should see that the network responds _consistently_ to both instances of the same input pattern. For example, if it responds `a` to the first `0` input, then it also responds `a` to the second input right after that. Similarly, if the network responds `b` to the first trial of an input pattern, then it also responds `b' to the second trial of the input pattern. There is no biasing toward `a` after the first trial, and no evidence of activation priming here.

* Set `Decay` to 0 instead of 1, and do another `Init` and `Run`. You should now observe a very different pattern, where the responses to the second trial of an input pattern are more likely to be `a` than the first trial of the same input pattern. This looks like a "sawtooth" kind of jaggy pattern in the test plot.

> **Question 7.8:** Comparing the 1st trials and 2nd trials of each input pattern (the 1st and 2nd 0, the 1st and 2nd 1, and so on), report the number of times the network responded 'b' to the first trial and 'a' to the second trial. How does this number of instances of activation-based priming compare to the 0 instances observed at baseline with Decay set to 1?.

You can explore the extent of residual activity needed to show this activation-based priming by adjusting the `Decay` parameter and running `Test` again. (Because no learning takes place during testing, you can explore at will, and go back and verify that Decay = 1 still produces mostly `b`'s).  In our tests increasing Decay (using this efficient search sequence: 0, .5, .8, .9, .95, .98, .99), we found a critical transition between .98 and .99. That is, a tiny amount of residual activation with Decay = .98 (= .02 residual activity) was capable of driving some activation-based priming. This suggests that the network is delicately balanced between the two attractor states, and even a tiny bias can push it one way or the other. The similar susceptibility of the human brain to such activation-based priming effects suggests that it too may exhibit a similar attractor balancing act.

# References

* Movellan, J. R., & McClelland, J. L. (1993). Learning Continuous Probability Distributions with Symmetric Diffusion Networks. Cognitive Science, 17, 463–496.

* O’Reilly, R. C., & Hoeffner, J. H. (2000). Competition, priming, and the past tense U-shaped developmental curve.

