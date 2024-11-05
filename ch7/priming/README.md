Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

This simulation explores the neural basis of _priming_ -- the often surprisingly strong impact of residual traces from prior experience, which can be either _weight-based_ (small changes in synapses) or _activation-based_ (residual neural activity). In the first part, we see how small weight changes caused by the standard slow cortical learning rate can produce significant behavioral priming, causing the network to favor one output pattern over another. Likewise, residual activation can bias subsequent processing, but this is short-lived and transient compared to the long-lasting effects of weight-based priming.

Modeling these phenomena requires training a given input pattern to have two different possible outputs, like two different meanings of the same word (e.g., "bank" can mean "river bank" or "money bank") (i.e., a *one-to-many mapping*).  

Notice that the network has a standard three layer structure, with the `Input` presented to the bottom and `Output` produced at the top -- this is just to keep everything as simple and standard as possible.

* Click `TrainA` and `TrainB` in the control panel to see the two different sets of patterns that the network is trained on.

You should see that the `Input` patterns are identical for each set of patterns, but the `A` and `B` have different `Output` patterns paired with this same input. The `TrainAll` patterns are just the combination of both of these A and B sets, and it is what we use to do initial training of the network.

# Network Training

First, we simulate the learning of the semantic background knowledge of these two different associated outputs for each input.

* Press `Init` and `Step Trial` in the toolbar. After observing the standard minus / plus phase error-driven learning of the patterns, click on the `Train Epoch Plot` tab to view the plot of training progress, and do `Run`.

The graph shows two statistics of training: `PctErr` and `Correl`.  Because there are two possible correct outputs for any given input, cannot simply use a standard summed squared error (SSE) measure -- which target would you use when the network can validly produce either one?  Instead, we use a _closest pattern_ statistic to find which output pattern is the closest to the one that the network produced. The `PctErr` stat is based on whether the closest pattern is associated with the same input pattern (i.e., either the `A` or `B` version of the current input).  The `Correl` stat shows the correlation between the network's activity and the closest pattern (1 is a perfect match).  

You should see that the `PctErr` goes quickly down and bounces around near 0, while the `Correl` goes to 0.9 and above, indicating that the model learns to produce one of the two correct outputs for each input.

As something of an aside, it should be noted that the ability to learn this one-to-many mapping task depends critically on the presence of the kWTA inhibition in the network -- standard backpropagation networks will only learn to produce a _blend_ of both output patterns instead of learning to produce one output or the other (cf. [Movellan & McClelland, 1993](#references)). Inhibition helps by forcing the network to choose one output or the other, because both cannot be active at the same time under the inhibitory constraints.

Also, the bidirectional connectivity produces attractor dynamics that enable the network to exhibit a more nonlinear response to the input, by settling into one attractor or the other (you can reduce the strength of the top-down feedback connection, `.BackPath`, in Params to see that the Correl value is significantly reduced, showing that it isn't clearly representing either output, and is instead blending).  Finally, Hebbian learning also appears to be important here, because the network learns the task better with Hebbian learning than in a purely error driven manner. Hebbian learning can help to produce more distinctive representations of the two output cases by virtue of different correlations that exist in these two cases. [O'Reilly & Hoeffner (2000)](#references) provides a more systematic exploration of the contributions of these different mechanisms in this priming task. 

# Weight-Based Priming

**IMPORTANT** after this point, try to remember not to press `Init`, which will initialize the weights. If you do, you can just press `Open Trained Wts` to open a set of pretrained weights.  Also, because the network won't run any more training epochs once it is trained, we do need to start out by doing `Init` followed by `Open Trained Wts`, just this once.

Having trained the network with the appropriate _semantic_ background knowledge, we are now ready to assess its performance on the priming task. 

We will first see if we can prime the `B` outputs by a single training trial on each of them, using the same slow learning rate that is used in all of our cortical simulations -- e.g., the `objrec` model which learned to recognize objects from visual inputs.

First we need to establish a baseline measure of the extent to which the network at the end of training responds with either the `A` or `B` output.

* Select the `Test Trial Plot` to view the testing results, and then set the `Step` to `Epoch` and click `Step` to train one epoch with both `A` and `B` items as targets (i.e,. the `TrainAll` environment, that was used originally). It will automatically run the test after each epoch, so you will see the plot update.

This plot shows for each input (0-12) whether it responded with the A output (`IsA`) in the minus phase (i.e., `ActM`) and also gives the name of the closest output (`Closest`) for visual concreteness.  You should see that it responds with the `a` output on roughly half of the trials (there is no bias in the training -- it can randomly vary quite a bit though).

Next, we will change the training so that it only presents `B` output items.

* Click the `Set Env` button in the toolbar, and select `TrainB` -- this will configure to train on the `B` items only. Then do `Step Epoch` again.

The extent of priming is indicated by the *reduction* in `IsA` responses -- the network should respond `b` more frequently (even exclusively), based on a single learning experience with a relatively small learning rate.

To make sure this was not just a fluke, let's try to go the other way, and see how many of the current `b` responses can be flipped back over to `a` with a single exposure to the `a` outputs.

* Click `Set Env` and this time select the `TrainA`.  Then `Step Epoch`.

You should see a large number of the `b` responses have now flipped back to `a`, again based on a single exposure.

You can repeat this experiment a couple more times, flipping the `a`'s back to `b`'s, and then back to `a`'s again.

* Click the `Test Epoch Log` tab see a plot of all the testing results in summary form. 

> **Question 7.7:** Report the IsA results for each of the 3 data points, corresponding to TrainAll, TrainB, and TrainA (hover the mouse over the points to get the numbers, or click the `Table` button to see a table of the numbers).

You can optionally explore turning the `Lrate` parameter down to .01 or even lower (we are applying the parameters every trial so you don't need to do `Init` to get the parameter to take effect).  You should see that although the number of items that flip is reduced, even relatively low lrates can produce flips.

# Activation-Based Priming

Next, we can see to what extent residual activation from one trial to the next can bias processing.  To set up this test, we want the network to have a weight-based bias to respond with the `b` output:

* Click `Set Env` and select `TrainB`, and do `Step Epoch`, to ensure that it generally responds `b` (if you experimented with learning rate, or did a lot of tests with the weight-based case, it might be better to start fresh with `Init`, `Open Trained Wts` and then the `TrainB` etc).

Next, we will use the `TrainAll` patterns for testing, because they alternate between the `a` and `b` versions of each input when presented sequentially -- we will test for the extent to which the residual activation from the `a` item can bias processing on the subsequent `b` case.  Note that we are recording the response of the network in the *minus* phase, and then the specific `Output` is clamped in the plus phase (even during testing), so we can observe the effects of e.g., the `0_a` `Output` activation (with the `a` pattern) on the tendency to bias the network to produce an `a` response again for the 0 input, despite the weights being biased in favor of producing the `b` output.

* Click `Set Env` and select `TestAll` to use this full set of alternating patterns during _testing_, and then switch to `Test` instead of `Train` mode, and do `Init`, `Run` to see the baseline level of responding, while looking at the `Test Trial Plot`.

This is a baseline, because we are still clearing all of the activation out of the network between each input, due to the `Decay` parameter being set to the default of 1.

* Set `Decay` to 0 instead of 1, and do another `Init` and `Run`.  You should now observe several trials in which the `a` pattern is activated, for the 2nd of the two repeated inputs.

> **Question 7.8:** Report the number of times the network responded 'a' instead of 'b' for the 'b' test trials, relative to the baseline that you observed above with Decay set to 1.

You can explore extent of residual activity needed to show this activation-based priming by adjusting the `Decay` parameter and running `Test` again (no learning takes place during testing so you can explore at will, and go back and verify that Decay = 1 still produces mostly `b`'s).  In our tests increasing Decay using this efficient search sequence: 0, .5, .8, .9, .95, .98, .99, we found a critical transition between .98 and .99 -- i.e., a very tiny amount of residual activation with .98 (= .02 residual activity) was capable of driving a surprisingly large amount of activation-based priming.  This suggests that the network is delicately balanced between the two attractor states and even a very tiny bias can push it one way or the other. The similar susceptibility of the human brain to such activation-based priming effects suggests that it too may exhibit a similar attractor balancing act.

# References

* Movellan, J. R., & McClelland, J. L. (1993). Learning Continuous Probability Distributions with Symmetric Diffusion Networks. Cognitive Science, 17, 463–496.

* O’Reilly, R. C., & Hoeffner, J. H. (2000). Competition, priming, and the past tense U-shaped developmental curve.

