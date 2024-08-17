Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

This simulation illustrates how error-driven and hebbian learning can operate within a simple task-driven learning context, with no hidden layers. The situation is reduced to the simplest case, where a set of 4 input units project to 2 output units. The "task" is specified in terms of the relationships between patterns of activation over the input units, and the corresponding desired or **target** values of the output units. This type of network is often called a **pattern associator** because the objective is to associate patterns of activity on the input with those on the output.

# The Easy Task

You should see the network has 2 output units receiving inputs from 4 input units through a set of feedforward weights.

* Click the `Easy` button to view the first set of patterns the network will learn to associate.

As you can see, the input-output relationships to be learned in this "task" are simply that the leftmost two input units should make the left output unit active, while the rightmost units should make the right output unit active. This can be thought of as categorizing the first two inputs as "left" with the left output unit, and the next two as "right" with the right output unit.

This is a relatively easy task to learn because the left output unit just has to develop strong weights to the leftmost input units and ignore the ones to the right, while the right output unit does the opposite. Note that we are using FFFB inhibition, which tends to result in one active output unit (though not strictly).

The network is trained on this task by simply clamping both the input and output units to their corresponding values from the events in the environment, and performing pure BCM learning.

* Press `Init` and then `Step Trial` 4 times while you watch the network. 

You should see all 4 events from the environment presented in a random order.

* Now do `Test Trial` 4 times to test the network's responses to each input.

You will see the activations in the output units are different this time, because the output activations are not clamped to the correct answer, and are instead updated solely according to their current weights from the input units.  Thus, testing records the current *actual* performance of the network on this task, when it is not being "coached" (that is why it's a test).  This is equivalent to the *minus phase activations* during training, which can be viewed using the `ActM` variable in the NetView.

* Now click on the `TstTrlLog` button to see a record of all the testing trials.

The results of the test run you just ran are displayed. Each row represents one of the four events, with the input pattern and the actual output activations shown on the right. The `SSE` column reports the **summed squared error** (SSE), which is simply the summed difference between the actual output activation during testing (`o`) and the *target* value (`t`) that was clamped during training:

```
   SSE = Sum (t - o)^2
```

where the sum is over the 2 output units. We are actually computing the *thresholded* SSE, where absolute differences of less than 0.5 are treated as zero, so the unit just has to get the activation on the correct side of 0.5 to get zero error. We thus treat the units as representing underlying binary quantities (i.e., whether the pattern that the unit detects is present or not), with the graded activation value expressing something like the likelihood of the underlying binary hypothesis being true. All of our tasks specify binary input/output patterns.

With only a single training epoch (one *epoch* is one pass through all the training patterns), the output unit is likely making some errors.

* Click on the `TrnEpcPlot` tab to see a plot of SSE (summed also over all 4 training patterns) as the network trains.  Hit `Step Run` to see the network iterate over training patterns.  

Now you will see a summary plot across epochs of the sum of the thresholded SSE measure across all the events in the epoch. This shows what is often referred to as the **learning curve** for the network, and it should have decreased steadily down to zero, indicating that the network has learned the task.  Training will stop automatically after the network has exhibited 5 correct epochs in a row (`NZeroStop`) just to make sure it has really learned the problem), or it stops after 40 epochs (`MaxEpcs`) if it fails to learn.

Let's see what the network has learned.

* Click back on the `NetView` and then do `Test Trial` to see each of the different input patterns and the network's response.

You should see that it is producing the correct output units for each input pattern. You can also look at the `TstTrlLog` to see the behavior of the network across all four trials, all at once. You should see that the network has learned this easy task, turning on the left output for the first two patterns, and the right one for the next two. Now, let's take a look at the weights for the output unit to see exactly how this happened.

* In the `NetView`, click on `r.Wt` and then select the left `Output` unit to see its weights from the `Input`, then do the same for the right.

> **Question 4.3:** Describe the pattern of weights in qualitative terms for each of the two output units (e.g., left output has strong weights from the ?? input units, and weaker weights from the ?? input units).

> **Question 4.4:** Why would a Hebbian-style learning mechanism, which increases weights for units that are active together at the same time, produce the pattern of weights you just observed?  This should be simple qualitative answer, referring to the specific patterns of activity in the input and output of the Easy patterns.

# The Hard Task

Now, let's try a more difficult task.

* Set `Pats` to `Hard` instead of `Easy`.  Click the `Hard` button below that to pull up a view of the Hard patterns.

In this harder environment, there is overlap among the input patterns for cases where the left output should be on, and where it should be off (and the right output on). This overlap makes the task hard because the unit has to somehow figure out what the most distinguishing or *task relevant* input units are, and set its weights accordingly.

This task reveals a problem with Hebbian learning: it is only driven by the correlation between the output and input units, so it cannot learn to be sensitive to which inputs are more task relevant than others (unless this happens to be the same as the input-output correlations, as in the easy task). This hard task has a complicated pattern of overlap among the different input patterns. For the two cases where the left output should be on, the middle two input units are very strongly correlated with the output activity, while the outside two inputs are half-correlated. The two cases where the left output should be off (and the right one on) overlap considerably with those where it should be on, with the last event containing both of the highly correlated inputs. Thus, if the network just pays attention to correlations, it will tend to respond incorrectly to this last case.

Let's see what happens when we run the network on this task.

* After making sure you are still viewing the `r.Wt` receiving weights of the left output unit in the `NetView`, press the `Init` and `Step Run` buttons. After training (or even during), click back and forth between the left and right output units. Try multiple Runs to see what generally tends to happen. 

You should see that the weights into the left output unit increase, often with the two middle ones being more strongly increasing due to the higher correlation. The right output tends to have a strong weight from the 2nd input unit, and then somewhat weaker weights to the right two inputs, again reflecting the input correlations. Note that in contrast to a purely Hebbian learning mechanism, the BCM learning does not strictly follow the input correlations, as it depends significantly on the output unit activations over time as well, which determine the floating threshold for weight increase vs. decrease.

* Return to viewing the `Act` variable do `Test Trial` to see the network's response to the inputs.

You should see that the network is not getting all the right answers (you can also look at the `TstTrlLog` to see all events at once.)  This is also evident in the training SSE shown in the network view.

* Do several more `Step Run`s on this Hard task. You can try increasing the `MaxEpcs` to 50, or even 100, to give it more time to learn.

> **Question 4.5:** Does the network ever solve the task? Run the network several times. Report the final SSE at the end of training for each run (it is visible as `EpcSSE` in the left panel at the end of every `Step Run`).

Hebbian learning does not seem to be able to solve tasks where the correlations do not provide the appropriate weight values. In the broad space of tasks that people learn (e.g., naming objects, reading words, etc) it seems unlikely that there will always be a coincidence between correlational structure and the task solution. Thus, we must conclude that Hebbian learning by itself is of limited use for task learning. In contrast, we will see in the next section that error-driven learning, which specifically adapts the weights precisely to solve input/output mappings, can handle this Hard task without much difficulty.

# Exploration of Error-Driven Task Learning

* Select `ErrorDriven` instead of `Hebbian` for the `Learn` value in the left control panel, go back to `Easy` for the `Pats`, and then press `Init` to have it take effect.  Also set `MaxEpcs` to 100 -- error-driven learning can sometimes be a bit slow.

This will switch weight updating from the purely Hebbian (BCM) form of learning, to the form that is purely error driven, in terms of the contrast between plus (short term average) and minus (medium term) phases of activation. In this simple two-layer network, this form of learning is effectively equivalent to the Delta rule error-driven learning algorithm.   This sets the `Learn` params on the connections between Input and Output to have 0 amount of learning driven by the long-term running average activation (which corresponds to BCM Hebbian learning) and 100% of the learning driven by the medium-term floating threshold (which corresponds to error-driven learning).

Before training the network, we will explore how the minus-plus activation phases work in the simulator.

* Make sure that you are viewing activations in the network by selecting the `Act` button in the `NetView`, and do `Step Trial` to present a Hard training pattern.  

The activity will flicker over 4 **quarters** of time, where each quarter represents 25 msec, and the first 75 msec (3 quarters) of a 100 msec trial period constitutes the *expectation* or *minus phase*, followed by the final 25 msec which is the *outcome* or *plus phase*.

* To see the activation at each of these time points, you can use the VCR-like buttons at the bottom-right of the NetView, by the `Time` label -- step back and you'll see the cycle counter going back through each quarter increment.  Step forward to see it unfold in the proper order.  You can also click on the `ActQ1`, `ActQ2`, `ActM` and `ActP` variables to see the activity at the end of each quarter as well.

Learning occurs after the plus phase of activation.  You can recognize targets, like all external inputs, because their activations are exactly .95 or 0 -- note that we are clamping activations to .95 (not 1.0) because units cannot easily produce activations above .95 with typical net input values due to the saturating nonlinearity of the rate code activation function. You can also switch to viewing the `Targ` value, which will show you the target inputs prior to the activation clamping. In addition, the minus phase activation is always viewable as `ActM` and the plus phase as `ActP`.

The critical difference between error-driven learning and Hebbian is that error-driven learning is based directly on this difference between the expectation or guess produced in the minus phase, and the correct target activation in the plus phase.

* If your network did not make the wrong guess during the minus phase, keep doing `Step Trial` until it does, then click on the `r.DWt` variable to see the delta-weights (weight changes, i.e., learning) that occurred for incorrectly activated unit, versus the correctly activated one.  You should see that the weights go down for the erroneously activated unit, and up for the one that *should* have been activated -- this is the essence of error correction and occurs because learning is proportional to the change in activity over time for each output unit.

* Go ahead and `Step Run` the network to complete the training on the Easy task. 

The network should have no trouble learning this task, as you can see in the `TrnEpcPlot`. You can do more `Step Run`s to see how reliably and rapidly it learns this problem. Compared to Hebbian, it learns this Easy task more slowly.

But the real challenge is whether it can learn the Hard task, which Hebbian could not learn at all.

* Set `Learn` to `Hard`, press `Init` and `Step Run`. Do multiple repeated Runs, to see how reliably and quickly it learns overall (monitor `TrnEpcPlot` to make it run faster).

You should see that the network learns this task without much difficulty, because error-driven learning is directly a function of how well the network is actually doing, driving the weights specifically to solve the task, instead of doing something else like encoding correlational structure. Now we'll push the limits of even this powerful error-driven learning.

* Set `Learn` to `Impossible`, and click on the `Impossible` button to view the patterns.

Notice that each input unit in this environment is active equally often when the output is active as when it is inactive. That is, there is complete overlap among the patterns that activate the different output units. These kinds of problems are called *ambiguous cue* problems, or *nonlinear discrimination* problems (e.g., Sutherland & Rudy, 1989).  This kind of problem might prove difficult, because every input unit will end up being equivocal about what the output should do. Nevertheless, the input patterns are not all the same -- people could learn to solve this task fairly trivially by just paying attention to the overall patterns of activation. Let's see if the network can do this.

* Press `Init` and `Step Run`. Do it again, and again.. Increase the `MaxEpcs` higher than 100. 

> **Question 4.6:** Does the network ever learn to solve this "Impossible" problem? Report the final SSE values for your runs.

Because error-driven learning cannot learn what appears to be a relatively simple task, we conclude that something is missing.  Unfortunately, that is not the conclusion that Minsky & Papert reached in their highly influential book, *Perceptrons*. Instead, they concluded that neural networks were hopelessly inadequate because they could not solve problems like the one we just explored. This conclusion played a large role in the waning of the early interest in neural network models of the 1960s. As we'll see, all that was required was the addition of a hidden layer interposed between the input and output layers (and the necessary math to make learning work with this hidden layer, which is really just an extension of the chain rule used to derive the delta rule for two layers in the first place).


