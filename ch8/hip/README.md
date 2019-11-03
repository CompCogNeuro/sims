Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

In this exploration of the hippocampus model, we will use the same basic AB--AC paired associates list learning paradigm as we used in the standard cortical network previously (`abac`). The hippocampus should be able to learn the new paired associates (AC) without causing undue levels of interference to the original AB associations (see Figure 1), and it should be able to do this much more rapidly than was possible in the cortical model. This model is using the newer *Theta Phase* model of the hippocampus ([Ketz, Morkanda & O'Reilly, 2013](#references)), where the EC <-> CA1 projections along with all the other connections have an error-driven learning component organized according to the theta phase rhythm.  See [leabra hip](https://github.com/emer/leabra/tree/master/hip) on github for more implementational details.

![AB-AC Data](fig_ab_ac_data_catinf.png?raw=true "AB-AC Data")

**Figure 1:** Data from people learning AB-AC paired associates, and comparable data from McCloskey & Cohen (1989) showing *catastrophic interference* of learning AC on AB.

* Click on `TrainAB` and `TestAB` buttons to see how the AB training and testing lists are configured -- the A pattern is the first three groups of units (at the bottom of each pattern, going left-right, bottom-top), and the B pattern is the next three, which you can see most easily in the Test_AB patterns where these are blank (to be filled in by hippocampal pattern completion). The 2nd half of the pattern is the list context (as in the `abac` project).

# AB Training and Testing

Let's observe the process of activation spreading through the network during training.

* Set `TrainUpdt` to `Cycle` instead of `AlphaCycle`, and do `Init`, `Step Trial`.

You will see an input pattern from the AB training set presented to the network. As expected, during training, all three parts of the input pattern are presented (A, B, Context). You will see that activation flows from the `ECin` layer through the `DG, CA3` pathway and simultaneously to the `CA1`, so that the sparse `CA3` representation can be associated with the invertible `CA1` representation, which will give back this very `ECin` pattern if later recalled by the `CA3`.  You can use the Time VCR buttons in the lower right of the NetView to replay the settling process cycle-by-cycle.

* `Step Trial` through several more (but fewer than 10) training events, and observe the relative amount of pattern overlap between subsequent events on the `ECin, DG, CA3`, and `CA1` layers, by clicking back-and-forth between `ActQ0` and `Act`.  You can set `TrainUpdt` back to `AlphaCycle`.

You should have observed that the `ECin` patterns overlap the most, with `CA1` overlapping the next most, then `CA3`, and finally `DG` overlaps the least. The levels of FFFB overall inhibition parallel this result, with DG having a very high level of inhibition, followed by CA3, then CA1, and finally EC.

> **Question 8.4:** Using the explanation given earlier in the text about the pattern separation mechanism, and the relative levels of activity on these different layers, explain the overlap results for each layer in terms of these activity levels, in qualitative terms.

Each epoch of training consists of the 10 list items, followed by testing on 3 sets of testing events. The first testing set contains the AB list items, the second contains the AC list items, and the third contains a set of novel *Lure* items to make sure the network is treating novel items as such. The network automatically switches over to testing after each pass through the 10 training events.

* Do `Step Epoch` to step through the rest of the training epoch and then automatically into the testing of the patterns.  Press `Stop` after a couple of testing items, so you can use Time VCR buttons to rewind into the settling process during testing, to see how it all unfolds.

You should observe that during testing, the input pattern presented to the network is missing the second associate as we saw earlier (the B or C item in the pair), and that as the activation proceeds through the network, it fills in this missing part in the EC layers (pattern completion) as a result of activation flowing up through the `CA3`, and back via the `CA1` to the `ECout`.

* Click on `TstTrlPlot` tab, and let's start over by doing `Init` then `Step Epoch`.

You should see a plot of three values for each test item, ordered AB, AC, then Lure (you can click on `TrialName` to see the labels). The `TrgOnWasOff` shows how many units in `ECout` were off but should have been on, while `TrgOffWasOn` shows the opposite.  When both of these measures are relatively low (below .34 as set in `MemThr` in control panel), then the network has correctly recalled the original pattern, which is scored as a `Mem` = 1. A large `TrgOffWasOn` indicates that the network has *confabulated* or otherwise recalled a different pattern than the cued one. A large `TrgOnWasOff` indicates that the network has failed to recall much of the probe pattern.  The threshold on these factors assumes a distributed representation of associate items, such that the entire pattern need not be recalled.

In general, you should see `TrgOnWasOff` being larger than `TrgOffWasOn` -- the hippocampal network is "high threshold", which accords with extensive data on recollection and recall (see [Norman & O'Reilly, 2003](#references) for more discussion). 

* Do two more `Step Epoch` to do more learning on the AB items.  Press `TstStats` `etable.Table` in the control panel to pull up the exact numbers shown in the plot, summarized for each test case.

> **Question 8.5:** Report the total proportion of `Mem` responses from your `TstStats` for the AB, AC, and Lure tests.


# Detailed Testing: Pattern Completion in Action

Now that the network has learned something, we will go through the testing process in detail by stepping one cycle at a time.

* Click back on the `NetView`, then do `Test All` and then hit `Stop` so you can review the activation cycle-by-cycle through the `Time` VCR buttons, for an AB pattern.

You should see the studied A stimulus, an empty gap where the B stimulus would be, and a list context representation for the AB list in the `Input` and `ECin`. Since this was studied, it is likely that the network will be able to complete the B pattern, which you should be able to see visually as the gap in the `EC` activation pattern gets filled in. You should be able to see that the missing elements are filled in as a result of `CA3` units getting activated. Interestingly, you should also see that as these missing elements start to get filled in, the `ECout` activation feeds back to `ECin` and thus back through the `DG` and `CA3` layers, which can result in a shifting of the overall activation pattern. This is a "big loop" pattern completion process that complements the much quicker (often hard to see) pattern completion within `CA3` itself due to lateral excitatory connections among `CA3` units.

# AC Training and Interference

* Select the `TstEpcPlot` tab, and restart with `Init` and now do `Step Run`. As in the `abac` model, this will automatically train on AB until your network gets 1 (100% correct) on the `AB Mem` score (during *testing* -- the `TrnEpcPlot` value shows the results from training which have the complete `B` pattern and are thus much better), and then automatically switch to AC and train until it gets perfect Mem as well.

You can now observe the amount of interference on AB after training on AC -- it will be some but probably not a catastrophic amount.  To get a better sense overall, we need to run multiple samples.

* Do `Train` to run 10 runs through AB / AC training.  Then click on the `RunStats` `Table` to get the final stats across all 10 runs.

> **Question 8.6:** Again report the `Mem:Mean` (average) level for the AB, AC, and Lure tests in the `RunStats` table.  How well does this result compare to the human results shown in Figure 1?

In summary, you should find that this hippocampal model is able to learn rapidly and with much reduced levels of interference compared to the prior cortical model of this same task. Thus, the specialized biological properties of the hippocampal formation, and its specialized role in episodic memory, can be understood from a computational and functional perspective.

# References

Ketz, N., Morkonda, S. G., & O’Reilly, R. C. (2013). Theta coordinated error-driven learning in the hippocampus. PLoS Computational Biology, 9, e1003067.

Norman, K. A., & O’Reilly, R. C. (2003). Modeling hippocampal and neocortical contributions to recognition memory: A complementary-learning-systems approach. Psychological Review, 110(4), 611–646.

