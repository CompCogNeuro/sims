Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

In this exploration of the hippocampus model, we will use the same basic AB--AC paired associates list learning paradigm as we used in the standard cortical network previously (`abac`). The hippocampus should be able to learn the new paired associates (AC) without causing undue levels of interference to the original AB associations (see Figure 1), and it should be able to do this much more rapidly than was possible in the cortical model. This model is using the _Theremin_ model of [Zheng et al., 2022](#references), which is an updated version of the _Theta Phase_ model of the hippocampus ([Ketz, Morkanda & O'Reilly, 2013](#references)). The EC <-> CA1 projections along with all the other connections have an error-driven learning component organized according to the theta phase rhythm. 

![AB-AC Data](fig_ab_ac_data_catinf.png?raw=true "AB-AC Data")

**Figure 1:** Data from people learning AB-AC paired associates, and comparable data from McCloskey & Cohen (1989) showing *catastrophic interference* of learning AC on AB.

* Click on `Train AB` and `Test AB` buttons to see how the AB training and testing lists are configured. The "A" pattern is the first three groups of units (at the bottom of each pattern, going left-right, bottom-top), and the "B" pattern is the next three, which you can see most easily in the `Test AB` patterns where these are blank (to be filled in by hippocampal pattern completion). The 2nd half of the pattern is the list context (as in the `abac` project).

# AB Training and Testing

Let's observe the process of activation spreading through the network during training.

* Set `Train Step` to `Cycle` instead of `Trial`, and do `Init`, `Step Cycle`.

You will see an input pattern from the AB training set presented to the network. As expected, during training, all three parts of the input pattern are presented (A, B, Context). You will see that activation flows from the `ECin` layer through the `DG, CA3` pathway and simultaneously to the `CA1`, so that the sparse `CA3` representation can be associated with the invertible `CA1` representation, which will give back this very `ECin` pattern if later recalled by the `CA3`.  You can use the Time VCR buttons in the lower right of the NetView to replay the settling process cycle-by-cycle.

* Set `Step` back to `Trial` and `Step Trial` through several more (but fewer than 10) training events, and observe the relative amount of pattern overlap between subsequent events on the `ECin, DG, CA3`, and `CA1` layers, by clicking back-and-forth between `ActQ0` (previous trial) and `ActP` (current trial), in the `Phase` group of variables.

You should have observed that the `ECin` patterns overlap the most, with `CA1` overlapping the next most, then `CA3`, and finally `DG` overlaps the least. The levels of FFFB overall inhibition parallel this result, with DG having a very high level of inhibition, followed by CA3, then CA1, and finally EC.

> **Question 7.4:** Using the explanation given earlier in the text about the pattern separation mechanism, and the relative levels of activity and inhibition in these different layers, explain the overlap results for each layer in terms of these activity levels, in qualitative terms.

Each epoch of training consists of the 10 list items, followed by testing on 3 sets of testing events. The first testing set contains the AB list items, the second contains the AC list items, and the third contains a set of novel _Lure_ items to make sure the network is treating novel items as such. The network automatically switches over to testing after each pass through the 10 training events.

* Set step to `Epoch` and `Step Epoch` to step through the rest of the training epoch and then automatically into the testing of the patterns. Switch to the `Train Epoch Plot`, and do `Step Epoch` again so 2 epochs have been run.  You should see the `Mem` line rise up, indicating about 50% or so of the items have been accurately remembered. Then switch back to the `Network` tab, press `Test Init`, change `Test Step` to `Cycle`, and do `Test Cycle` to see the testing input propagate through the network (be sure to change back to viewing `Act`).

You should observe that during testing, the input pattern presented to the network is missing the second associate as we saw earlier (the B or C item in the pair), and that as the activation proceeds through the network, it fills in this missing part in the EC layers (pattern completion) as a result of activation flowing up through the `CA3`, and back via the `CA1` to the `ECout`.

* Click on `Test Trial Plot` tab, and do `Test Run`.

You should see a plot of the overall `Mem` memory statistic for the `AB`, `AC`, and `Lure` items.  To see how these memory statistics are scored.  First click on the `TrgOnWasOffCmp` line for the plot, which shows how many units in `ECout` in the "comparison" region (where the B or C items are) that were _off_ but should have been _on_.  These are the features of B item that the hippocampus needs to recall, and this measure indicates the extent to which it does so, with a high value indicating that the network has failed to recall much of the probe pattern.  

Then click on the `TrgOffWasOn` line, which shows the opposite: any features that were erroneously activated but should have been off. Thus, a large `TrgOffWasOn` indicates that the network has _confabulated_ or otherwise recalled a different pattern than the cued one. When both of these measures are relatively low (below a threshold of .34), then we score the network as having correctly recalled the original pattern (i.e., `Mem` = 1). The threshold on these factors assumes a distributed representation of associate items, such that the entire pattern need not be recalled.

In general, you should see `TrgOnWasOffCmp` being larger than `TrgOffWasOn` -- the hippocampal network is "high threshold", which accords with extensive data on recollection and recall (see [Norman & O'Reilly, 2003](#references) for more discussion). 

* Do more train `Step Epoch` steps to do more learning on the AB items, until all the AB items are getting a `Mem = 1` score.

> **Question 7.5:** Report the total proportion of `Mem` responses for the AB, AC, and Lure tests.


# Detailed Testing: Pattern Completion in Action

Now that the network has learned something, we will go through the testing process in detail by stepping one cycle at a time.

* Click back on the `Network`, then do `Test Init` and then test `Step Cycle` so you can see the activation cycle-by-cycle for an AB pattern.

You should see the studied A stimulus, an empty gap where the B stimulus would be, and a list context representation for the AB list in the `Input` and `ECin`. You will see the network complete the B pattern, which you should be able to see visually as the gap in the `EC` activation pattern gets filled in. You should be able to see that the missing elements are filled in as a result of `CA3` units getting activated. Interestingly, you should also see that as these missing elements start to get filled in, the `ECout` activation feeds back to `ECin` and thus back through the `DG` and `CA3` layers, which can result in a shifting of the overall activation pattern. This is a "big loop" pattern completion process that complements the much quicker (often hard to see) pattern completion within `CA3` itself due to lateral excitatory connections among `CA3` units.

# AC Training and Interference

* Select the `Test Epoch Plot` tab, and restart with `Init` and now do `Step Run`. As in the `abac` model, this will automatically train on AB until your network gets 1 (100% correct) on the `AB Mem` score (during _testing_ -- the `Train Epoch Plot` value shows the results from training which have the complete `B` pattern and are thus much better), and then automatically switch to AC and train until it gets perfect Mem as well.

You can now observe the amount of interference on AB after training on AC -- it will be some but probably not a catastrophic amount.  To get a better sense overall, we need to run multiple samples.

* Do `Train Run` to run 10 runs through AB / AC training, and click on the `Train Run Plot` to see the results, with the `Tst*Mem` stats from the testing run. Then click on the `RunStats Plot`, which reports summary statistics on the `TstABMem` results.

> **Question 7.6:** Report the `TstABMem:Mean` (average) values for the AB items. In general the AC and Lure items should all be at 1 and 0 respectively. How well does this result compare to the human results shown in Figure 1?

In summary, you should find that this hippocampal model is able to learn rapidly and with much reduced levels of interference compared to the prior cortical model of this same task. Thus, the specialized biological properties of the hippocampal formation, and its specialized role in episodic memory, can be understood from a computational and functional perspective.

# References

* Ketz, N., Morkonda, S. G., & O’Reilly, R. C. (2013). Theta coordinated error-driven learning in the hippocampus. PLoS Computational Biology, 9, e1003067. http://www.ncbi.nlm.nih.gov/pubmed/23762019  [PDF](https://ccnlab.org/papers/KetzMorkondaOReilly13.pdf)

* Norman, K. A., & O’Reilly, R. C. (2003). Modeling hippocampal and neocortical contributions to recognition memory: A complementary-learning-systems approach. Psychological Review, 110(4), 611–646. [PDF](https://ccnlab.org/papers/NormanOReilly03.pdf)

* Zheng, Y., Liu, X. L., Nishiyama, S., Ranganath, C., & O’Reilly, R. C. (2022). Correcting the hebbian mistake: Toward a fully error-driven hippocampus. PLOS Computational Biology, 18(10), e1010589. https://doi.org/10.1371/journal.pcbi.1010589 [PDF](https://ccnlab.org/papers/ZhengLiuNishiyamaEtAl22.pdf)

