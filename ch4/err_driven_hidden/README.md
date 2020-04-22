Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

This simulation shows how XCAL error driven learning can train a hidden layer to solve problems that are otherwise impossible for a simple two layer network (as we saw in the Pattern Associator (`pat_assoc`) exploration, which should be completed first before doing this one).

# Exploration

This project is identical to the pattern associator one, with one major exception: we have introduced a `Hidden` layer with 4 units.  Note that there are only feedforward connections from the input to this hidden layer, because the input is clamped in both minus and plus phases and so would not be affected by feedback connections anyway, but that there are bidirectional connections between the `Hidden` and `Output` layers, as required for the XCal algorithm to be able to do error-driven learning. By default, the Learn rule value is set to `ErrorDriven`, and `Pats` is `Impossible`.

* Click on `Init` and then `Step Run`. 

As in the pattern associator project, the `TrnEpcPlot` displays the SSE error measure over epochs of training.  The `TstTrlLog` is updated after every 5 epochs of training, and it also shows the states of the hidden units.

Also as before, the training of the network stops automatically after it gets the entire training set correct 5 epochs in a row (i.e., `NZeroStop` = 5). Note that this 5 correct repetitions criterion filters out the occasional spurious solutions that can happen due to the somewhat noisy behavior of the network during learning, as evidenced by the jagged shape of the learning curve. The reason for this noisy behavior is that a relatively small change in the weight can lead to large overall changes in the network's behavior due to the bidirectional activation dynamics, which produces a range of different responses to the input patterns.

This sensitivity of the network is a property of all attractor networks (i.e., networks having bidirectional connectivity), but is not typical of feedforward networks. Thus, a feedforward backpropagation network learning this same task will have a smooth, monotonically decreasing learning curve (see Backpropogation SubTopic). Some people have criticized the nature of learning in attractor networks because they do not share the smoothness of backpropagation. However, we find the benefits of bidirectional connectivity and attractor dynamics, together with the biological plausibility of the learning rule, to far outweigh the aesthetics of the learning curve. Furthermore, larger networks exhibit smoother learning, because they have more "mass" and are thus less sensitive to small weight changes.

* Press `Init` and then `Step Run` several times to get a sense of how fast the network learns in general. You can click on the `RunPlot` tab to just see a summary of how long it took to get to the first zero epoch SSE.  Note that there will be an occasional set of initial weights that fails to learn (which shows a -1 for the first zero epoch) -- it just gets stuck in an attractor that prevents further learning.
 
> **Question 4.7a:** Provide a general characterization of how many epochs it took for your network to learn this impossible problem (e.g., slowest, fastest, rough average).

* Press `Test Trial` 4 times to see the testing results and refresh the `TstTrlPlot` after learning..

> **Question 4.7b:** Can you figure out how the hidden units have made the problem solvable, whereas it was impossible for a network without a hidden layer? Step through a few trials in a network after learning and report which hidden units are active for the four events, and also report the weights from these units to the output units. (note this will differ across network runs - just give one example).

In general, the hidden layer categorizes two of the non-overlapping input patterns into the same representation, which then makes it easy to drive the appropriate output unit. This is a specific case of the more general principle that hidden layers enable the network to transform or categorize the input patterns in "smarter" ways, enabling all manner of more abstract patterns to be recognized. A good way to test this is to analyze the patterns of hidden unit activity for the different input patterns in a network that never learns the problem. Think about how you might modify the network architecture so that it is likely to be able to reliably learn the problem every time. You can try to test this yourself (you can ask the professor or teaching assistant for help to do this).
