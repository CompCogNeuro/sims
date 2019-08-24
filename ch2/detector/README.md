# Introduction

*Summary:* This simulation shows how an individual neuron can act like a detector, picking out specific patterns from its inputs and responding with varying degrees of selectivity to the match between its synaptic weights and the input activity pattern.

We will see how a particular pattern of weights makes a simulated neuron respond more to some input patterns than others. By adjusting the level of excitability of the neuron, we can make the neuron respond only to the pattern that best fits its weights, or in a more graded manner to other patterns that are close to its weight pattern. This provides some insight into why the point neuron activation function works the way it does.

# The Network and Input Patterns

We begin by examining the Network View panel (right-most window). The network has an  layer that will have patterns of activation in the shape of different digits, and these input neurons are connected to the receiving neuron via a set of weighted synaptic connections. Be sure you are familiar with the operation of the Net View, which is explained in
the Emergent wiki here: . We can view the pattern of weights (synaptic strengths) that this receiving unit has from the input, which should give us an idea about what this unit will detect.

You should now see the input grid lit up in the pattern of an 8. This is the weight pattern for the receiving unit for connections from the input units, with the weight value displayed in the corresponding sending (input) unit. Thus, when the input units have an activation pattern that matches this weight pattern, the receiving unit will be maximally activated. Input patterns that are close to the target 8 input will produce graded activations as a function of how close they are. Thus, this pattern of weights determines what the unit detects, as we will see. First, we will examine the patterns of inputs that will be presented to the network.

The display that comes up shows all of the different *input patterns* (input data) that will be presented to the receiving unit to measure its detection responses. Each row of the display represents a single *event* that will be presented to the network. As you can see, the input data this case contains the digits from 0 to 9, represented in a simple font on a 5x7 grid of pixels (picture elements). Each pixel in a given event (digit) will drive the corresponding input unit in the network.

# Running the Network

To see the receiving neuron respond to these input patterns, we will present them one-by-one, and determine why the neuron responds as it does given its weights. Thus, we need to view the activations again in the network window.

Each Step causes the input units to have their activation values fixed or **clamped** to the values given in the input pattern of the event, followed by a **settling** process where the activation of the receiving unit is iteratively updated over a series of **cycles** according to the point neuron activation function (just as the unit in the previous simulation was updated over time). This settling process continues until the activations in the network approach an *equilibrium* (i.e., the change in activation from one cycle to the next, shown as variable da in the simulator, is below some tolerance level). The network view is updated during every step of this settling process, but it happens so quickly you likely won't really see it.

You should have seen the input pattern of the digits 0 and 1 in the input layer. However, the receiving unit showed an activity value of 0 for both inputs, meaning that it was not activated above threshold by these input patterns. Before getting into the nitty-gritty of why the unit responded this way, let's proceed through the remaining digits and observe how it responds to other inputs.

You should have seen the receiving unit activated when the digit 8 was presented, with an activation of zero for all the other digits. Thus, as expected, the receiving unit acts like an 8 detector.

We can use a graph to view the pattern of receiving unit activation across the different input patterns.

The graph shows the activation for the unit as a function of trial (and digit) number along the X axis. You should see a flat line with a single peak at 8.

# Computing Net Input

Now, let's try to understand exactly why the unit responds as it does. The key to doing so is to understand the relationship between the pattern of weights and the input pattern. Thus, we will configure the NetView to display both the weights and the current input pattern.

The number of inputs having a weight of 1 that you just calculated should correspond to the total excitatory input or **net input** to the receiving unit:

  - {{\< math \>}} g_e(t) = \\frac{1}{n} \\sum_i x_i w_i {{\< /math
    \>}}

Let's confirm this.

Compare the values in the graph with the numbers you computed -- they should be proportional to each other -- there are some normalization factors that convert the raw counts into the actual net input values shown in the graph.

As a result of working through the net input calculation, you should now have a detailed understanding of how the net excitatory input to the neuron reflects the degree of match between the input pattern and the weights. You have also observed how the activation value can ignore much of the graded information present in this input signal. Now, we will explore how we can change how much information is conveyed by the activation signal. We will manipulate the leak current (), which has a default value of 2, which is sufficient to oppose the strength of the excitatory inputs for all but the strongest (best fitting) input pattern (the 8).

# Manipulating Leak

It is clearly important how responsive the neuron is to its inputs. However, there are tradeoffs associated with different levels of responsivity. The brain solves this kind of problem by using many neurons to code each input, so that some neurons can be more "high threshold" and others can be more "low threshold" types, providing their corresponding advantages and disadvantages in specificity and generality of response. The bias weights can be an important parameter in determining this behavior. As we will see in the next chapter, our tinkering with the value of the leak current g_bar.l is also partially replaced by the inhibitory input, which plays an important role in providing a dynamically adjusted level of inhibition for counteracting the excitatory net input. This ensures that neurons are generally in the right responsivity range for conveying useful information, and it makes each neuron's responsivity dependent on other neurons, which has many important consequences as one can imagine from the above explorations.

Finally, we can peek under the hood of the simulator to see how events are presented to the network. This is done using something called a **Program**, which is like a conductor that orchestrates the presentation of the events in the input data to the network. We interact with programs through program *control panels* (not to be confused with the overall simulation control panels -- see the Wiki page:  for more info).

Although the simulation exercises will not typically require you to access these program control panels directly, they are always an option if you want to obtain greater control, and you will have to rely on them when you make your own simulations.
