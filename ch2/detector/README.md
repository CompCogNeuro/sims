Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

This simulation shows how an individual neuron can act like a detector, picking out specific patterns from its inputs and responding with varying degrees of selectivity to the match between its synaptic weights and the input activity pattern.

We will see how a particular pattern of weights makes a simulated neuron respond more to some input patterns than others. By adjusting the level of excitability of the neuron, we can make the neuron respond only to the pattern that best fits its weights, or in a more graded manner to other patterns that are close to its weight pattern. This provides some insight into why the point neuron activation function works the way it does.

# The Network and Input Patterns

We begin by examining the `NetView` tab, showing the Detector network. The network has an `Input` layer that will have patterns of activation in the shape of different digits, and these input neurons are connected to the receiving neuron (`RecvNeuron`) via a set of weighted synaptic connections. We can view the pattern of weights (synaptic strengths) that this receiving unit has from the input, which should give us an idea about what this unit will detect.

* Select `r.Wt` as the value you want to display (on the left side of the 3D network view) and then click on the `RecvNeuron` to view its receiving weights.

You should now see the `Input` grid lit up in the pattern of an `8`. This is the weight pattern for the receiving unit for connections from the input units, with the weight value displayed in the corresponding sending (input) unit. Thus, when the input units have an activation pattern that matches this weight pattern, the receiving unit will be maximally activated. Input patterns that are close to the target `8` input will produce graded activations as a function of how close they are. Thus, this pattern of weights determines what the unit detects, as we will see. First, we will examine the patterns of inputs that will be presented to the network.

* Click on the `DigitPats` button (next to the `Pats` label) on the left side of the window -- this *control panel* contains all the main ingredients for this model.

The display that comes up shows all of the different *input patterns* that will be presented ("clamped") onto the `Input` layer, so we can see how the receiving unit responds. Each row of the display represents a single *trial* that will be presented to the network. As you can see, the input data this case contains the digits from 0 to 9, represented in a simple font on a 5x7 grid of pixels (picture elements). Each pixel in a given event (digit) will drive the corresponding input unit in the network.

# Running the Network

To see the receiving neuron respond to these input patterns, we will present them one-by-one, and determine why the neuron responds as it does given its weights. Thus, we need to view the activations again in the network window.

* Select `Act` in the `NetView` to view activations, then click `Test Trial` in the toolbar at the top of the window.

This activates the pattern of a `0` (zero) in the `Input`, and shows 20 cycles of **settling** process where the activation of the receiving unit is iteratively updated over a series of **cycles** according to the point neuron activation function (just as the unit in the `neuron` simulation was updated over time).  We have selected 20 cycles as enough time for the receiving neuron to fully respond to the input.

The receiving unit showed an activity value of 0 because it was not activated above threshold by the `0` input pattern.  Before getting into the nitty-gritty of why the unit responded this way, let's proceed through the remaining digits and observe how it responds to other inputs.

* Press `Test Trial` for each of the other digits, until the number `8` shows up. 

You should have seen the receiving unit finaly activated when the digit `8` was presented, with an activation of zero for all the other digits. Thus, as expected, the receiving unit acts like an `8` detector: only when the input perfectly matches the input weights is there enough excitatory input to drive the receiving neuron above its firing threshold.

* You can use the "VCR" style buttons after the `Time` label at the bottom right of the `NetView` to review each cycle of updating, to see the progression of activation over time.

* Go ahead and do one more `Test Trial` to see what happens with `9`.

We can use a graph to view the pattern of receiving unit activation across the different input patterns.

* Click on the `TstTrlPlot` (test trial plot) tab.

The graph shows the activation (`Act`) for the unit as a function of trial (and digit) number along the X axis. You should see a flat line with a single peak at 8.  

# Computing Excitatory Conductance `Ge` (Net Input)

Now, let's try to understand exactly why the unit responds as it does. The key to doing so is to understand the relationship between the pattern of weights and the input pattern.

* Click the `Pats` again (or find the already-open window if you didn't close it), and place that scrolled to the `8` digit next to the netview window, so you can see both.  Then do `Init` in the toolbar and `Test Trial` for each input digit in turn.

> **Question 2.8:** For each digit, report the number of active Input units where there is also a weight of 1 according to the `8` digit pattern.  In other words, report the *overlap* between the input activity and the weight pattern.

The number of inputs having a weight of 1 that you just calculated should correspond to the total excitatory input `Ge`, also called the **net input**, going into the receiving unit, which is a function of the average of the sending activation `Act` times the weight `Wt` over all the units, with a correction factor for the expected activity level in the layer, `Alpha`:

```
Ge = (1 / Alpha) * [Sum(Act * Wt) / N]
```

You can click on `Ge` in the netview to see these values as you step through the inputs.

* Do `Init` and `Test Trial` to see the `0` input again.  If you hover over the RecvUnit with your mouse, you should see it has a value of `Ge = .3352..`.  To apply the above equation, you should have observed that `0` has 6 units in common with `8`, and `N` = 35 (7*5), so that is about .1714.  Next, we need to apply the `Alpha` correction factor, which we set to be the activity level of the `8`, which is 17 of the 35 units active.  Thus, we should get:

```
Ge = (1 / (17 / 35)) * (6 / 35) = .3529...
```

That is not quite right.  The problem is that our `Input` activations are not quite 1, but rather .95, because it is impossible for our rate-coded neuron activations to get all the way to 1, so we clip them at a maximum of .95.  If you multiply the above number by .95, you indeed get the correct `Ge = .3352..`!  

As a result of working through the Ge net input calculation, you should now have a detailed understanding of how the net excitatory input to the neuron reflects the degree of match between the input pattern and the weights. You have also observed how the activation value can ignore much of the graded information present in this input signal, due to the presence of the **threshold**.  This gives you a good sense for *why* neurons have these thresholds: it allows them to filter out all the "sub-threshold noise" and only communicate a clear, easily-interpreted signal when it has detected what it is looking for.

# Manipulating Leak

Next, we will explore how we can change how much information is conveyed by the activation signal. We will manipulate the leak current (`GbarL`), which has a default value of 2, which is sufficient to oppose the strength of the excitatory inputs for all but the strongest (best fitting) input pattern (the 8).

**IMPORTANT:** you must press `Init` for changes in `GbarL` to take effect!

* Reduce the `GbarL` value from 2 to 1.8, and do `Init` then `Test Trial` (you might want to change `ViewUpdt` to `AlphaCycle` instead of `Cycle` so it only shows the final result of setting for each input).  You can alternatively just hit `Test All` and look at the `TstTrlPlot`.

> **Question 2.9:** What happens to the pattern of receiving neuron activity over the different digits when you change GbarL to 1.8, 1.5, and 2.3 -- which input digits does it respond to for each case?  In terms of the tug-of-war model between excitatory and inhibition & leak (i.e., GbarL = leak), why does changing leak have this effect (a simple one-sentence answer is sufficient)?

> **Question 2.10:** Why might it be beneficial for the neuron to have a lower level of leak (e.g., GbarL = 1.8 or 1.5) compared to the original default value, in terms of the overall information that this neuron can convey about the input patterns it is "seeing"?

It is clearly important how responsive the neuron is to its inputs. However, there are tradeoffs associated with different levels of responsivity. The brain solves this kind of problem by using many neurons to code each input, so that some neurons can be more "high threshold" and others can be more "low threshold" types, providing their corresponding advantages and disadvantages in specificity and generality of response. The bias weights can be an important parameter in determining this behavior. As we will see in the next chapter, our tinkering with the value of the leak current Gbar.L is also partially replaced by the inhibitory input, which plays an important role in providing a dynamically adjusted level of inhibition for counteracting the excitatory net input. This ensures that neurons are generally in the right responsivity range for conveying useful information, and it makes each neuron's responsivity dependent on other neurons, which has many important consequences as one can imagine from the above explorations.


