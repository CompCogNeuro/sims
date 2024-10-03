*Back to [All Sims](https://github.com/CompCogNeuro/sims)*

# Introduction

This simulation illustrates the basic properties of neural spiking and rate-code activation, reflecting a balance of excitatory and inhibitory influences (including leak and synaptic inhibition).

In this model, the `Network` only shows a single neuron which is "injected" with excitatory current (as neuroscientists might do with an electrode injecting current into a single neuron).  If you do `Run Cycles` in the toolbar you will see it get activated, but to really understand what is going on, we need to see the relationship among multiple variables as shown in the `Test Cycle Plot`.

# Plot of Neuron variables over time

* Press the `Test Cycle Plot` tab in the right panel to display the graph view display.  If you haven't done `Run Cycles` yet, do it now so you can see the results of running with the default parameters.

Only the excitatory and leak currents are operating here, with their conductances (`GbarE`, `GbarL`) as shown in the control panel.  You should
see various lines plotted over 200 time steps (*cycles*) on the X axis.

Here is a quick overview of each of the variables -- we'll go through them individually next (see for more details on how to determine what is being graphed, and how to configure it):

* `Ge` = total excitatory input conductance to the neuron, which is generally a function of the number of open excitatory synaptic input channels at any given point in time (`Ge(t)`) and the overall strength of these input channels, which is given by `GbarE`.  In this simple model, `Ge(t)` goes from 0 prior to cycle 10, to 1 from 10-160, and back to 0 thereafter. Because `GBarE = .3` (by default), the net value goes up to .3 from cycle 10-160. The timing of the input is controlled by the `OnCycle` and `OffCycle` parameters.

* `Inet` = net current (sum of individual excitation and leak    currents), which is excitatory (upward) when the excitatory input comes on, and then oscillates as the action potential spikes fire. In general this reflects the net balance between the excitatory net input and the constant leak current (plus inhibition, which is not present in this simulation).

* `Vm` = membrane potential, which represents integration of all inputs into neuron. This starts out at the resting potential of .3 (= -70mV in biological units), and then increases with the excitatory input. As you can see, the net current (Inet) shows the *rate of change* of the membrane potential while it is elevated prior to spiking. When Vm gets above about .5, a spike is fired, and Vm is then reset back to .3, starting the cycle over again.

* `Act` = activation. This shows the amount of activation (rate of firing) -- by default the model is set to discrete spiking, so this value is computed from the running-average measured inter-spike-interval (*ISI*).  It is first computed after the *second* spike, as that is the only point when the ISI is available.  If you turn the `Spike` setting to off, then the Act value is computed directly.

* `Spike` = discrete spiking -- this goes to 1 when the neuron fires a discrete spike, and 0 otherwise.

* `Gk` = conductance of sodium-gated potassium (k) channels, which drives adaptation -- this conductance increases during spikes, and decays somewhat in between, building up over time to cause the rate of spiking to adapt or slow down over time.

# Spiking Behavior

The default parameters that you just ran show the spiking behavior of a neuron. This is implementing a modified version of the Adaptive Exponential function (see [CCN Textbook](https://github.com/CompCogNeuro/book)) or AdEx model, which has been shown to provide a very good reproduction of the firing behavior of real cortical pyramidal neurons. As such, this is a good representation of what real neurons do. We have turned off the exponential aspect of the AdEx model here to make parameter manipulations more reliable -- a spike is triggered when the membrane potential Vm crosses a simple threshold of .5. (In contrast, when exponential is activated (you can find it in the `SpikeParams`), the triggering of a spike is more of a dynamic exponential process around this .5 threshold level, reflecting the strong nonlinearity of the sodium channels that drive spiking.)

At the broadest level, you can see the periodic spikes that fire as the membrane potential gets over the firing threshold, and it is then reset back to the rest level, from which it then climbs back up again, to repeat the process again and again. Looking at the overall rate of spiking as indexed by the spacing between spikes (i.e., the *ISI* or inter-spike-interval), you can see that the spacing increases over time, and thus the rate decreases over time.  This is due to the **adaptation** property of the AdEx model -- the spike rate adapts over time.

From the tug-of-war model, you should expect that increasing the amount of excitation coming into the neuron will increase the rate of firing, by enabling the membrane potential to reach threshold faster, and conversely decreasing it will decrease the rate of firing. Furthermore, increasing the leak or inhibitory conductance will tug more strongly against a given level of excitation, causing it to reach threshold more slowly, and thus decreasing the rate of firing.

This intuitive behavior is the essence of what you need to understand about how the neuron operates -- now let's see it in action.

# Manipulating Parameters

Now we will use some of the parameters in the control panel to explore the properties of the point neuron activation function.

## Excitatory

First, we will focus on `GbarE`, which controls the amount of excitatory conductance. In general, we are interested in seeing how the neuron membrane potential reflects a balance of the different inputs coming into it (here just excitation and leak), and how the spiking rate responds to the resulting membrane potential.

* Increase `GbarE` from .3 to .4 (and then do `Run Cycles` to see the effects). Then observe the effects of decreasing GbarE to .2 and all the way down to .1. 

> **Question 2.1:** Describe the effects on the rate of neural spiking of increasing `GbarE` to .4, and of decreasing it to .2, compared to the initial value of .3 (this is should have a simple answer).

> **Question 2.2:** Is there a qualitative difference in the neural spiking when `GbarE` is decreased to .1, compared to the higher values -- what important aspect of the neuron's behavior does this reveal?

By systematically searching the parameter range for `GbarE` between .1 and .2, you should be able to locate the point at which the membrane potential just reaches threshold.

> **Question 2.3:** To 2 decimal places (e.g., 0.15), what value of `GbarE` puts the neuron just over threshold, such that it spikes at this value, but not at the next value below it?

* Note: you can see the specific numerical values for any point in the graph by hovering the mouse over the point.  It will report which variable is being reported as well as the value.

> **Question 2.4 (advanced):** Using one of the equations for the equilibrium membrane potential from the Neuron chapter, compute the exact value of excitatory input conductance required to keep Vm in equilibrium at the spiking threshold. Show your math. This means rearranging the equation to have excitatory conductance on one side, then substituting in known values. (note that: Gl is a constant = .3; Ge is 1 when the input is on; inhibition is not present here and can be ignored) -- this should agree with your empirically determined value.

## Leak

You can also manipulate the value of the leak conductance, , which controls the size of the leak current -- recall that this pulls the opposite direction as the excitatory conductance in the neural tug-of-war.

* Press the `Defaults` button in the toolbar to restore the default parameters, then manipulate the `GbarL` parameter in .1 increments (.4, .5, .2 etc) and observe the effects on neural spiking. 

> **Question 2.5:** What value of GbarL just prevents the neuron from being able to spike (in .1 increments) -- explain this result in terms of the tug-of-war model relative to the GbarE excitatory conductance.

> **Question 2.6 (advanced):** Use the same technique as in question 2.4 to directly solve for the value of GbarL that should put the neuron right at it's spiking threshold using the default values of other parameters -- show your math.
 

## Driving / Reversal Potentials

* Press `Defaults` in the toolbar to restore the default parameters. Then manipulate the `ErevE` and `ErevL` parameters and observe their effects on the spiking rate. 

You should see that decreasing `ErevE` reduces the spiking rate, because it makes the excitatory input pull less strongly up on the membrane potential. Increasing `ErevL` produces greater spiking by making leak pull less strongly down.

# Rate Coded Activations

Next, we'll see how the discrete spiking behavior of the neuron can be approximated by a continuous rate-coded value. The `Act` line in the graphs has been tracking the actual rate of spiking to this point, based on the inverse of the ISI.  The *Noisy X-over-X-plus-1* activation function can directly compute a rate-code activation value for the neuron, instead of just measuring the observed rate of spiking. As explained in the Neuron chapter, this rate code activation has several advantages (and a few disadvantages) for use in neural simulations, and is what we typically use.

* Press `Defaults` to start out with default parameters, then turn off the `Spike` parameter, and `Run Cycles` with the various parameter manipulations that you explored above. 

You should see that the Act line in the graph now rises up and then decreases slowly due to accommodation, without the discrete spiking values observed before. Similarly, the Vm membrane potential value rises up and decreases slowly as well, instead of being reset after spiking.

> **Question 2.7:** Compare the spike rates with rate coded activations by reporting the `Act` values just before cycle 160 (e.g., cycle 155) for GbarE = .2, .3, .4 with `Spike` = false, and the corresponding values in the `Spike` = true case for the same GbarE values. Hover the mouse over the `Act` line to get the exact value.

You should have observed that the `Act` value tracks the actual spiking rate reasonably well, indicating that *Noisy X-over-X-plus-1* is a resonable approximation to the actual neural spiking rate.

* To more systematically compare spiking vs. the rate-code function, click the `SpikeVsRatePlot` tab, and then click the `Spike Vs Rate` button in the toolbar -- this will alternate between these two functions for a range of `GbarE` values, and plot the results in the SpikeVsRate plot.

The resulting graph shows the `GbarE` values on the X axis plotted against the *Noisy X-over-X-plus-1* rate-code activation (`rate` line) and actual spiking rate (`spike`) on the Y axis. This indicates that the rate code function is a reasonable approximation of the spiking rate function, at least in capturing the actual spike rate. In terms of information processing dynamics in the network itself, discrete spiking is inevitably different from the rate code model in many ways, so one should never assume that the two are identical. Nevertheless, the practical benefits of using the rate-code approximation are substantial and thus we often accept the risk to make initial progress on understanding more complex cognitive functions using this approximation.

# Noise

An important aspect of spiking in real neurons is that the timing and intervals between spikes can be quite random, although the overall rate of firing remains predictable. This is obviously not evident with the single constant input used so far, which results in regular firing.  However, if we introduce noise by adding randomly generated values to the net input, then we can see a more realistic level of variability in neural firing. Note that this additional noise plays a similar role as the convolution of noise with the XX1 function in the noisy XX1 function, but in the case of the noisy XX1 we have a deterministic function that incorporates the averaged effects of noise, while here we are actually adding in the random values themselves, making the behavior stochastic.

* Change the variance of the noise generator (`Noise` in the control panel) from 0 to .2, and do `Run Cycles`. You should see the `Ge` line is now perturbed significantly with the noise.

It can be difficult to tell from a single run whether the spike timing is random -- the neuron still fires with some regularity.

* Do many Runs and observe the extent of variability in spikes as the plot updates.

Even with this relatively high level of noise, the spike timing is not completely uniform -- the spikes still form clusters at relatively regulalry-spaced intervals. If you increase `Noise` all the way to .5, the spikes will be more uniformly distributed. However, note that even with the high levels of variability in the specific spike timing, the overall rate of spiking recorded by `Act` at the end of the input does not change that much. Thus, the rate code is a highly robust reflection of the overall net input.

In the brain (or large networks of simulated spiking neurons), there are high levels of variability in the net input due to variability in the spike firing of the different inputs coming into a given neuron. As measured in the brain, the statistics of spike firing are captured well by a *Poisson* distribution, which has variability equal to the mean rate of spiking, and reflects essentially the maximum level of noise for a given rate of spiking. Neurons are noisy.

# Adaptation

Cortical pyramidal neurons exhibit the property of spike rate adaptation. We are now using a more advanced form of adaptation than the form from the original AdEx model, based on sodium-gated potassium channels (K_na), which is turned on by the `KNaAdpat` parameter in the control panel. You can explore the basic effect of adaptation by turning this on and off. 

You should observe that spiking is perfectly regular throughout the entire period of activity without adaptation, whereas with adaptation the rate decreases significantly over time. One benefit of adaptation is to make the system overall more sensitive to changes in the input -- the biggest signal strength is present at the onset of a new input, and then it "habituates" to any constant input. This is also more efficient, by not continuing to communicate spikes at a high rate for a constant input signal that presumably has already been processed after some point. As we will see in some other simulations later on, this adaptation also allows us to account for various perceptual and cognitive phenomena. 

For those who want to explore the software a bit more: If you want to make the adaptation effect more extreme, you can click on the "Neuron" label in the Netview, and a dialog box will open up. If you scroll down, you will see various parameters associated with the neuron layer, including `GBarE` and `GBarL` (which should be the same values as those you altered in the control panel). But you will also see others that were not in the control panel. To increase the effect of adaptation you can increase `GBarK` -- the magnitude of KNA adaptation effect as a conductance. Increase that from the default of 1 to a much larger value (e.g., 10) and you should see much stronger adaptation effects. 




