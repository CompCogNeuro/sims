Back to [All Sims](https://github.com/CompCogNeuro/sims)

# Introduction

This simulation illustrates the basic properties of neural spiking and
rate-code activation, reflecting a balance of excitatory and inhibitory
influences (including leak and synaptic inhibition).

# Orientation to the Software (ControlPanel and Views)

As this is the first simulation project in the textbook, we begin with
some introductory orientation (see the [Emergent](/Emergent "wikilink")
documentation for more complete information). All of the major controls
and parameters for the simulation are contained within the (NOTE: these
links work when viewing this documentation within the emergent
simulator, and not otherwise) object located in the middle of the 3
panels visible in the main project window (you can access it from the
tab at the top of the middle panel). The right panel contains various 3D
graphical displays of simulation data, including the network (NetView or
Network View) and various graphs and grid-like displays (Graph view,
Grid view).

In this simulation, there are two different ways to view the results,
selectable by the tabs at the top of the right side of the window:

-   The tab shows the (very simple) network that is being simulated,
    with a single sending (input) neuron (at the bottom) that sends
    activation to the receiving neuron (at the top). We are primarily
    concerned with how the receiving neuron responds to the activation
    input from the sending neuron.
-   The tab shows a graph of the receiving unit's main variables (see
    below for details) over time, in response to the sending activation.

We will see this single input being turned on and then off again, and
observe the response of the receiving neuron. To see this, we can run
the simulation.

At the bottom of the are 4 buttons: :

1.  -- initializes the graph display and starts the simulation over from
    wherever it might have left off.

2.  -- runs the full set of cycles of activation updating (updating of
    the equations that govern the behavior of the neural unit),
    displaying the results in the Network and CycleOutputData frames on
    the right hand side of the window.

3.  -- runs one single cycle of activation updating (in more complex
    models, multiple levels of stepping will be available)

4.  -- if running, this will stop running.

You should see that very shortly after the input neuron comes on
(indicated by the yellow color), the receiving neuron is activated by
this input, firing a sequence of discrete action potentials or *spikes*.
To get a better idea of the precise trajectory of this activation, it is
much more convenient to use the **Graph View**, which displays the
information graphically over time, allows multiple variables to be
viewed at the same time, and even allows multiple runs (e.g., with
different parameters) to be compared with each other.

# The Graph View

-   Press the tab in the right panel to display the graph view display.

Only the excitatory and leak currents are operating here, with their
conductances () and reversal potentials () as shown in the control
panel.

-   Press the button and then the button on the control panel to display
    a new graph.

This produces a plot using the current (default) parameters. You should
see various lines plotted over 200 time steps (*cycles*) on the X axis.

Here is a quick overview of each of the variables -- we'll go through
them individually next (see for more details on how to determine what is
being graphed, and how to configure it):

-   (black line) = net input, which is the total excitatory input to the
    neuron (net = g_e(t) \* g_bar_e). g_e(t) is the proportion of
    excitatory ion channels open, and it goes from 0 prior to cycle 10,
    to 1 from 10-160, and back to 0 thereafter. Because g_bar_e = .3
    (by default), the net value goes up to .3 from cycle 10-160. The
    timing of the input is controlled by the parameters and (the total
    number of cycles is controlled by ).

-   (red line) = net current (sum of individual excitation and leak
    currents), which is excitatory (upward) when the excitatory input
    comes on, and then oscillates as the action potential spikes fire.
    In general this reflects the net balance between the excitatory net
    input and the constant leak current (plus inhibition, which is not
    present in this simulation).

-   (blue line) = membrane potential, which represents integration of
    all inputs into neuron. This starts out at the resting potential of
    .3 (= -70mV in biological units), and then increases with the
    excitatory input. As you can see, the net current (I_net) shows the
    <i>rate of change</i> of the membrane potential. When v_m gets
    above about .5, a spike is fired, and v_m is then reset back to .3,
    starting the cycle over again.

-   (green line) = activation. This shows the amount of activation sent
    to other units -- by default in this model it is set to discrete
    SPIKE mode, so it is 0 except on the cycle when v_m gets over
    threshold. When act_fun is set to the rate-code NOISY_XX1
    function, it reflects the expected rate of spiking as a number
    between 0-1. Note - the green line is often hidden by the act_eq
    line- to see it, click on the CycleOutputData graph, then open the
    CycleOutputData tab in the ControlPanel then uncheck all the other
    lines that may be hiding it.

-   (purple line) = rate-code equivalent activation -- computes a
    running average of spikes per cycle in SPIKE mode, providing a
    measure of the rate-code value corresponding to the current spiking
    behavior. If a rate code like NOISY_XX1 is being used, then it is
    the same as act, and is hidden by that value (all you see is the
    green line).

-   (orange line) = adaptation variable -- increases during spikes, and
    decays somewhat in between, building up over time to cause the rate
    of spiking to adapt or slow down over time.

# Spiking Behavior

The default parameters that you just ran show the spiking behavior of a
neuron. This is implementing a modified version of the [Adaptive
Exponential](/CCNBook/Neuron/AdEx "wikilink") or AdEx model, which has
been shown to provide a very good reproduction of the firing behavior of
real cortical pyramidal neurons. As such, this is a good representation
of what real neurons do. We have turned off the exponential aspect of
the AdEx model here to make parameter manipulations more reliable -- a
spike is triggered when the membrane potential Vm crosses a simple
threshold of .5. (In contrast, when exponential is activated (with the
flag), the triggering of a spike is more of a dynamic exponential
process around this .5 threshold level, reflecting the strong
nonlinearity of the sodium channels that drive spiking.)

At the broadest level, you can see the periodic green spikes that fire
as the membrane potential gets over the firing threshold, and it is then
reset back to the rest level, from which it then climbs back up again,
to repeat the process again and again. Looking at the overall rate of
spiking as indexed by the spacing between spikes, you can see that it
decreases over time -- this is due to the **adaptation** property of the
AdEx model -- the spike rate adapts over time.

From the tug-of-war model, you should expect that increasing the amount
of excitation coming into the neuron will increase the rate of firing,
by enabling the membrane potential to reach threshold faster, and
conversely decreasing it will decrease the rate of firing. Furthermore,
increasing the leak or inhibitory conductance will tug more strongly
against a given level of excitation, causing it to reach threshold more
slowly, and thus decreasing the rate of firing.

This intuitive behavior is the essence of what you need to understand
about how the neuron operates -- now let's see it in action.

# Manipulating Parameters

Now we will use some of the parameters in the control panel to explore
the properties of the point neuron activation function.

## Excitatory

First, we will focus on , which controls the amount of excitatory
conductance. In general, we are interested in seeing how the unit
membrane potential reflects a balance of the different inputs coming
into it (here just excitation and leak), and how the spiking rate
responds to the resulting membrane potential.

By systematically searching the parameter range for g_bar.e between .1
and .2, you should be able to locate the point at which the membrane
potential just reaches threshold.

## Leak

You can also manipulate the value of the leak conductance, , which
controls the size of the leak current -- recall that this pulls the
opposite direction as the excitatory conductance in the neural
tug-of-war.

## Driving / Reversal Potentials

You should see that decreasing reduces the spiking rate, because it
makes the excitatory input pull less strongly up on the membrane
potential. Increasing produces greater spiking by making leak pull less
strongly down.

# Rate Coded Activations

Next, we'll see how the discrete spiking behavior of the neuron can be
approximated by a continuous rate-coded value. The blue line in the
graphs has been tracking the actual rate of spiking to this point -- it
goes up when a spike occurs and then decreases slowly in the interim,
with the aggregate value over time becoming a closer reflection of the
actual spiking rate. But the NOISY_XX1 activation function can directly
compute a rate-code value for the neuron, instead of just measuring the
observed rate of spiking. As explained in the Neuron chapter, this rate
code activation has several advantages (and a few disadvantages) for use
in neural simulations, and is what we typically use.

You should see that the green line in the graph now rises up and then
decreases slowly due to accommodation, without the discrete spiking
values observed before. Similarly, the blue membrane potential value
rises up and decreases slowly as well, instead of being reset after
spiking.

You should have observed that the value tracks the actual spiking rate
reasonably well, indicating that NOISY_XX1 is a resonable approximation
to the actual neural spiking rate.

The resulting graph shows the values on the X axis plotted against the
NOISY_XX1 rate-code activation ( line) and actual SPIKE rate () on the
Y axis. This indicates that the rate code function is a reasonable
approximation of the spiking rate function, at least in capturing the
actual spike rate. In terms of information processing dynamics in the
network itself, discrete spiking is inevitably different from the rate
code model in many ways, so one should never assume that the two are
identical. Nevertheless, the practical benefits of using the rate-code
approximation are substantial and thus we often accept the risk to make
initial progress on understanding more complex cognitive functions using
this approximation.

# Noise

An important aspect of spiking in real neurons is that the timing and
intervals between spikes can be quite random, although the overall rate
of firing remains predictable. This is obviously not evident with the
single constant input used so far, which results in regular firing.
However, if we introduce noise by adding randomly generated values to
the net input, then we can see a more realistic level of variability in
neural firing. Note that this additional noise plays a similar role as
the convolution of noise with the XX1 function in the noisy XX1
function, but in the case of the noisy XX1 we have a deterministic
function that incorporates the averaged effects of noise, while here we
are actually adding in the random values themselves, making the behavior
stochastic.

-   Change the variance of the noise generator ( in the control panel)
    from 0 to .2, and press and then . You should see the red line is
    now perturbed significantly with the noise.

It can be difficult to tell from a single run whether the spike timing
is random -- the unit still fires with some regularity.

Even with this relatively high level of noise, the spike timing is not
completely uniform -- the spikes still form clusters at relatively
regulalry-spaced intervals. If you increase all the way to .5, the
spikes will be more uniformly distributed. However, note that even with
the high levels of variability in the specific spike timing, the overall
rate of spiking recorded at the end of the input does not change that
much. Thus, the rate code is a highly robust reflection of the overall
net input.

In the brain (or large networks of simulated spiking neurons), there are
high levels of variability in the net input due to variability in the
spike firing of the different inputs coming into a given neuron. As
measured in the brain, the statistics of spike firing are captured well
by a *Poisson* distribution, which has variability equal to the mean
rate of spiking, and reflects essentially the maximum level of noise for
a given rate of spiking. Neurons are noisy.

# Adaptation

Cortical pyramidal neurons exhibit the property of spike rate
adaptation. We are now using a more advanced form of adaptation than the
form from the original AdEx model, based on sodium-gated potassium
channels (K_na), as determined by the parameters shown in the control
panel. You can explore the basic effect of adaptation by turning this on
and off.

You should observe that spiking is perfectly regular throughout the
entire period of activity without adaptation, whereas with adaptation
the rate decreases significantly over time. One benefit of adaptation is
to make the system overall more sensitive to changes in the input -- the
biggest signal strength is present at the onset of a new input, and then
it "habituates" to any constant input. This is also more efficient, by
not continuing to communicate spikes at a high rate for a constant input
signal that presumably has already been processed after some point.

