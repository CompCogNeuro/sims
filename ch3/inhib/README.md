Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

![The Necker Cube](fig_necker_cube.png?raw=true "The Necker Cube")

# Introduction

This simulation explores how inhibitory interneurons can dynamically
control overall activity levels within the network, by providing both
feedforward and feedback inhibition to excitatory pyramidal neurons.
This inhibition is critical when neurons have bidirectional excitatory
connections, as otherwise the positive feedback loops will result in the
equivalent of epileptic seizures -- runaway excitatory activity.

The network in the right view panel contains a 10x10 unit input layer,
which projects to both the 10x10 hidden layer of excitatory units, and a
layer of 20 inhibitory neurons. These inhibitory neurons will regulate
the activation level of the hidden layer units, and should be thought of
as the inhibitory units for the hidden layer (even though they are in
their own layer for the purposes of this simulation). The ratio of 20
inhibitory units to 120 total hidden units (17 percent) is like that
found in the cortex, which is commonly cited as roughly 15 percent
(White, 1989a; Zilles, 1990). The inhibitory neurons are just like the
excitatory neurons, except that their outputs contribute to the
inhibitory conductance of a neuron instead of its excitatory
conductance. We have also set one of the activation parameters to be
different for these inhibitory neurons, as discussed below.

Let's begin as usual by viewing the weights of the network.

Most of the weights are random, except for those from the inhibitory
units, which are fixed at a constant value of .5. Notice also that the
hidden layer excitatory units receive from the input and inhibitory
units, while the inhibitory units receive feedforward connections from
the input layer, and feedback connections from the excitatory hidden
units, as well as inhibitory connections from themselves.

Now, we will run the network. Note the graph view above the network,
which will record the overall levels of activation (average activation)
in the hidden and inhibitory units.

You will see the input units activated by a random activity pattern, and
after several cycles of activation updating, the hidden and inhibitory
units will become active. The activation appears quite controlled, as
the inhibition counterbalances the excitation from the input layer. From
the average activity plotted in the graph above the network, you should
see that the hidden layer (black line) has around 20 percent activation.

In the next sections, we manipulate some of the parameters in the
control panel to get a better sense of the principles underlying the
inhibitory dynamics in the network. Because we will be running the
network many times, you may want to toggle the network display off to
speed up the settling process if your computer is running too slowly
(the graph log contains the relevant information anyway).

## Strength of Inhibitory Conductances

Let's start by manipulating the maximal conductance for the inhibitory
current into the excitatory units, , which multiplies the level of
inhibition coming into the hidden layer (excitatory) neurons. Clearly,
one would predict that this plays an important role.

Now, let's see what happens when we manipulate the corresponding
parameter for the inhibition coming into the inhibitory neurons, . You
might expect to get results similar to those just obtained for
ff_hidden_g_bar.i, but be careful -- inhibition upon inhibitory
neurons could have interesting consequences.

With a of .6, you should see that the excitatory activation drops, but
the inhibitory level stays roughly the same! With a value of 1.0, the
excitatory activation level increases, but the inhibition again remains
the same. This is a difficult phenomenon to understand, but the
following provide a few ways of thinking about what is going on.

First, it seems straightforward that reducing the amount of inhibition
on the inhibitory neurons should result in more activation of the
inhibitory neurons. If you just look at the very first blip of activity
for the inhibitory neurons, this is true (as is the converse that
increasing the inhibition results in lower activation). However, once
the feedback inhibition starts to kick in as the hidden units become
active, the inhibitory activity returns to the same level for all runs.
This makes sense if the greater activation of the inhibitory units for
the ff_inhib_g_bar.i = .6 case then inhibits the hidden units more
(which it does, causing them to have lower activation), which then would
result in *less* activation of the inhibitory units coming from the
feedback from the hidden units. This reduced activation of the
inhibitory neurons cancels out the increased activation from the lower
ff_inhib_g_bar.i value, resulting in the same inhibitory activation
level. The mystery is why the hidden units remain at their lower
activation levels once the inhibition goes back to its original
activation level.

One way we can explain this is by noting that this is a *dynamic*
system, not a static balance of excitation and inhibition. Every time
the excitatory hidden units start to get a little bit more active, they
in turn activate the inhibitory units more easily (because they are less
apt to inhibit themselves), which in turn provides just enough extra
inhibition to offset the advance of the hidden units. This battle is
effectively played out at the level of the *derivatives* (changes) in
activations in the two pools of units, not their absolute levels, which
would explain why we cannot really see much evidence of it by looking at
only these absolute levels.

A more intuitive (but somewhat inaccurate in the details) way of
understanding the effect of inhibition on inhibitory neurons is in terms
of the location of the thermostat relative to the AC output vent -- if
you place the thermostat very close to the AC vent (while you are
sitting some constant distance away from the vent), you will be warmer
than if the thermostat was far away from the AC output. Thus, how
strongly the thermostat is driven by the AC output vent is analogous to
the ff_inhib_g_bar.i parameter -- larger values of
ff_inhib_g_bar.i are like having the thermostat closer to the vent,
and will result in higher levels of activation (greater warmth) in the
hidden layer, and the converse for smaller values.

## Roles of Feedforward and Feedback Inhibition

Next we assess the importance and properties of the feedforward versus
feedback inhibitory projections by manipulating their relative
strengths. The control panel has two parameters that determine the
relative contribution of the feedforward and feedback inhibitory
pathways: applies to the feedforward weights from the input to the
inhibitory units, and applies to the feedback weights from the hidden
layer to the inhibitory units. These parameters (specifically the .rel
components of them) uniformly scale the strengths of an entire
projection of connections from one layer to another, and are the
arbitrary wt_scale.rel (r_k) relative scaling parameters described in
[Net Input Detail](/CCNBook/Neuron/NetInput "wikilink") from the [Neuron
Chapter](/CCNBook/Neuron "wikilink").

Due to the relative renormalization property of these .rel parameters,
you should see that the same overall level of inhibitory activity is
achieved, but it now happens quickly in a feedforward way, which then
clamps down on the excitatory units from the start -- they rise very
slowly, but eventually do achieve the same levels as before.

These exercises should help you to see that a combination of both
feedforward and feedback inhibition works better than either alone, for
clear principled reasons. Feedforward can anticipate incoming activity
levels, but it requires a very precise balance that is both slow and
brittle. Feedback inhibition can react automatically to different
activity levels, and is thus more robust, but it is also purely
reactive, and thus can be unstable and oscillatory unless coupled with
feedforward inhibition.

## Effects of Learning

One of the important things that inhibition must do is to compensate
adequately for the changes in weight values that accompany learning.
Typically, as units learn, they develop greater levels of variance in
the amount of excitatory input received from the input patterns, with
some patterns providing strong excitation to a given unit and others
producing less. This is a natural result of the specialization of units
for representing (detecting) some things and not others. We can test
whether the current inhibitory mechanism adequately handles these
changes by simulating the effects of learning, by giving units
excitatory weight values with a higher level of variance.

In this case, the network's weights are produced by generating random
numbers with a mean of .25, and a uniform variance around that mean of
.2.

The weights are then initialized with the same mean but a variance of .7
using Gaussian (normally) distributed values. This produces a much
higher variance of excitatory net inputs for units in the hidden layer.
There is also an increase in the total overall weight strength with the
increase in variance because there is more room for larger weights above
the .25 mean, but not much more below it.

You should observe a greater level of excitation using the trained
weights compared to the initial untrained weights.

## Bidirectional Excitation

To make things simpler at the outset, we have so far been exploring a
relatively easy case for inhibition where the network does not have
bidirectional excitatory connectivity, which is where inhibition really
becomes essential to prevent runaway positive feedback dynamics. Now,
let's try running a network with two bidirectionally connected hidden
layers.

In extending the network to the bidirectional case, we also have to
extend our notions of what feedforward inhibition is. In general, the
role of feedforward inhibition is to anticipate and counterbalance the
level of excitatory input coming into a layer. Thus, in a network with
bidirectional excitatory connectivity, the inhibitory neurons for a
given layer also have to receive the top-down excitatory connections,
which play the role of "feedforward" inhibition.

The graph view shows the average activity for only the first hidden and
inhibitory layers (as before). Note that the initial part up until the
point where the second hidden layer begins to be active is the same as
before, but as the second layer activates, it feeds back to the first
layer inhibitory neurons, which become more active, as do the excitatory
neurons. However, the overall activity level remains quite under control
and not substantially different than before. Thus, the inhibition is
able to keep the positive feedback dynamics fully in check.

Next, we will see that inhibition is differentially important for
bidirectionally connected networks.

This reduces the amount of inhibition on the excitatory neurons. Note
that this has a relatively small impact on the initial, feedforward
portion of the activity curve, but when the second hidden layer becomes
active, the network becomes catastrophically over activated -- an
epileptic fit!

## Exploration of kWTA Inhibition

This should reproduce the standard activation graph for the case with
actual inhibitory neurons.

The *k* value of this function is set according to the in the control
panel (this proportion value of .20 is automatically translated into a
corresponding k value, 20 in this case, by the software).

The activations will be fairly low, well below the 20% activity target.
This is because of the untrained weights -- not enough neurons get above
threshold with these low weight values, and the default .1 leak levels.
This demonstrates that kWTA is really k *or less* WTA -- there is
flexibility on the lower end depending on how much excitation the
neurons actually receive.

You should see the hidden activities approach the 20% target now (they
will be less than 20% due to graded activation levels, even though
roughly 20% of the neurons are actually active). Now you should be able
to see more clearly how smoothly the activation dynamics are controlled
-- levels rise up quickly and just stabilize at the target level. kWTA
is very efficient by exactly anticipating the correct level of
inhibition.
