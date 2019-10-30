Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

This simplified basal ganglia (BG) network shows how dopamine bursts can
reinforce *Go* (direct pathway) firing for actions that lead to reward,
and dopamine dips reinforce *NoGo* (indirect pathway) firing for actions
that do not lead to positive outcomes, producing Thorndike's classic
*Law of Effect* for instrumental conditioning, and also providing a
mechanism to learn and select among actions with different reward
probabilities over multiple experiences.

This model is based on the connection between dopamine D1 and D2
receptor properties and the direct and indirect pathways as originally
described in {{&lt; incite "Frank05}}. The version used here, for simple
demonstration purposes, uses the same core code as the PBWM algorithm
described in the [Executive Function
Chapter](/CCNBook/Executive "wikilink"), which was originally published
as . It does not include the more detailed aspects of disinhibitory
gating circuitry in the BG, which are explored in the published papers
and available elsewhere (see below).

The overall function of the model is to evaluate a given action being
considered in the frontal cortex, and decide whether to execute that
action on the basis of a learned history of reward / punishment outcomes
associated with the action. This history of reward / punishment is
learned by the model as you run it, based on the frequency of positive
vs. negative dopamine signals associated with each action option. For
each trial, if there is a dopamine burst (i.e., from a positive
outcome), then simulated dopamine D1 receptors in the direct pathway
cause the weights to increase for the active neurons in the MatrixGo
(direct) pathway of the striatum. Unlike glutamate or GABA, the effect
of dopamine is not directly excitatory or inhibitory, but rather depends
on the type of receptor and also how much excitatory input is already
present. Phasic bursts of dopamine excites those D1 neurons that are
already receiving excitatory synaptic input - and thus an increase in
dopamine preferentially amplifies activity in active Go neurons (i.e.,
those that selected the PFC action in response to the stimulus Input);
and this increase in activity is associated with LTP. So even though
there is no supervised 'target' for training the model what to do,
actions can get reinforced if they produce outcomes that are better than
expected (i.e. dopamine levels go up and the Go neuron activity
increases). In contrast to the Go units, dopamine inhibits NoGo neurons
due to its effects on D2 receptors, and hence these units undergo LTD
during rewards. This opposite effect also plays out in the case of
dopamine dips (i.e., from a worse-than-expected or even overtly negative
outcome). In this case, NoGo neurons receiving cortical input (i.e.
those representing the action that was selected) actually become more
active when dopamine levels go down (due to removal of the inhibitory D2
effect . This increase in NoGo activity for active neurons causes those
specific neurons that would suppress the selected action for this input
stimulus to increase their weights, which causes this action to be
*avoided* more in the future. This mechanism of "opponency" allows the
basal ganglia to learn and represent both the benefits (predicted reward
probability) and costs (predicted probability of negative outcomes)
separately, where the level of dopamine in the system can be used to
regulate the degree to which choices are made based on benefits or costs
(hence affecting risk taking, consistent with effects of dopamine
manipulations across species; for more details see {{&lt; incite
"CollinsFrank14" &gt;}}.

This will present one of the 6 input stimuli, which in turn activates a
corresponding action representation in the PFCout layer, representing
the superficial layers of PFC neurons (layers 2-3), in this case
premotor cortex rather than prefrontal cortex. And then as you continue
to cycle, you should see activation in the MatrixGo and MatrixNoGo
layers (for visualization convenience, all the connections are localist
to units in the same position as the input units). These layers
represent the *matrisome* (aka Matrix, which sounds cooler) medium spiny
neurons (MSN's) of the striatum, e.g., in the dorsal region, which are
interconnected with the frontal cortical action planning brain areas
(both at the level of premotor cortex but also more anteriorly for
abstract decisions). As you continue to cycle, activation spreads to the
GPeNoGo (globus pallidus external segment) and Thalamus layers. Note
that in the actual BG system (and in our more detailed models), Go units
don't directly excite the thalamus but instead they inhibit downstream
neurons in the globus pallidus internal segment (GPi); GPi neurons are
normally tonically active and send inhibitory projections to thalamus.
So instead of exciting thalamus, Go unit activity has a disinhibitory
function, enabling the Thalamus to get active by removing tonic
inhibition by GPi. In contrast, NoGo units have the opposite effect
(they inhibit GPe which in turn inhibits GPi, which itself inhibits
Thalamus...). This disinhibition circuitry through has its own set of
computational functions (i.e., there is a method to all this madness),
but for simplicity here, we abstract away from this circuitry, using one
common GPi_Thalamus layer that summarizes the functions of GPi and
Thalamus together. In this abstraction, Go units excite the
'GPi_Thalamus' which directly excites the PFC, whereas NoGo units
excite GPe which in turn inhibits GPi_Thalamus. This allows us to
represent the basic notion of opponency, whereby activity in the direct
Go pathway competes with the NoGo pathway for each action by affecting
the degree of thalamocortical excitability. You can explore more
detailed *emergent* models with disinhibitory dynamics, which allow one
to capture various physiological and behavioral data on Michael Frank's
website: <http://ski.clps.brown.edu/BG_Projects/Emergent_7.0+/> -- these
currently require the extra 7.1 "LTS" package available on the emergent
website: <https://grey.colorado.edu/emergent>

When a given GPi_Thalamus unit gets above a threshold level of
activation (0.5) (capturing the disinhibition that would be present in
the real system) to allow activation to flow from the PFCout layer to
the PFCout_deep layer, i.e., deep layer "output" neurons in this
frontal area. This is our current understanding of the net effect of BG
disinhibition of the thalamus: the thalamus is bidirectionally
interconnected with the deep layer PFC neurons, and disinhibiting it
allows these deep neurons to become active. These PFC deep layer neurons
then project to other areas in the frontal cortex and to other
subcortical targets -- for example frontal eye field (FEF) neurons
project to the superior colliculus and directly influence saccadic motor
actions, while primary motor cortex deep neurons project all the way
down to the spinal cord and drive muscle contraction patterns there.
Thus, this transition between superficial to deep activation, under
control of the BG disinhibition, is the neural correlate of deciding to
execute a motor action. As we discuss more in the [Executive Function
Chapter](/CCNBook/Executive "wikilink"), in most areas of frontal
cortex, this deep-layer activation has more indirect effects that
include maintaining a strong top-down activation signal on other areas
of cortex, which ultimately guide and shape behavior according to more
abstract action plans -- i.e., in most cases it is not as simple as
directly activating a set of muscles in response to a stimulus input!
Nevertheless, hierarchical extensions of the BG model, where multiple
cascading PFC-BG loops interact, have been used to simulate more
elaborated action selection processes in which the basic computational
function of the BG is similar at each level.

# Learning

Dopamine (DA) from the SNc modulates the relative balance of activity in
Go versus NoGo units via simulated D1 and D2 receptors. Dopamine effects
are greatest on those striatal units that are already activated by
corticostriatal glutamatergic input. Go units activated by the current
stimulus and motor response are further excited by D1 receptor
stimulation. In contrast, DA is uniformally inhibitory on NoGo units via
D2 receptors. This differential effect of DA on Go and NoGo units, via
D1 and D2 receptors, directly affects performance (i.e., more DA leads
to more Go and associated response vigor, faster reaction times) and,
critically, learning, as described above. Specifically, dopamine bursts
reinforce Go learning and weaken NoGo learning, while dips have the
opposite effects, and these make sense in terms of reinforcing actions
associated with positive outcomes, and avoiding those associated with
less-positive or negative ones. By integrating reinforcement history
over multiple trials this system can also learn which actions are
probabilistically more rewarding / punishing than others, allowing it to
select the best option among available alternatives.

, showing that Parkinson's Disease (PD) patients OFF their medications
were more likely to learn to avoid the B stimulus that was only rewarded
20% of the time, while PD patients ON their meds learned more to choose
the A stimulus that was rewarded 80% of the time. Age-matched control
Seniors were more balanced in learning across both cases. These results
make sense in terms of PD OFF having low dopamine and favoring D2
dopamine-dip based learning, while PD ON has elevated dopamine from the
meds that also "fill in the dips", producing a bias toward D1 burst
learning and away from D2 dip learning." &gt;}}

You can see that there 5 repetitions of each stimulus / action (labeled
a-f), and that each has a different proportion of trials receiving a
positive dopamine burst (yellow) vs a dip (cyan). This simulates a
simplified version of the *probabilistic selection task* {{&lt; cite
"FrankSeebergerOReilly04}}, where human participants were asked to
choose among different Japanese characters in a two-alternative
forced-choice task and had to learn which characters were
probabilistically more rewarding (e.g., for the A-B pair of characters,
A was rewarded 80% of the time while B was only rewarded 20%). Note that
participants (and models) can learn that A is the most rewarding, that B
is the least rewarding, or both -- you can't tell just by looking at
choices among A and B. After initial training on specific pairs of
stimuli, we then tested on all different paired combinations (e.g., A is
paired with other stimuli that have on average neutral 50% probability,
and B is paired with those same stimuli; thus any bias in Go vs NoGo
learning will show up as better performance approaching A or avoiding B
in these test trials). Critically, this allows one to see the difference
between a Go bias for rewarded stimuli vs. a NoGo bias for non-rewarded
ones. Empirically, we found that Parkinson's patients off of their
medications, who have reduced levels of dopamine, learned more NoGo than
Go, while those on their medications learned more Go than NoGo, while
age-matched controls were somewhere in between (). This basic pattern
has now been reported in various other experiments and tasks. We will
see that we can explain these results in our simple model.

You should have observed that the model learned a sensible action
valuation representation given the relative dopamine outcomes associated
with these actions, similar to the participants in the probabilistic
selection task, who were reliably able to choose the more consistently
rewarded stimuli over those that were less frequently rewarded. You
should also have noticed that while the matrix units encode a more
continuous representation of the reward probabilities, the net output of
the system reflected a threshold-like behavior that chooses any action
that is has more good than bad outcomes, while avoiding those with the
opposite profile.

# Simulating Parkinson's Disease and Dopamine Medications

.25 compare to that of the "intact" network from before, with
burst_da_gain{{=}}1, in the MatrixGo and NoGo pathways, and the
PFCOut_deep output layer? How does this correspond with the results
from PD patients OFF meds, as shown in ? Recall that the PFCOut_deep
layer reflects the net impact of the learning on action valuation, so
units that have high avg_act correspond to those that the system would
deem rewarding on average -- you should notice a difference in how
rewarding an action needs to be before the system will reliably select
it.}}

Next, we can simulate effects of DA medication given to PD patients --
for example levodopa increases the synthesis of dopamine. In addition to
increasing dopamine availability, medications also continually and
directly stimulate dopamine D2 receptors (so-called D2 agonists), which
has the effect of blunting the impact of any dips in dopamine (i.e.,
even when dopamine levels go down, the drugs will continue to occupy D2
receptors and prevent NoGo units from getting excited and learning).

One interesting side-effect of PD meds is that a subset of patients
develop a gambling habit as a result of taking these meds! This can be
explained in the model in terms of the shift in the balance between Go
vs. NoGo learning due to the meds -- all those failures to win count for
less, and the rare wins count for more.

Although this very simple model can account for the key qualitative
features of dopamine-based learning in the BG to promote adaptive action
selection, and also for these fascinating patterns of effects in PD
patients, there are a number of more complex issues that must be solved
to produce a more realistic model of the full complexities of the
decision making process that underlies complex behavior. To start,
decision making is not just a simple Go/NoGo decision on a single
action. The more elaborate models of the BG circuitry (available at
Michael Frank's website:
<http://ski.clps.brown.edu/BG_Projects/Emergent_7.0+/>) allow for it to
select among multiple actions (where action selection involves both Go
and NoGo activity for multiple actions in parallel), and explore i) the
differential roles of dopamine on learning vs. choice (i.e., risky
decision making), ii) the function of the subthalamic nucleus and the
'hyperdirect' pathway for preventing impulsive actions in response to
decision conflict, (iii) the role of cholinergic interneurons for
optimizing learning as a function of uncertainty, and (iv) hierarchical
interactions among multiple cortico-BG circuits for more advanced
learning and abstraction of hierarchical task rules during decision
making that supports generalization to new situations.

Beyond these functions, in the real world, the dopaminergic outcomes
associated with actions almost never come immediately after the action
is taken -- often multiple sequences of actions are required, with
outcomes arriving some minutes, hours or even later! In our more
complete PBWM (prefrontal-cortex basal-ganglia working memory) model
covered in the [Executive Function
Chapter](/CCNBook/Executive "wikilink"), we show how these same BG
dynamics and learning mechanisms can support maintenance and updating of
activation-based "working memory" representations in PFC, to bridge
longer gaps of time. Furthermore, properties of brain systems that drive
phasic dopamine firing, covered in the
[RL](/CCNBook/Sims/Motor/RL "wikilink") and
[PVLV](/CCNBook/Sims/Motor/PVLV "wikilink") models, help to transfer
phasic dopamine signals from firing at the time of later outcomes, to
earlier stimuli that reliably predict these later outcomes -- this is
helpful for driving action learning to achieve *sub-goals* or
*milestones* along the way toward a larger desired outcome. Furthermore,
we'll see that a synaptic tagging-based *trace* learning mechanism is
very effective in bridging these temporal gaps, and solves a number of
different problems that other mechanisms cannot. Another critical
element missing from this model is the ability to explicitly represent
the nature of the outcomes of different actions, and to reason about
these outcomes in relation to factors such as effort, difficulty and
uncertainty -- these capabilities require the functions of the
orbitofrontal cortex (OFC), anterior cingulate cortex (ACC), and other
ventral / medial PFC brain areas, all working in conjunction with these
basic BG and dopaminergic systems. Developing such models is at the
forefront of current research.
