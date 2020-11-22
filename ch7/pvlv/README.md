Back to [All Sims](https://github.com/CompCogNeuro/sims) (also for general info and executable downloads)

# Introduction

This simulation explores the PVLV (Primary Value, Learned Value) learning algorithm, which considers the role of different brain areas in controlling dopamine cell firing during learning about reward and punishment in classical conditioning tasks [Mollick et al, 2020](#references).  It represents a more flexible and biologically-detailed approach to the computations explored in the `rl_cond` model.

There are many brain areas involved in the phasic firing of dopaminecells in the VTA (ventral tegmental area) and SNc (substantia nigra,pars reticulata). The PVLV model integrates contributions from the most important of these areas within a coherent overall computationalframework including: 1) multiple sub-regions of the amygdala, an arealong implicated in affective processing of both positive and negativeemotion; 2) multiple pathways within the ventral striatum (VS, which includes the nucleus accumbens, NAc), alsoimportant in many aspects of emotional expression; and, 3) the lateralhabenula (LHb) pathway, recently identified as the substrate responsible for the inhibitory pausing (dipping) of dopamine neuron activity [Matsumoto & Hikosaka, 2007; Matsumoto & Hikosaka, 2009](#references).

The basic functions of the model can be seen in Pavlovian conditioning tasks, where neutral cues (conditioned stimuli; CSs) are paired with rewards or punishments (unconditioned stimuli; USs), resulting in the acquisition of conditioned responses (CRs), for example: the sound of a bell producing salivation in anticipation of a tasty food reward in Pavlov's famous dog, or, the onset of a light producing freezing before being shocked. Critically, phasic dopamine responses that initially occur for unexpected USs come to occur at the time of the CS instead. PVLV models the neurobiological mechanisms that cause this change in dopamine signaling to occur and proposes that this system can account for much of the behavioral manifestations of Pavlovian conditioning as well. Also important is the idea that subjects come to anticipate the imminent occurrence of *specific* USs after experiencing particular CSs, representing the expected US in the form of a working memory-like goal-state in the orbital frontal cortex (OFC). This distinguishes the PVLV framework from more abstract models that treat affective outcomes as merely good orbad.

# Overview of the PVLV Network

The overarching idea behind the PVLV model [OReilly et al, 2007](#references) is that there are two separate brain systems underlying two separate aspects of reward learning: the *Primary Value (PV)* and *Learned Value (LV)* systems. Specifically, the *ventral striatum* learns to expect US outcomes (PV learning) and causes the phasic dopamine signal to reflect the difference between this expectation and the actual US outcome value experienced. This difference is termed a *reward prediction error* or *RPE*. At the same time, the *amygdala* learns to associate CSs with US outcomes (rewards and punishments), thus acquiring new CS-value associations (LV learning). This division of labor is consistent with a considerable amount of data [Hazy et al, 2010](#references). The current model has a greatly elaborated representation of the amygdala and ventral striatal circuitry, including explicitly separate pathways for appetitive vs. aversive processing, as well as incorporating a central role for the *lateral habenula* (LHb) in driving pauses in dopamine cell firing (dipping) for worse than expected outcomes. Figure 1 provides a big-picture overview of the model.

![PV.1](fig_bvpvlv_pv_lv_only.png?raw=true "PV.1")

**Figure 1:** Simplified diagram of major components of the PVLV model, with the LV Learned Value component in the Amygdala and PV Primary Value component in the Ventral Striatum (principally the Nucleus Accumbens Core, NAc).  LHb: Lateral Habenula, RMTg: RostroMedial Tegmentum, PPTg: PendunculoPontine Tegmentum, LHA: Lateral Hypothalamus, PBN: Parabrachial Nucleus. See [PVLV Code](https://github.com/emer/leabra/tree/master/pvlv) for a more detailed figure and description of the implementation.

# Basic Appetitive Conditioning

We begin our exploration with basic appetitive conditioning, which is the simplest case of classical conditioning. A neutral stimulus, represented in the `Stim_In` layer, is paired with Reward (the US) in the `PosPV` layer. One CS (A; 1st unit in `Stim_In`), is paired with reward 100% of the time (e.g., a fixed amount of H2O), while the second CS (B; 2nd unit) is followed by the same quantity of reward only 50% of the time. Over learning, the layers in the PVLV network learn to modify dopamine cell firing levels in the `VTAp` and `VTAn` layers. The `VTAp` layer represents the typically responding dopamine cells in the VTA and SNc and is the main focus of the PVLV model; the `VTAn` represents a small subset of dopamine cells recently shown to respond to aversive USs with unequivocal bursting, but whose functional behavior so far remains poorly characterized and currently has no influence on learning in PVLV. During the simulation you will see that, early in training, a large dopamine burst occurs in the VTAp to the initially unexpected reward, but that with further training this response diminishes and the dopamine burst progressively moves to the time of the CS as the LV (amygdala) system learns the CS-US contingency.

**Tip:** This simulation uses a flexible stepping mechanism that allows the user to set both the `Grain` of the step size (e.g., `Cycle`, `Quarter`, etc.) as well as the `N` number of steps to be taken at a time. For example, by setting the `Grain` to `Trial` and `N` 10, each click will run 10 trials and then stop.

* To begin, confirm that `PosAcq` is selected in the leftmost button in the task bar (if something else is displayed click on the button and select PosAcq from the dropdown menu).  Also in the task bar find the `StepGrain` button and confirm it is set to `AlphaFull` and `StepN` is set to `1`.  Confirm that the `NetView` tab is active and the network is visible in the visualizer panel. Click `Init` and then `StepRun` once to step one alpha cycle (= 100 cycles). Now click `StepRun` a second time and you should see the `Stim_In` and `Context_In` input layers become active (t1 timestep). Click `StepRun` two more times to get to the t3 timestep.

You should now see the first unit of the `PosPV` layer get active (assuming an A trial; PosPV may not be on if a B trial), representing a specific kind of reward being given to the network. Also, note which `USTime_In` unit is active along the horizontal (time) dimension when the reward is presented. Each USTime horizontal row encodes a temporally evolving representation hypothesized to be in orbitofrontal cortex (OFC), which allows the network to learn about specific temporal contingencies between CSs and and individual rewards or punishments. Overall, you can think of this scheme as just a more complex version of the complete serial compound (CSC)) temporal representation used in the`rl_cond` model.

* Click `StepRun` several more times to step through enough A (100% reward) and B (50%) trials to get a good understanding of how their inputs differ. 

In particular, you might observe the active `USTime_In` unit "jump" to the bottom/foreground row on the timestep following an actual reward delivery. You can think of this as a kind of resetting of a stopwatch in anticipation of the *next* occurrence of that US reflecting the idea that USs themselves can be predictors of subsequent US events -- after all, where there are some berries there are likely to be more!

* Switch to viewing the `TrialTypeData` tab in the visualizer, and then click `Run` (not `StepRun` this time) in the task bar to complete the training run.

You will see the activity plotted for several key layers (different brain areas). Three trial types are shown together, updating after each block of trials as learning proceeds. The trial with CS A predicting a 100% probability of reward is shown on the left. For the case in which CS B is rewarded only 50% of the time there are two trial types: 1) when reward is omitted; and, 2) when reward is delivered.

As the network trains, pay attention first to the CS A (100%) trial on the left, noting especially `VTAp` activity (solid black line). You should see a big peak (phasic dopamine) initially when reward (US) is delivered (A_Rf_POS_t3; CS A = positive reinforcement, time step 3). Over time, this peak decreases, as the activity in `VSPatchPosD1` (solid green line) increases. This is the ventral striatum Primary Value (PV) substrate learning to expect the US, and it sends shunt-like inhibition to the VTA, mitigating bursting. This basic dynamic reflects the canonical Rescorla-Wagner delta learning rule as discussed in the main chapter.

You should also note that `VTAp` activity progressively increases at the`A_Rf_POS_t1` timestep, which is when the A stimulus (CS) is turned on. Note that `CEl_Acq_Pos_D1` activity (part of amygdala; solid red line) also increases at that time step -- this is what drives `VTAp` bursting at CS onset and reflects Learned Value (LV) learning in the amygdala associating the CS with the US. This learning is facilitated by the phasic dopamine signal at the time of the US, and as that diminishes due to PV learning in the ventral striatum, so does learning in the LV pathway.

Thus, the two basic aspects of phasic dopamine firing, which result from the single TD equation in the `rl_cond` model, actually emerge from two brain systems working in concert. These are the key pathways for positive-valence acquisition. It is worth noting that the "D1" in the name of some layers reflects the fact that the D1 class of post-synaptic dopamine receptors respond to increased dopamine by strengthening glutamatergic synapses that happen to have been active just prior.

Next, let's take a closer look at the case of 50% reward.

* If the network has not yet stopped running on its own, click `Stop` in the tool bar.  With `TrialTypeData` still displayed `Init` the network again and then `Run` so can watch the 50% trials this time.

Focus this time on the two trials on the right and watch the progression of `VTAp` activity over time for the two trial types. For both you should see that VTAp activity starts to increase at the t1 timestep before you see any dipping in one of the trials at t3. This is because it takes awhile to develop an expectation of reward to drive the dips. This is also why the CS-onset VTAp activity for B trials initially mirrors that for A trials even though it gets rewarded only half the time, reflecting only the magnitude of reward initially. Watch both trial types as training proceeds and note how the signaling comes to balance out, reflecting the expected value of reward. VTAp activity driven by the onset of the CS B settles at around 0.5, or half of that for CS A. Likewise, the delivery of reward at timestep t3 produces VTAp activity of ~0.5 while reward omission produces a dip of -0.5, both reflecting an expected value of 0.5 due to 50% reward probability.

* `Stop` again if need be and then switch to viewing `NetView` so we can examine some of the weights that have been learned. Click `r.Wt` in the vertical ribbon along the left border and then, in the network itself, click on the first unit (of four) in the `VSPatchPosD1` layer toward the lower right of the display, just above `LHbRMTg`.

> **Question 7.7:** Which units from the `USTime_In` layer does the `VSPatchPosD1` receive weights from, and are these the same units that were active when the reward was presented? How do these weights (from USTime_In to VSPatchPosD1) allow the network to mitigate the dopamine burst at the time of an expected reward?

* When done change back to displaying the `Act` variable in the `NetView` display.

# Extinction

In extinction learning, a CS that was previously rewarded is subsequently paired with no reward. A critical idea is that extinction is not simply the *unlearning* of the previous association between CS and US, but is instead a kind of second level of learning superimposed on the first -- with the original learning largely preserved. A second key idea, related to the first, is that extinction learning is particularly influenced by *context* -- in many cases, the reason an expected outcome does not occur can be attributed to other factors -- including the broader setting in which the omission of reward or punishment is now occurring, i.e., the context. Learning about such contextual contingencies is important for modulating expectations appropriately.

In the next simulation, we will again pair CSs A and B with reward as before, but then follow that with training in which rewards are always withheld. After that, we'll explore a simulation that specifically explores the differential role of context in extinction learning.

* In the task bar select `PosExt` and then `Init`. Then, change the `StepGrain` parameter to `Block`.  As the name implies this changes the step size to run a full block of trials each time.  In addition, to get to the end of the acquisition phase we need to change the number of steps to run for each click. Click `StepN` and type '50' followed by `Enter`.  Click `StepRun` to get the acquisition phase going and then select the `TrialTypeData` tab in the visualizer. 

You should see the same three trial types as before: a single CS A trial that is always rewarded; and, two CS B trials, one rewarded and one not. We now want to edit the `TrialTypeData` display so we can follow the activity of some additional layers during the extinction phase, which we can do even while the network is running.

* In the `TrialTypeData` display click ON the check box next to `LHbRMTg_act` to display the activity for that layer.

You should see a solid blue line appear in the graph. This displays `LHbRMTg` activity as training proceeds, corresponding to the function of the lateral habenula primarily (LHb = lateral habenula; RMTg = rostromedial tegmentum, an intermediary between the LHb and VTA). Late in training note how its activity at the t3 timestep has come to reflect the delivery (downward deflection) or omission (upward) of reward more-or-less symmetrically for the two types of B trials. In contrast, `LHbRMTg` activity comes to approach baseline for the A trials since there are never any negative outcomes (i.e., omitted rewards) and the initially large negative responses to reward delivery are systematically predicted away. Now let's see what happens during extinction.

The extinction phase also goes for 50 epochs, but to start we want to watch the network timestep-by-timestep early in extinction to understand what the network is doing. After that, we'll switch to the `TrialTypeData` tab again to watch basic extinction play out.

* With `TrialTypeData` displayed change `StepN` to 1 (`Enter`). Then click `StepRun` once to run one block and advance to the extinction phase.

**Tip:** If the display doesn't change from three to two trial types just click `StepRun` again until it does since it may have an extra block or so to transition to the extinction phase.

You should see the `TrialTypeData` display change to reflect the fact that there are only two types of trials now (A omit, B omit).

* Once the `TrialTypeData` display has transitioned select the `NetView` tab so we can watch the network timestep-by-timestep early in extinction training. In the task bar change `StepGrain` back to `AlphaFull` to step one timestep at a time.  Click `StepRun` once and check that the trial name ends in "_t0" in the field at the bottom of the NetView  display. If not, `StepRun` one timestep at a time until it does.  Trial type (A or B) does not matter. `StepRun` one more time and should see the `Stim_In` and `Context_In` input layers become active on the t1 timestep. Click `StepRun` two more times and then make sure the trial name ens with "_t3".   

Note that now the `PosPV` layer is not active for either trial type. Also note that the `VTAp` is significantly negative (blue) registering the omission of expected reward, while the `LHbRMTg` layer next to it is significantly positive (red-yellow). This reflects the fact that the lateral habenula has been shown to drive pauses in dopamine cell firing in response to the omission of an expected reward.

* Keep clicking `StepRun` until you've followed several examples of both A and B type trials.

You should be able to tell that the phasic dopamine dips (`VTAp` activity on t3 time steps) are weaker (lighter blue) for B than for A trials, reflecting the different expected values for the two trial types. Note which `Context_In` units are on for the two trial types, especially focusing on the A trial (1st unit, 1st row), the same units active during the acquisition phase. Later on we'll see what happens when different context units are activated during extinction relative to acquisition.

* Switch back to the `TrialTypeData` tab in the visualizer.  Click `Run` to observe the changing layer activities as extinction training proceeds to completion.

While observing the `TrialTypeData` graph as extinction proceeds note that `VTAp` activity (black) at the time of the omitted reward gradually becomes less negative and eventually returns to the zero baseline for both trial types. This is because `LHbRMTg` activity (blue) itself returns to its baseline as well. In parallel, note also that the positive VTAp activity at the time of CS-onset progressively decreases, even becoming negative. This reflects the underlying neurobiology in which it has been found that some dopamine cells acquire pausing after extinction training; others retain some bursting; and, still others exhibit a biphasic burst-then-pause pattern of firing. In the PVLV model the negative dopamine signal at CS-onset is driven by positive activity in the LHbRMTg layer, which in turn is driven by learning in the `VSMatrixPosD2` layer.

* Click the `VSMatrixPosD1_act` and `VSMatrixPosD2_act` check boxes ON.

You should see two new lines come on in the `TrialTypeData` graph: dark blue = `VSMatrixPosD1_act`; turquoise = `VSMatrixPosD2_act`. Note the greater activity in the VSMatrixPosD2 relative to VSMatrixPosD1 -- this is what is responsible for the positive LHbRMtg activity (blue) driving the net negative dopamine signal. Note also that CElAcqPosD1 activity (red) remains positive for both trial types meaning that there is still some positive drive to dopamine cells as well, consistent with the empirical data showing that bursting persists in some dopamine cells after extinction in addition to those showing pausing, often as a bi-phasic burst-then-pause pattern. Thus, although PVLV doesn't have the temporal resolution to display a bi-phasic response it does exhibit behavior reflecting the substrates capable of producing all three patterns of dopamine response.

* After extinction training is complete, click on the `NetView` tab and click on `r.Wt` in the vertical ribbon along the left border so we can look at the strength of individual receiving weights.  Click around on several units in the first (leftmost) unit pool in the `BLAmgPosD1` layer, and the first unit in `CElAcqPosD1`, taking note of which sending units display significant weights.

> **Question 7.8:** Why do you think these units still have strong weights from `Stim_In`? How might this explain the idea that the original learning during acquisition is not completely erased after extinction? How might conditioned responses be extinguished (not expressed) if these weights are still strong? Hint: `BLAmygPosD2` activity inhibits `BLAmygPosD1` activity.

* When you're done change back to displaying the `Act` variable in the `NetView` display.

## Renewal: The special role of context in extinction

An important upshot of the conditioning literature is that extinction learning is not simply the erasure of acquisition; there are several circumstances under which extinguished behaviors can be recovered. For example, in *spontaneous recovery*, conditioned responses that have been fully extinguished by the end of a session will typically reappear when the subject is re-tested the following day, albeit in weaker than original form. Further extinction training is typically followed by spontaneous recovery as well, although the recovery is progressively weaker with each extinction/recovery/extinction cycle. Similarly, even after several extinction/recovery/extinction training cycles in which virtually no sign of spontaneous recovery remains, subsequent exposure to the original US (but no CS) can bring about the re-emergence of the extinguished behavior in response to a subsequent exposure to the original CS, often very robustly. This US-triggered effect is known as *reinstatement* and it goes to show that even after extensive extinction training a significant trace of the original CS-US pairing remains.

In addition to spontaneous recovery and reinstatement, a third extinction-related phenomenon called *renewal* has proven particularly seminal in deepening our understanding of extinction learning by highlighting the special role played by context in extinction learning [Bouton, 2004](#references). Briefly, if you do extinction in a different context (B) from the original acquisition context (A), and then switch back to the original context A to perform a test with the CS, you see that the just-extinguished conditioned response is now vigorously expressed. This pattern, known as *ABA renewal*, suggests that the context is modulating whether extinction is expressed or not. But, why do we say that the context is particularly important for the expression of *extinction* instead of for the expression of the original acquisition? The answer comes from experiments using a variation of the renewal paradigm called ABC renewal.

What if post-extinction testing were to be performed not back in the original acquisition context, but in a wholly different, third context (C)? Which learning -- original acquisition or subsequent extinction -- will win out? That is, will the original conditioned response be expressed or not? Since the extinction learning is more recent it might seem reasonable to expect that perhaps it will win out. It turns out, however, that when exposed to the original CS in a third, novel context the original conditioned response is vigorously expressed. This indicates that the context is modulating the expression of extinction more than it is modulating the expression of original acquisition. Even more compelling is the case of so-called AAB renewal in which acquisition and extinction are carried out in the *same* context (A) but then testing is when a new context (B) is introduced. It turns out that conditioned responses are significantly expressed in the novel context B indicating that context was relatively less important during the original acquisition phase, but became critically important during the extinction learning phase.

Recent empirical findings have specifically implicated the basolateral amygdalar complex (BLA) in context-dependent extinction learning. Briefly, there are two populations of neurons in the basolateral amygdala, some that increase their activity as associations are learned (acquisition neurons), and another population (extinction neurons), that increase their activity in response to extinction training [Herry et al, 2008](#references). Critically, these researchers also found that the extinction neurons are preferentially innervated by contextual inputs from the medial PFC. These results are captured in the PVLV model in the form of distinct BLAmygPosD1 (acquisition) and BLAmygPosD2 (extinction) layers.

In the following simulation we will explore ABA renewal to illustrate how context information may be integrated into the overall framework in order to perform these kinds of fine-grained discriminations. PVLV reproduces both ABC and AAB renewal straightforwardly, but we won't simulate these since the principles involved are identical. This time we will only be training with CS A (100% rewarded) trials.

* In the task bar select `AbaRenewal` and then `Init` the network.  Set/confirm the `StepGrain` to `Block` and change `StepN` to 25 (`Enter`). Click `StepRun` to get the initial acquisition phase going.

In the `NetView` display watch the acquisition training for awhile, noting especially which `Context_In` unit is active.

* Switch to the `TrialTypeEpochFirst` tab so we can observe the LV and PV learning curves play out in tandem.

The `TrialTypeBlockFirst` graph tracks phasic dopamine signaling (VTAp activity) separately for each timestep as it evolves over training. The two most relevant time steps are of course t1 (CS-onset = purple line) and t3 (US-onset = dark red). Note how both curves asymptote in opposite directions to reflect LV learning (t1) and PV learning (t3). The network will stop after 25 blocks which is right before the transition to extinction training.

* Once the network stops after 25 blocks, switch back to the `NetView` tab to prepare to watch the network as it transitions to the extinction phase. Click `Run` to complete the extinction phase.  

**Tip:** If you wish you can set `StepN` to '1' and click `StepRun` a few times first to make it easier to watch the transition.

Very quickly you should see a transition in the activity of different `Context_In` units in the layer between the acquisition and extinction phases. And, of course, the `PosPV` layer never becomes active again after the transition. These are the only changes to the inputs of the network.

* Switch back to `TrialTypeBlockFirst` to watch the evolution of the CS-onset (light purple) and US-onset (beige) dopamine signals as extinction proceeds.  After extinction training is complete, two renewal test trials are run that expose the network to the CS twice -- once in context A and once in context B.  Since these are uninterpretable in the TrialTypeblockFirst graph go back to the `TrialTypeData` tab.

In the `TrialTypeData` graph note the stark contrast in the CS-onset dopamine signals (`VTAp` activity; black line; timestep t1) when the CS is presented in context A (left trial) versus context B. The network has reproduced a version of the ABA renewal effect highlighting the context-specificity of extinction learning.

> **Question 7.9:** From an evolutionary perspective, why would a separate extinction mechanism be preferable to an erasure-type mechanism of the original learning? Relate your answer to the special sensitivity of extinction learning to context.

# Aversive Conditioning

For the final PVLV simulation we will look at how the same basic mechanisms involved in appetitive conditioning can support aversive conditioning as well -- that is, learning in the context of negative primary outcomes like pain, shock, nausea, and so on. Phasic dopamine signaling in aversive conditioning can be thought of as a kind of mirror-image of appetitive conditioning, but with some important anomalies that reflect basic differences in the ecological contingencies that pertain under threat. Chief among these is the obvious difference in the stakes involved during any single event: while failure to obtain a reward may be disappointing, there will generally be more opportunities.  On the other hand, failure to avoid a predator means there literally will be no tomorrow. Thus, threats must have a kind of systematic priority over opportunities.

This simulation will pair one CS (D) with a negative US 100% of the time, and another (E) only 50% of the time.

* Select `NegAcq` in the task bar and then `Init` the network.  Set the `StepGrain` to `AlphaFull` and set `StepN` to 1.  With the `NetView` visible click `StepRun` once and check the trial name in the field at the top right. If it is an E instead of D trial keep clicking until you get the t0 timestep of a D trial (trial name: D_Rf_NEG_t0). Now click `StepRun` once more to activate the `Stim_In` and `Context_In` layers, noting which units become active. Now, watching the `USTime_In` layer, click `StepRun` two more times to get to the t3 timestep.

You should have observed that `USTime_In` unit activity advancing timestep-by-timestep, just as we saw for the appetitive case (although with different units). Note that the network is receiving a punishment in the `NegPV` layer on the t3 timestep. Also note that `VTAp` activity is negative (blue) when punishment is delivered and `LHbRMTg` is positive (red-yellow), reflecting the fact that the latter is responsible for driving the former [Matsumoto & Hikosaka, 2007](#references). Finally, find the `VTAn` layer to the right of LHbRMTg and note that it also has positive (red-yellow) activity. VTAn represents a small minority of dopamine cells shown to respond to aversive outcomes with unequivocal bursting.

* Switch to the `TrialTypeData` tab and click `Run` to watch the evolution of network activity as training proceeds.

You should observe large dopamine dips (`VTAp`; black line) initially to the negative US for both D (left) and E trial types, which gradually decreases over time as the network learns. Corresponding to the key substrate responsible for PV learning in the appetitive case (VSPatchPosD1), the corresponding `VSPatchNegD2` units are learning to anticipate the punishment US so as to mitigate the LHbRMTg response to it, and thus the dopamine dips. Note, however, that even for the 100% punishment (D) trials the US-onset dopamine signal is never completely predicted away by the end of training. This reflects the empirical finding that dopamine responses to aversive primary outcomes appear not to completely go away even when fully expected [Matsumoto & Hikosaka, 2009](#references). This idea is implemented in PVLV by a gain factor (< 1) applied to the predictive inputs from VSPatchNegD2. The effect of this gain factor also shows up as an asymmetry in the 50% punishment (E) trials: note how the dip for punishment delivery remains proportionally greater than the burst for punishment omission, even after extensive training.

In parallel, the network is also acquiring dopamine dips in response to both CSs, along with a corresponding increase in `LHbRMTg` activity (blue). Note how the acquired dopamine dip is greater for the D (100% punishment) CS than the E (50%) CS, consistent with electrophysiological data showing that habenula activity (and thus dopamine cell pausing) scales with increased probability of punishment, effectively approximating expected value [Matsumoto & Hikosaka, 2009](#references). Now let's look a little deeper into what is going on with the network to produce these results.

* With the `TrialTypeData` graph still displayed, uncheck the displays for the VSPatchPosD1 and VSPatchPosD2 layers. It is worth noting in passing that these layers' activity levels are nil anyway since they were not involved in the processing of the negative primary outcomes.  Likewise, click OFF the VSMatrixPosD1 and VSMatrixPosD2 layers. Now, click ON the check boxes for `VSPatchNegD2_act`, `VSPatchNegD1_act`, `VSMatrixNegD2_act`, and `VSMatrixNegD1_act`

**Tip:** If the display doesn't update, click into the `TrialTypeData` display itself and hit the F5 function key to update the display.

First, take note of the strong `VSPatchNegD2` activity (brown-red line) at the t3 timesteps. This is what mitigates `LHbRMT` responses to the negative US, and thus the amount of negative activity in `VTAp`. Next, note the activity level for the VSMatrixNegD2 layer (beige line). In explicit contrast to the appetitive case, the acquired response to CS-onset is not being driven by the amygdala, but is instead driven by the acquired activity in this layer via the LHbRMTg. Nonetheless, it is important to understand that the amygdala is critically involved in many aspects of aversive conditioning (e.g., see strong `CElAcqNegD2` activity; red), even if it does not directly drive dopamine signaling. Finally, note how the activity level in the `VTAn` layer (pink) is the exact mirror-image of VTAp, both of which are being driven by LHbRMTg activity in the model.

> **Question 7.9a:** From an evolutionary perspective why would separate pathways for learning about aversive vs. appetitive primary outcomes be preferable to a single system for both?  Conversely, in terms of dopamine signaling, how might the positive responses to primary aversive outcomes in the `VTAn` layer be problematic if those signals were to be conveyed to downstream units that also receive signals from the `VTAp`?


------------------------------------------------------------------------

# (Optional) Advanced Explorations

Now that you have explored some of the basics of Pavlovian conditioning, this optional section has some more advanced explorations for those with a more in-depth interest in this area. These are only a small sample of the many capabilities of the PVLV model.

## Conditioned Inhibition

Conditioned Inhibition is an interesting and understudied phenomenon that focuses on negative prediction errors (which occur when there is less reward than expected), that are associated with the presence of a stimulus that reliably indicates when these reductions in reward occur (the *conditioned inhibitor*). This omission of an expected reward has been found to cause a dopamine dip (the same signal that occurs for negative stimuli). Conditioned inhibition occurs when a CS that has been associated with reward, is presented simultaneously with the inhibitor CS, along with an omission or reduction in reward. This causes a dopamine dip at the time a reward was usually presented, which trains a negative association for the inhibitor. Critically, after many trials of conditioned inhibition, the presentation of the inhibitor by itself causes a dopamine dip [Tobler et al, 2003](#references). One intuitive example of this is going to a soda machine and seeing an "OUT OF ORDER" sign, which means that you won't get soda. Since the "OUT OF ORDER" sign means you won't get soda that you usually expect from the soda machine, you form a negative association for it and may be disappointed the next time you see it on a soda machine. In this simulation, we will get into the mechanisms that allow the brain to learn a dopamine dip for the conditioned inhibitor.

This conditioned inhibition phenomenon is particularly interesting and challenging for models of conditioning because it takes a previously neutral stimulus and turns it into a negative-valence stimulus *without ever presenting any overt negative outcomes!* The fact that the inhibitor behaves like a CS that was associated with an overt negative outcome (e.g., pain) means that the dopamine dip associated with disappointment is by itself fundamentally capable of driving these negative learning pathways.

In our model of conditioned inhibition, we are going to take the previously trained CS (A), and pair it with a *conditioned inhibitor* (X), that always predicts the omission of reward.

* Select `PosCondInhib` in the task bar and set `StepGrain` to `Block`. Click `Init` and then set the `StepN:` to '25'.  Click `StepRun` once to run the initial acquisition phase.  While that is running make sure `NetView` is displayed.  Once it stops set the `StepN` back to '1' and then start clicking `StepRun` -- you should start seeing trials with two units active in the `Stim_In` layer. You may need to click `StepRun` up to several times to start seeing the two active units.  These two active units represent the conditioned stimulus (A) and the conditioned inhibitor (X). Now change `StepGrain` to `AlphaFull` and then click `StepRun` one timestep at a time until you see the network is on an AX trial (two Stim_In units active) and the `USTime_In` layer has two units on at the second position (timestep t3).   

Note that the `LHbRMTg` layer has positive activity (red-yellow), while the `VTAp` is negative (blue).

* Now click on `r.DWt` at the left of the `NetView` display.  Click on the first `VSMatrixPosD2` unit. 

You should see that the `Stim_In` units representing the A and the X stimuli are highlighted, representing a positive weight change from those units to `VSMatrixPosD2`. The dopamine dip has caused potentiation of the weights from those stimuli to D2 units, reflecting the biological finding that corticostriatal synapses onto D2 MSNs are *strengthened* by dopamine decreases - see [Gerfen & Surmeier, 2011](#references). This can be interpreted as representing an association of those stimuli with reward omissions.

* Switch back to viewing the `TrialTypeData` tab.  Click on `Run` to watch learning proceed as conditioned inhibition training finishes.

Note how the negative `VTAp_act` (black) and positive `LHbRMTg_act` (blue) activities gradually reduce over time, as the omission of reward predicted by the X conditioned inhibitor itself becomes expected.

**Tip:** You may want to switch back and forth with the `NetView` tab to watch the activity of the layers as stimuli are presented. If so, switch back to `TrialTypeData` to continue.

At the end of conditioned inhibition training three test trials are run: A alone, X alone, and AX. (Reward is never presented in any case). Note that the network shows a dopamine dip to the conditioned inhibitor (X) meaning that it has acquired negative valence, in accordance with the [Tobler et al., 2003](#references) data. This is caused by activity in the `LHbRMTg`, which reflects activity of the `VSMatrixPosD2` that has learned an association of the X conditioned inhibitor with reward omission. See [PVLV Code](https://github.com/emer/leabra/tree/master/pvlv) if you wish to learn more about the computations of the various ventral striatum and amygdala layers in the network.

> **Optional Question** Why does the network continue to show a partial dopamine burst to the A stimulus when it is presented alone? Hint: You may want to watch the network run again and note the different trial types. What is the purpose of interleaving A_Rf trials with the AX trials?

## Blocking

A crucial area of research on learning in general, particularly the dopamine system, is the blocking effect [(Waelti et al, 2001)](#references). In a blocking experiment, you take a CS (A) that has been previously trained with a reward association, and in a subsequent training session present it with another to-be-blocked CS (B), again followed by the same amount of reward. Since the A CS was fully trained on the pairing with reward, it predicts away the dopamine burst to the US.

However, if there is no US dopamine left, then it can't be used for learning about the other CS (B) even though it is being paired with reward. This is called "blocking" because the learning to that second CS is blocked by the CS (A) already having a full prediction of the reward. Interestingly, if you change the size or type of the reward that is given, then you can learn about the second CS (this is called "unblocking") [(McDannald et al, 2011)](#references).

In PVLV, we use the `VSPatchPosD1` layer to control the dopamine for a US (you'll remember that it receives a timing signal from the `USTime_In`). When you present the already learned A CS, these `VSPatchPosD1` weights have learned to fully block US dopamine, so there is no burst to the reward.

* Select `PosBlocking` in the task bar and then in `TrialTypeData` click OFF all the layer displays except `VTAp_act`. Then click `Init` and `Run` the network watching for when the initial A+ training transitions to the blocking training phase (AB+). 

After the preliminary acquisition phase and the AB+ blocking training phase has started you should note that the dopamine for A and AB start out exactly the same and undergo virtually no change during training, indicating that there is no difference between prediction and expectation for both trial types. At the end of AB+ training a test trial is run in which the network is presented the blocked CS (B) alone. Note that `VTAp` activity is essentially a flat line, indicating that there has been little dopamine signaling acquired by the blocked CS, consistent withthe Waelti et al., 2001 data.

## Safety signal learning: negative conditioned inhibition

When you think of negative valence learning, it is interesting that something that predicts a punishment will NOT occur often acquires positive associations. Think of that warm and fuzzy feeling you get at home in the wintertime, sitting in front of a warm fireplace that keeps away all the cold snow. These signs that a punishment will not occur are called "safety signals", and some data has shown that dopamine neurons respond with a burst to the offset of a punishment (Brischoux et al, 2009). We draw on this data to show that the dopamine bursts you get for the omission of a punishment can train up positive associations for these safety signals. In our simulation, we are going to take the negatively trained CS (D), and pair it with another stimulus (U), that predicts the omission of the punishment.

* Select `NegCondInhib` in the task bar and in `TrialTypeData` turn OFF display for all the layers except `VTAp`.  Click `Init` and `Run` to run the full simulation. 

The full simulation goes through three sequential phases: *aversive acquisition* -> *safety signal training* ->  *test*. Once the network has stopped you will see three test trials displayed: DU; D alone; U alone. Note the dopamine burst to the U CS that predicts the omission of a punishment, meaning that it has acquired positive valence. You may remember that we used the `VSMatrixPosD2` pathway to learn about conditioned inhibitors in the appetitive case. Here in the safety signal case, learning in the corresponding `VSMatrixNegD1` pathway produce an analogous, opposite effect.

# References

* Bouton, M. E. (2004). Context and behavioral processes in extinction. Learning & Memory, 11(5), 485–494. http://dx.doi.org/10.1101/lm.78804

* Brischoux, F., Chakraborty, S., Brierley, D. I., & Ungless, M. A. (2009). Phasic excitation of dopamine neurons in ventral {VTA} by noxious stimuli. Proceedings of the National Academy of Sciences USA, 106(12), 4894–4899. http://www.ncbi.nlm.nih.gov/pubmed/19261850

* Gerfen, C. R., & Surmeier, D. J. (2011). Modulation of striatal projection systems by dopamine. Annual Review of Neuroscience, 34, 441–466. http://www.ncbi.nlm.nih.gov/pubmed/21469956

* Hazy, T. E., Frank, M. J., & O’Reilly, R. C. (2010). Neural mechanisms of acquired phasic dopamine responses in learning. Neuroscience and Biobehavioral Reviews, 34(5), 701–720. http://www.ncbi.nlm.nih.gov/pubmed/19944716

* Herry, C., Ciocchi, S., Senn, V., Demmou, L., Müller, C., & Lüthi, A. (2008). Switching on and off fear by distinct neuronal circuits. Nature, 454(7204), 1–7. http://www.ncbi.nlm.nih.gov/pubmed/18615015

* Matsumoto, M., & Hikosaka, O. (2007). Lateral habenula as a source of negative reward signals in dopamine neurons. Nature, 447, 1111–1115. http://www.ncbi.nlm.nih.gov/pubmed/17522629

* Matsumoto, O., & Hikosaka, M. (2009). Representation of negative motivational value in the primate lateral habenula. Nature Neuroscience, 12(1), 77–84. http://www.citeulike.org/user/nishiokov/article/3823302

* McDannald, M. A., Lucantonio, F., Burke, K. A., Niv, Y., & Schoenbaum, G. (2011). Ventral striatum and orbitofrontal cortex are both required for model-based, but not model-free, reinforcement learning. The Journal of Neuroscience, 31(7), 2700–2705. https://doi.org/10.1523/JNEUROSCI.5499-10.2011

* Mollick, J. A., Hazy, T. E., Krueger, K. A., Nair, A., Mackie, P., Herd, S. A., & O’Reilly, R. C. (2020). A systems-neuroscience model of phasic dopamine. Psychological Review, Advance online publication. https://doi.org/10.1037/rev0000199

* O’Reilly, R. C., Frank, M. J., Hazy, T. E., & Watz, B. (2007). PVLV: The primary value and learned value Pavlovian learning algorithm. Behavioral Neuroscience, 121(1), 31–49. http://www.ncbi.nlm.nih.gov/pubmed/17324049

* Tobler, P. N., Dickinson, A., & Schultz, W. (2003). Coding of predicted reward omission by dopamine neurons in a conditioned inhibition paradigm. Journal of Neuroscience, 23, 10402–10410. http://www.ncbi.nlm.nih.gov/pubmed/14614099

* Waelti, P., Dickinson, A., & Schultz, W. (2001). Dopamine responses comply with basic assumptions of formal learning theory. Nature, 412, 43–48. http://www.ncbi.nlm.nih.gov/pubmed/11452299

