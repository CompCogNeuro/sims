# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from leabra import go, pygiv, env, rand, erand, etensor
from enum import Enum
import os


class Actions(Enum):
    """
    Actions are SIR actions
    """

    Store = 0
    Ignore = 1
    Recall = 2
    ActionsN = 3


class SIREnv(pyviews.ClassViewObj):
    """
    SIREnv implements the store-ignore-recall task
    """

    def __init__(self):
        super(SIREnv, self).__init__()
        self.Nm = str()
        self.SetTags("Nm", 'desc:"name of this environment"')
        self.Dsc = str()
        self.SetTags("Dsc", 'desc:"description of this environment"')
        self.NStim = int(4)
        self.SetTags(
            "NStim", 'desc:"number of different stimuli that can be maintained"'
        )
        self.RewVal = float(1)
        self.SetTags(
            "RewVal", 'desc:"value for reward, based on whether model output = target"'
        )
        self.NoRewVal = float(0)
        self.SetTags("NoRewVal", 'desc:"value for non-reward"')
        self.Act = Actions.Store
        self.SetTags("Act", 'desc:"current action"')
        self.Stim = int()
        self.SetTags("Stim", 'desc:"current stimulus"')
        self.Maint = int()
        self.SetTags("Maint", 'desc:"current stimulus being maintained"')
        self.Input = etensor.Float64()
        self.SetTags("Input", 'desc:"input pattern with stim"')
        self.CtrlInput = etensor.Float64()
        self.SetTags("CtrlInput", 'desc:"input pattern with action"')
        self.Output = etensor.Float64()
        self.SetTags("Output", 'desc:"output pattern of what to respond"')
        self.Reward = etensor.Float64()
        self.SetTags("Reward", 'desc:"reward value"')
        self.Run = env.Ctr()
        self.SetTags(
            "Run", 'view:"inline" desc:"current run of model as provided during Init"'
        )
        self.Epoch = env.Ctr()
        self.SetTags(
            "Epoch",
            'view:"inline" desc:"number of times through Seq.Max number of sequences"',
        )
        self.Trial = env.Ctr()
        self.SetTags(
            "Trial", 'view:"inline" desc:"trial is the step counter within epoch"'
        )

    def Name(ev):
        return ev.Nm

    def Desc(ev):
        return ev.Dsc

    def SetNStim(ev, n):
        """
        SetNStim initializes env for given number of stimuli, init states
        """
        ev.NStim = n
        ev.Input.SetShape(go.Slice_int([n]), go.nil, go.Slice_string(["N"]))
        ev.CtrlInput.SetShape(
            go.Slice_int([Actions.ActionsN.value]), go.nil, go.Slice_string(["N"])
        )
        ev.Output.SetShape(go.Slice_int([n]), go.nil, go.Slice_string(["N"]))
        ev.Reward.SetShape(go.Slice_int([1]), go.nil, go.Slice_string(["1"]))
        if ev.RewVal == 0:
            ev.RewVal = 1

    def Validate(ev):
        if ev.NStim <= 0:
            return fmt.Errorf(
                "SIREnv: %s NStim == 0 -- must set with SetNStim call", ev.Nm
            )
        return go.nil

    def State(ev, element):
        if element == "Input":
            return ev.Input
        elif element == "CtrlInput":
            return ev.CtrlInput
        elif element == "Output":
            return ev.Output
        elif element == "Reward":
            return ev.Reward
        return go.nil

    def StimStr(ev, stim):
        """
        StimStr returns a letter string rep of stim (A, B...)
        """
        return chr(ord("A") + stim)

    def String(ev):
        """
        String returns the current state as a string
        """
        return "%s_%s_mnt_%s_rew_%g" % (
            ev.Act,
            ev.StimStr(ev.Stim),
            ev.StimStr(ev.Maint),
            ev.Reward.Values[0],
        )

    def Init(ev, run):
        ev.Run.Scale = env.Run
        ev.Epoch.Scale = env.Epoch
        ev.Trial.Scale = env.Trial
        ev.Run.Init()
        ev.Epoch.Init()
        ev.Trial.Init()
        ev.Run.Cur = run
        ev.Trial.Cur = -1
        ev.Maint = -1

    def SetState(ev):
        """
        SetState sets the input, output states
        """
        ev.CtrlInput.SetZeros()
        ev.CtrlInput.Values[ev.Act.value] = 1
        ev.Input.SetZeros()
        if ev.Act != Actions.Recall:
            ev.Input.Values[ev.Stim] = 1
        ev.Output.SetZeros()
        ev.Output.Values[ev.Stim] = 1

    def SetReward(ev, netout):
        """
        SetReward sets reward based on network's output
        """
        cor = ev.Stim
        rw = netout == cor
        if rw:
            ev.Reward.Values[0] = float(ev.RewVal)
        else:
            ev.Reward.Values[0] = float(ev.NoRewVal)
        return rw

    def StepSIR(ev):
        """
        Step the SIR task
        """
        while True:
            ev.Act = Actions(rand.Intn(int(Actions.ActionsN.value)))
            if ev.Act == Actions.Store and ev.Maint >= 0:
                continue
            if ev.Act == Actions.Recall and ev.Maint < 0:
                continue
            break
        ev.Stim = rand.Intn(ev.NStim)
        if ev.Act == Actions.Store:
            ev.Maint = ev.Stim
        if ev.Act == Actions.Ignore:
            pass
        if ev.Act == Actions.Recall:
            ev.Stim = ev.Maint
            ev.Maint = -1
        ev.SetState()

    def Step(ev):
        ev.Epoch.Same()
        ev.StepSIR()
        if ev.Trial.Incr():
            ev.Epoch.Incr()
        return True

    def CounterCur(ev, scale):
        if scale == env.Run:
            return ev.Run.Cur
        if scale == env.Epoch:
            return ev.Epoch.Cur
        if scale == env.Trial:
            return ev.Trial.Cur
        return -1

    def CounterPrv(ev, scale):
        if scale == env.Run:
            return ev.Run.Prv
        if scale == env.Epoch:
            return ev.Epoch.Prv
        if scale == env.Trial:
            return ev.Trial.Prv
        return -1

    def CounterChg(ev, scale):
        if scale == env.Run:
            return ev.Run.Chg
        if scale == env.Epoch:
            return ev.Epoch.Chg
        if scale == env.Trial:
            return ev.Trial.Chg
        return -1
