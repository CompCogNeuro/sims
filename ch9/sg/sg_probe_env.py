# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from leabra import go, pygiv, env, rand, erand, etensor, esg

import os

class ProbeEnv(pygiv.ClassViewObj):
    """
    ProbeEnv generates sentences using a grammar that is parsed from a
    text file.  The core of the grammar is rules with various items
    chosen at random during generation -- these items can be
    more rules terminal tokens.
    """

    def __init__(self):
        super(ProbeEnv, self).__init__()
        self.Nm = str()
        self.SetTags("Nm", 'desc:"name of this environment"')
        self.Dsc = str()
        self.SetTags("Dsc", 'desc:"description of this environment"')
        self.Words = go.Slice_string()
        self.SetTags("Words", 'desc:"list of words used for activating state units according to index"')
        self.WordState = etensor.Float32()
        self.SetTags("WordState", 'desc:"current sentence activation state"')
        self.Run = env.Ctr()
        self.SetTags("Run", 'view:"inline" desc:"current run of model as provided during Init"')
        self.Epoch = env.Ctr()
        self.SetTags("Epoch", 'view:"inline" desc:"number of times through Seq.Max number of sequences"')
        self.Trial = env.Ctr()
        self.SetTags("Trial", 'view:"inline" desc:"trial is the step counter within sequence - how many steps taken within current sequence -- it resets to 0 at start of each sequence"')

    def Name(ev):
        return ev.Nm

    def Desc(ev):
        return ev.Dsc

    def Validate(ev):
        """
        InitTMat initializes matrix and labels to given size
        """
        return go.nil

    def State(ev, element):
        if element == "Input":
            return ev.WordState
        return go.nil

    def Init(ev, run):
        ev.Run.Scale = env.Run
        ev.Epoch.Scale = env.Epoch
        ev.Trial.Scale = env.Trial
        ev.Trial.Max = len(ev.Words)
        ev.Run.Init()
        ev.Epoch.Init()
        ev.Trial.Init()
        ev.Run.Cur = run
        ev.Trial.Cur = -1

        ev.WordState.SetShape(go.Slice_int([len(ev.Words)]), go.nil, go.Slice_string(["Words"]))

    def String(ev):
        """
        String returns the current state as a string
        """
        if ev.Trial.Cur < len(ev.Words):
            return "%s" % (ev.Words[ev.Trial.Cur])
        else:
            return ""

    def RenderState(ev):
        """
        RenderState renders the current state
        """
        ev.WordState.SetZeros()
        if ev.Trial.Cur > 0 and ev.Trial.Cur < len(ev.Words):
            ev.WordState.SetFloat1D(ev.Trial.Cur, 1)

    def Step(ev):
        ev.Epoch.Same()
        if ev.Trial.Incr():
            ev.Epoch.Incr()
        ev.RenderState()
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
        return False


