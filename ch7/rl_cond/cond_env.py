# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from leabra import go, pygiv, env, rand, erand, etensor

class OnOff(pygiv.ClassViewObj):
    """
    OnOff represents stimulus On / Off timing
    """

    def __init__(self):
        super(OnOff, self).__init__()
        self.Act = True
        self.SetTags("Act", 'desc:"is this stimulus active -- use it?"')
        self.On = int()
        self.SetTags("On", 'desc:"when stimulus turns on"')
        self.Off = int()
        self.SetTags("Off", 'desc:"when stimulu turns off"')
        self.P = float()
        self.SetTags("P", 'desc:"probability of being active on any given trial"')
        self.OnVar = int()
        self.SetTags("OnVar", 'desc:"variability in onset timing (max number of trials before/after On that it could start)"')
        self.OffVar = int()
        self.SetTags("OffVar", 'desc:"variability in offset timing (max number of trials before/after Off that it could end)"')
        self.CurAct = False
        self.SetTags("CurAct", 'view:"-" desc:"current active status based on P probability"')
        self.CurOn = int()
        self.SetTags("CurOn", 'view:"-" desc:"current on / off values using Var variability"')
        self.CurOff = int()
        self.SetTags("CurOff", 'view:"-" desc:"current on / off values using Var variability"')
        
    def Set(oo, act, on, off):
        oo.Act = act
        oo.On = on
        oo.Off = off
        oo.P = 1 # default

    def TrialUpdt(oo):
        """
        TrialUpdt updates Cur state at start of trial
        """
        if not oo.Act:
            return
        oo.CurAct = erand.BoolP(oo.P)
        oo.CurOn = oo.On - oo.OnVar + 2*rand.Intn(oo.OnVar+1)
        oo.CurOff = oo.Off - oo.OffVar + 2*rand.Intn(oo.OffVar+1)

    def IsOn(oo, tm):
        """
        IsOn returns true if should be on according current time
        """
        return oo.Act and oo.CurAct and tm >= oo.CurOn and tm < oo.CurOff


class CondEnv(pygiv.ClassViewObj):
    """
    String returns the current state as a string
    """

    def __init__(self):
        super(CondEnv, self).__init__()
        self.Nm = str()
        self.SetTags("Nm", 'desc:"name of this environment"')
        self.Dsc = str()
        self.SetTags("Dsc", 'desc:"description of this environment"')
        self.TotTime = int()
        self.SetTags("TotTime", 'desc:"total time for trial"')
        self.CSA = OnOff()
        self.SetTags("CSA", 'view:"inline" desc:"Conditioned stimulus A (e.g., Tone)"')
        self.CSB = OnOff()
        self.SetTags("CSB", 'view:"inline" desc:"Conditioned stimulus B (e.g., Light)"')
        self.CSC = OnOff()
        self.SetTags("CSC", 'view:"inline" desc:"Conditioned stimulus C"')
        self.US = OnOff()
        self.SetTags("US", 'view:"inline" desc:"Unconditioned stimulus -- reward"')
        self.RewVal = float()
        self.SetTags("RewVal", 'desc:"value for reward"')
        self.NoRewVal = float()
        self.SetTags("NoRewVal", 'desc:"value for non-reward"')
        self.Input = etensor.Float64()
        self.SetTags("Input", 'desc:"one-hot input representation of current option"')
        self.Reward = etensor.Float64()
        self.SetTags("Reward", 'desc:"single reward value"')
        self.Run = env.Ctr()
        self.SetTags("Run", 'view:"inline" desc:"current run of model as provided during Init"')
        self.Epoch = env.Ctr()
        self.SetTags("Epoch", 'view:"inline" desc:"number of times through Seq.Max number of sequences"')
        self.Trial = env.Ctr()
        self.SetTags("Trial", 'view:"inline" desc:"one trial is a pass through all TotTime Events"')
        self.Event = env.Ctr()
        self.SetTags("Event", 'view:"inline" desc:"event is one time step within Trial -- e.g., CS turning on, etc"')

    def Name(ev):
        return ev.Nm

    def Desc(ev):
        return ev.Dsc

    def Defaults(ev):
        ev.TotTime = 20
        ev.CSA.Set(True, 10, 16)
        ev.CSB.Set(False, 2, 10)
        ev.CSC.Set(False, 2, 5)
        ev.US.Set(True, 15, 16)

    def Validate(ev):
        if ev.TotTime == 0:
            ev.Defaults()
        return 0

    def State(ev, element):
        if element == "Input":
            return ev.Input
        if element == "Reward":
            return ev.Reward
        return go.nil

    def String(ev):
        """
        String returns the current state as a string
        """
        return "S_%d_%v" % (ev.Event.Cur, ev.Reward.Values[0])

    def Init(ev, run):
        ev.Input.SetShape(go.Slice_int([3, ev.TotTime]), go.nil, go.Slice_string(["3", "TotTime"]))
        ev.Reward.SetShape(go.Slice_int([1]), go.nil, go.Slice_string(["1"]))
        ev.Run.Scale = env.Run
        ev.Epoch.Scale = env.Epoch
        ev.Trial.Scale = env.Trial
        ev.Event.Scale = env.Event
        ev.Run.Init()
        ev.Epoch.Init()
        ev.Trial.Init()
        ev.Event.Init()
        ev.Run.Cur = run
        ev.Event.Max = ev.TotTime
        ev.Event.Cur = -1
        ev.TrialUpdt()

    def TrialUpdt(ev):
        """
        TrialUpdt updates all random vars at start of trial
        """
        ev.CSA.TrialUpdt()
        ev.CSB.TrialUpdt()
        ev.CSC.TrialUpdt()
        ev.US.TrialUpdt()

    def SetInput(ev):
        """
        SetInput sets the input state
        """
        ev.Input.SetZeros()
        tm = ev.Event.Cur
        if ev.CSA.IsOn(tm):
            ev.Input.Values[tm] = 1
        if ev.CSB.IsOn(tm):
            ev.Input.Values[ev.TotTime+tm] = 1
        if ev.CSC.IsOn(tm):
            ev.Input.Values[2*ev.TotTime+tm] = 1

    def SetReward(ev):
        """
        SetReward sets reward for current option according to probability -- returns true if rewarded
        """
        tm = ev.Event.Cur
        rw = ev.US.IsOn(tm)
        if rw:
            ev.Reward.Values[0] = float(ev.RewVal)
        else:
            ev.Reward.Values[0] = float(ev.NoRewVal)
        return rw

    def Step(ev):
        ev.Epoch.Same()
        ev.Trial.Same()

        incr = ev.Event.Incr()
        ev.SetInput()
        ev.SetReward()

        if incr:
            ev.TrialUpdt()
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
        if scale == env.Event:
            return ev.Event.Cur
        return -1

    def CounterPrv(ev, scale):
        if scale == env.Run:
            return ev.Run.Prv
        if scale == env.Epoch:
            return ev.Epoch.Prv
        if scale == env.Trial:
            return ev.Trial.Prv
        if scale == env.Event:
            return ev.Event.Prv
        return -1
        
    def CounterChg(ev, scale):
        if scale == env.Run:
            return ev.Run.Chg
        if scale == env.Epoch:
            return ev.Epoch.Chg
        if scale == env.Trial:
            return ev.Trial.Chg
        if scale == env.Event:
            return ev.Event.Chg
        return False
        
