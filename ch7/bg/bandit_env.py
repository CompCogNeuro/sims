# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from leabra import go, pygiv, env, rand, erand, etensor


class BanditEnv(pyviews.ClassViewObj):
    """
    BanditEnv simulates an n-armed bandit, where each of n inputs is associated with
    a specific probability of reward.
    """

    def __init__(self):
        super(BanditEnv, self).__init__()
        self.Nm = str()
        self.SetTags("Nm", 'desc:"name of this environment"')
        self.Dsc = str()
        self.SetTags("Dsc", 'desc:"description of this environment"')
        self.N = int()
        self.SetTags("N", 'desc:"number of different inputs"')
        self.P = []
        self.SetTags("P", 'desc:"no-inline" desc:"probabilities for each option"')
        self.RewVal = float(1)
        self.SetTags("RewVal", 'desc:"value for reward"')
        self.NoRewVal = float(-1)
        self.SetTags("NoRewVal", 'desc:"value for non-reward"')
        self.Option = env.CurPrvInt()
        self.SetTags("Option", 'desc:"bandit option current / prev"')
        self.RndOpt = True
        self.SetTags(
            "RndOpt",
            'desc:"if true, select option at random each Step -- otherwise must be set externally (e.g., by model)"',
        )
        self.Input = etensor.Float64()
        self.SetTags("Input", 'desc:"one-hot input representation of current option"')
        self.Reward = etensor.Float64()
        self.SetTags("Reward", 'desc:"single reward value"')
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

    def SetN(ev, n):
        """
        SetN initializes env for given number of options, and inits states
        """
        ev.N = n
        ev.P = [0.0] * n
        ev.Input.SetShape(go.Slice_int([n]), go.nil, go.Slice_string(["N"]))
        ev.Reward.SetShape(go.Slice_int([1]), go.nil, go.Slice_string(["1"]))
        if ev.RewVal == 0:
            ev.RewVal = 1

    def Validate(ev):
        if ev.N <= 0:
            print("BanditEnv: %s N == 0 -- must set with SetN call" % ev.Nm)
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
        return "S_%d_%g" % (ev.Option.Cur, ev.Reward.Values[0])

    def Init(ev, run):
        ev.Run.Scale = env.Run
        ev.Epoch.Scale = env.Epoch
        ev.Trial.Scale = env.Trial
        ev.Run.Init()
        ev.Epoch.Init()
        ev.Trial.Init()
        ev.Run.Cur = run
        ev.Trial.Cur = -1
        ev.Option.Cur = 0
        ev.Option.Prv = -1

    def RandomOpt(ev):
        """
        RandomOpt selects option at random -- sets Option.Cur and returns it
        """
        op = rand.Intn(ev.N)
        ev.Option.Set(op)
        return op

    def SetInput(ev):
        """
        SetInput sets the input state
        """
        ev.Input.SetZeros()
        ev.Input.Values[ev.Option.Cur] = 1

    def SetReward(ev):
        """
        SetReward sets reward for current option according to probability -- returns true if rewarded
        """
        p = ev.P[ev.Option.Cur]
        rw = erand.BoolP(p)
        if rw:
            ev.Reward.Values[0] = float(ev.RewVal)
        else:
            ev.Reward.Values[0] = float(ev.NoRewVal)
        return rw

    def Step(ev):
        ev.Epoch.Same()

        if ev.RndOpt:
            ev.RandomOpt()
        ev.SetInput()
        ev.SetReward()

        if ev.Trial.Incr():
            ev.Epoch.Incr()
        return True

    def Action(ev, element, input):
        pass

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
