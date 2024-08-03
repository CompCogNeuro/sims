# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from leabra import go, pygiv, env, rand, erand, etensor, esg

import os


class SentGenEnv(pyviews.ClassViewObj):
    """
    SentGenEnv generates sentences using a grammar that is parsed from a
    text file.  The core of the grammar is rules with various items
    chosen at random during generation -- these items can be
    more rules terminal tokens.
    """

    def __init__(self):
        super(SentGenEnv, self).__init__()
        self.Nm = str()
        self.SetTags("Nm", 'desc:"name of this environment"')
        self.Dsc = str()
        self.SetTags("Dsc", 'desc:"description of this environment"')
        self.Rules = esg.Rules()
        self.SetTags(
            "Rules",
            'desc:"core sent-gen rules -- loaded from a grammar / rules file -- Gen() here generates one sentence"',
        )
        self.PPassive = float()
        self.SetTags(
            "PPassive", 'desc:"probability of generating passive sentence forms"'
        )
        self.WordTrans = {}
        self.SetTags(
            "WordTrans", 'desc:"translate unambiguous words into ambiguous words"'
        )
        self.Words = go.Slice_string()
        self.SetTags(
            "Words",
            'desc:"list of words used for activating state units according to index"',
        )
        self.WordMap = {}
        self.SetTags("WordMap", 'desc:"map of words onto index in Words list"')
        self.Roles = []
        self.SetTags(
            "Roles",
            'desc:"list of roles used for activating state units according to index"',
        )
        self.RoleMap = {}
        self.SetTags("RoleMap", 'desc:"map of roles onto index in Roles list"')
        self.Fillers = []
        self.SetTags(
            "Fillers",
            'desc:"list of filler concepts used for activating state units according to index"',
        )
        self.FillerMap = {}
        self.SetTags("FillerMap", 'desc:"map of roles onto index in Words list"')
        self.AmbigVerbs = []
        self.SetTags("AmbigVerbs", 'desc:"ambiguous verbs"')
        self.AmbigNouns = []
        self.SetTags("AmbigNouns", 'desc:"ambiguous nouns"')
        self.AmbigVerbsMap = {}
        self.SetTags("AmbigVerbsMap", 'desc:"map of ambiguous verbs"')
        self.AmbigNounsMap = {}
        self.SetTags("AmbigNounsMap", 'desc:"map of ambiguous nouns"')
        self.CurSentOrig = []
        self.SetTags(
            "CurSentOrig", 'desc:"original current sentence as generated from Rules"'
        )
        self.CurSent = []
        self.SetTags(
            "CurSent",
            'desc:"current sentence, potentially transformed to passive form"',
        )
        self.NAmbigNouns = int()
        self.SetTags("NAmbigNouns", 'desc:"number of ambiguous nouns"')
        self.NAmbigVerbs = int()
        self.SetTags("NAmbigVerbs", 'desc:"number of ambiguous verbs (0 or 1)"')
        self.SentInputs = []
        self.SetTags(
            "SentInputs",
            'desc:"generated sequence of sentence inputs including role-filler queries"',
        )
        self.SentIndex = env.CurPrvInt()
        self.SetTags("SentIndex", 'desc:"current index within sentence inputs"')
        self.QType = str()
        self.SetTags(
            "QType", 'desc:"current question type -- from 4th value of SentInputs"'
        )
        self.WordState = etensor.Float32()
        self.SetTags("WordState", 'desc:"current sentence activation state"')
        self.RoleState = etensor.Float32()
        self.SetTags("RoleState", 'desc:"current role query activation state"')
        self.FillerState = etensor.Float32()
        self.SetTags("FillerState", 'desc:"current filler query activation state"')
        self.Run = env.Ctr()
        self.SetTags(
            "Run", 'view:"inline" desc:"current run of model as provided during Init"'
        )
        self.Epoch = env.Ctr()
        self.SetTags(
            "Epoch",
            'view:"inline" desc:"number of times through Seq.Max number of sequences"',
        )
        self.Seq = env.Ctr()
        self.SetTags("Seq", 'view:"inline" desc:"sequence counter within epoch"')
        self.Tick = env.Ctr()
        self.SetTags("Tick", 'view:"inline" desc:"tick counter within sequence"')
        self.Trial = env.Ctr()
        self.SetTags(
            "Trial",
            'view:"inline" desc:"trial is the step counter within sequence - how many steps taken within current sequence -- it resets to 0 at start of each sequence"',
        )

    def Name(ev):
        return ev.Nm

    def Desc(ev):
        return ev.Dsc

    def Validate(ev):
        """
        InitTMat initializes matrix and labels to given size
        """
        # ev.Rules.Validate()
        return go.nil

    def State(ev, element):
        if element == "Input":
            return ev.WordState
        if element == "Role":
            return ev.RoleState
        if element == "Filler":
            return ev.FillerState
        return go.nil

    def Init(ev, run):
        ev.Run.Scale = env.Run
        ev.Epoch.Scale = env.Epoch
        ev.Seq.Scale = env.Sequence
        ev.Tick.Scale = env.Tick
        ev.Trial.Scale = env.Trial
        ev.Run.Init()
        ev.Epoch.Init()
        ev.Seq.Init()
        ev.Tick.Init()
        ev.Trial.Init()
        ev.Run.Cur = run
        ev.Trial.Cur = -1  # init state -- key so that first Step() = 0
        ev.SentIndex.Set(-1)

        ev.Rules.Init()
        ev.MapsFmWords()

        ev.WordState.SetShape(
            go.Slice_int([len(ev.Words)]), go.nil, go.Slice_string(["Words"])
        )
        ev.RoleState.SetShape(
            go.Slice_int([len(ev.Roles)]), go.nil, go.Slice_string(["Roles"])
        )
        ev.FillerState.SetShape(
            go.Slice_int([len(ev.Fillers)]), go.nil, go.Slice_string(["Fillers"])
        )

    def MapsFmWords(ev):
        ev.WordMap = {}
        for i, wrd in enumerate(ev.Words):
            ev.WordMap[wrd] = i
        ev.RoleMap = {}
        for i, wrd in enumerate(ev.Roles):
            ev.RoleMap[wrd] = i
        ev.FillerMap = {}
        for i, wrd in enumerate(ev.Fillers):
            ev.FillerMap[wrd] = i
        ev.AmbigVerbsMap = {}
        for i, wrd in enumerate(ev.AmbigVerbs):
            ev.AmbigVerbsMap[wrd] = i
        ev.AmbigNounsMap = {}
        for i, wrd in enumerate(ev.AmbigNouns):
            ev.AmbigNounsMap[wrd] = i

    def CurInputs(ev):
        """
        CurInputs returns current inputs triple from SentInputs
        """
        if ev.SentIndex.Cur >= 0 and ev.SentIndex.Cur < len(ev.SentInputs):
            return ev.SentInputs[ev.SentIndex.Cur]
        return []

    def String(ev):
        """
        String returns the current state as a string
        """
        cur = ev.CurInputs()
        if len(cur) >= 4:
            return "%s %s=%s %s" % (cur[0], cur[1], cur[2], cur[3])
        return ""

    def NextSent(ev):
        """
        NextSent generates the next sentence and all the queries for it
        """

        ev.CurSent = ev.Rules.Gen()

        ev.Rules.States.TrimQualifiers()
        ev.SentStats()
        ev.SentIndex.Set(0)
        if "Case" in ev.Rules.States:
            cs = ev.Rules.States["Case"]
            if cs == "Passive":
                ev.SentSeqPassive()
            else:
                ev.SentSeqActive()
        else:
            if erand.BoolProb(ev.PPassive, -1):
                ev.SentSeqPassive()
            else:
                ev.SentSeqActive()

    def TransWord(ev, word):
        """
        TransWord gets the translated word
        """
        word = word.lower()
        if word in ev.WordTrans:
            tr = ev.WordTrans[word]
            return tr
        return word

    def SentStats(ev):
        """
        SentStats computes stats on sentence (ambig words)
        """
        ev.NAmbigNouns = 0
        ev.NAmbigVerbs = 0
        for wrd in ev.CurSent:
            wrd = ev.TransWord(wrd)
            if wrd in ev.AmbigVerbsMap:
                ev.NAmbigVerbs += 1
            if wrd in ev.AmbigNounsMap:
                ev.NAmbigNouns += 1

    def CheckWords(ev, wrd, role, fill):
        """
        CheckWords reports errors if words not found, if not empty
        """
        if not wrd in ev.WordMap:
            print("word not found in WordMap: %s, sent: %s", (wrd, ev.CurSent))
        if not role in ev.RoleMap:
            print("word not found in RoleMap: %s, sent: %s", (role, ev.CurSent))
        if not fill in ev.FillerMap:
            print("word not found in FillerMap: %s, sent: %s", (fill, ev.CurSent))

    def NewInputs(ev):
        ev.SentInputs = []

    def AddRawInput(ev, word, role, fill, stat):
        """
        AddRawInput adds raw input
        """
        ev.SentInputs.append([word, role, fill, stat])

    def AddInput(ev, sidx, role, stat):
        """
        AddInput adds a new input with given sentence index word and role query
        stat is an extra status var: "revq" or "curq" (review question, vs. current question)
        """
        wrd = ev.TransWord(ev.CurSent[sidx])
        fill = ev.Rules.States[role]
        ev.CheckWords(wrd, role, fill)
        ev.AddRawInput(wrd, role, fill, stat)

    def AddQuestion(ev, role):
        """
        AddQuestion adds a new input with 'question' word and role query
        automatically marked as a "revq"
        """
        wrd = "question"
        fill = ev.Rules.States[role]
        ev.CheckWords(wrd, role, fill)
        ev.AddRawInput(wrd, role, fill, "revq")

    def SentSeqActive(ev):
        """
        SentSeqActive active form sentence sequence, with incremental review questions
        """
        ev.NewInputs()
        ev.AddRawInput("start", "Action", "None", "curq")
        mod = ""
        if "Mod" in ev.Rules.States:
            mod = ev.Rules.States["Mod"]
        seq = ["Agent", "Action", "Patient", mod]
        for si in range(3):
            sq = seq[si]
            ev.AddInput(si, sq, "curq")
            if si == 1:
                ev.AddInput(si, "Agent", "revq")
            if si == 2:
                ev.AddInput(si, "Action", "revq")
        slen = len(ev.CurSent)
        if slen == 3:
            return

        for si in range(slen - 1):
            ri = rand.Intn(3)
            ev.AddInput(si, seq[ri], "revq")
        ev.AddInput(slen - 1, mod, "curq")
        ri = rand.Intn(3)
        if "FinalQ" in ev.Rules.States:
            fq = ev.Rules.States["FinalQ"]
            for i, s in enumerate(seq):
                if seq[i] == fq:
                    ri = i
                    break
        ev.AddInput(slen - 1, seq[ri], "revq")

    def SentSeqPassive(ev):
        """
        SentSeqPassive passive form sentence sequence, with incremental review questions
        """
        ev.NewInputs()
        ev.AddRawInput("start", "Action", "None", "curq")
        mod = ev.Rules.States["Mod"]
        seq = ["Agent", "Action", "Patient", mod]
        ev.AddInput(2, "Patient", "curq")
        ev.AddRawInput("was", "Patient", ev.Rules.States["Patient"], "revq")
        ev.AddInput(1, "Action", "curq")
        ev.AddRawInput("by", "Action", ev.Rules.States["Action"], "revq")
        ev.AddInput(0, "Agent", "curq")
        # note: we already get review questions for free with was and by
        # get any modifier words with random query
        slen = len(ev.CurSent)
        for si in range(slen - 1):
            ri = rand.Intn(3)  # choose a role to query at random
            ev.AddInput(si, seq[ri], "revq")
        ev.AddInput(slen - 1, mod, "curq")
        ri = rand.Intn(3)  # choose a role to query at random
        # ev.AddQuestion(seq[ri])
        ev.AddInput(slen - 1, seq[ri], "revq")

    def RenderState(ev):
        """
        RenderState renders the current state
        """
        ev.WordState.SetZeros()
        ev.RoleState.SetZeros()
        ev.FillerState.SetZeros()
        cur = ev.CurInputs()
        if cur == 0:
            return
        widx = ev.WordMap[cur[0]]
        ev.WordState.SetFloat1D(widx, 1)
        ridx = ev.RoleMap[cur[1]]
        ev.RoleState.SetFloat1D(ridx, 1)
        fidx = ev.FillerMap[cur[2]]
        ev.FillerState.SetFloat1D(fidx, 1)
        ev.QType = cur[3]

    def NextState(ev):
        """
        NextState generates the next inputs
        """
        if ev.SentIndex.Cur < 0:
            ev.NextSent()
        else:
            ev.SentIndex.Incr()
        if ev.SentIndex.Cur >= len(ev.SentInputs):
            ev.NextSent()
        ev.RenderState()

    def Step(ev):
        ev.Epoch.Same()
        ev.NextState()
        ev.Trial.Incr()
        ev.Tick.Incr()
        if ev.SentIndex.Cur == 0:
            ev.Tick.Init()
            if ev.Seq.Incr():
                ev.Epoch.Incr()
        return True

    def CounterCur(ev, scale):
        if scale == env.Run:
            return ev.Run.Cur
        if scale == env.Epoch:
            return ev.Epoch.Cur
        if scale == env.Sequence:
            return ev.Seq.Cur
        if scale == env.Tick:
            return ev.Tick.Cur
        if scale == env.Trial:
            return ev.Trial.Cur
        return -1

    def CounterPrv(ev, scale):
        if scale == env.Run:
            return ev.Run.Prv
        if scale == env.Epoch:
            return ev.Epoch.Prv
        if scale == env.Sequence:
            return ev.Seq.Prv
        if scale == env.Tick:
            return ev.Tick.Prv
        if scale == env.Trial:
            return ev.Trial.Prv
        return -1

    def CounterChg(ev, scale):
        if scale == env.Run:
            return ev.Run.Chg
        if scale == env.Epoch:
            return ev.Epoch.Chg
        if scale == env.Sequence:
            return ev.Seq.Chg
        if scale == env.Tick:
            return ev.Tick.Chg
        if scale == env.Trial:
            return ev.Trial.Chg
        return False
