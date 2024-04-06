# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from leabra import go, pygiv, env, rand, erand, etensor

import os


def read_as_string(fnm):
    # reads file as string
    if not os.path.isfile(fnm):
        return ""
    with open(fnm, "r") as f:
        val = f.read()
    return val


class SemEnv(pygiv.ClassViewObj):
    """
    SemEnv presents paragraphs of text, loaded from file(s)
    This assumes files have all been pre-filtered so only relevant words are present.
    """

    def __init__(self):
        super(SemEnv, self).__init__()
        self.Nm = str()
        self.SetTags("Nm", 'desc:"name of this environment"')
        self.Dsc = str()
        self.SetTags("Dsc", 'desc:"description of this environment"')
        self.Sequential = False
        self.SetTags(
            "Sequential",
            'desc:"if true, go sequentially through paragraphs -- else permuted"',
        )
        self.Order = go.Slice_int()
        self.SetTags(
            "Order",
            'desc:"permuted order of paras to present if not sequential -- updated every time through the list"',
        )
        self.TextFiles = []
        self.SetTags("TextFiles", 'desc:"paths to text files"')
        self.Words = go.Slice_string()
        self.SetTags("Words", 'desc:"list of words, in alpha order"')
        self.WordMap = {}
        self.SetTags("WordMap", 'view:"-" desc:"map of words onto index in Words list"')
        self.CurParaState = etensor.Float32()
        self.SetTags("CurParaState", 'desc:"current para activation state"')
        self.Paras = []
        self.SetTags("Paras", 'view:"-" desc:"paragraphs"')
        self.ParaLabels = []
        self.SetTags(
            "ParaLabels",
            'view:"-" desc:"special labels for each paragraph (provided in first word of para)"',
        )
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
            "Trial",
            'view:"inline" desc:"trial is the step counter within epoch -- this is the index into Paras"',
        )

    def Name(ev):
        return ev.Nm

    def Desc(ev):
        return ev.Dsc

    def Defaults(ev):
        pass

    def Validate(ev):
        return go.nil

    def State(ev, element):
        if element == "Input":
            return ev.CurParaState
        return go.nil

    def Init(ev, run):
        ev.Run.Scale = env.Run
        ev.Epoch.Scale = env.Epoch
        ev.Trial.Scale = env.Trial
        ev.Run.Init()
        ev.Epoch.Init()
        ev.Trial.Init()
        ev.Run.Cur = run
        ev.InitOrder()

        nw = len(ev.Words)
        ev.CurParaState.SetShape(go.Slice_int([nw]), go.nil, go.Slice_string(["Words"]))

    def InitOrder(ev):
        """
        InitOrder initializes the order based on current Paras, resets Trial.Cur = -1 too
        """
        np = len(ev.Paras)
        ev.Order = rand.Perm(np)
        # and always maintain Order so random number usage is same regardless, and if
        # user switches between Sequential and random at any point, it all works..
        ev.Trial.Max = np
        ev.Trial.Cur = -1  # init state -- key so that first Step() = 0

    def OpenTexts(ev, txts):
        """
        OpenTexts opens multiple text files -- use this as main API
        even if only opening one text
        """
        ev.TextFiles = txts
        ev.Paras = []
        for tf in ev.TextFiles:
            ev.OpenText(tf)

    def OpenText(ev, fname):
        """
        OpenText opens one text file
        """
        txt = read_as_string(fname)
        ev.ScanText(txt)

    def ScanText(ev, txt):
        """
        ScanText scans given text file from reader, adding to Paras
        """
        lbl = ""
        lns = txt.splitlines()
        cur = []
        for ln in lns:
            sp = ln.split()
            if len(sp) == 0:
                ev.Paras.append(cur)
                ev.ParaLabels.append(lbl)
                cur = []
                lbl = ""
            else:
                coli = sp[0].find(":")
                if coli > 0:
                    lbl = sp[0][:coli]
                    sp = sp[1:]
                for s in sp:
                    cur.append(s)
        if len(cur) > 0:
            ev.Paras.append(cur)
            ev.ParaLabels.append(lbl)

    def CheckWords(ev, wrds):
        """
        CheckWords checks that the words in the slice (one word per index) are in the list.
        Returns True = error for any missing words.
        """
        missing = ""
        for wrd in wrds:
            if not wrd in ev.WordMap:
                missing += wrd + " "
        if missing != "":
            print("CheckWords: these words were not found: %s", (missing))
            return True
        return False

    def SetParas(ev, paras):
        """
        SetParas sets the paragraphs from list of space-separated word strings -- each string is a paragraph.
        calls InitOrder after to reset.
        returns error for any missing words (from CheckWords)
        """
        ev.Paras = []
        ev.ParaLabels = []
        err = False
        for i, ps in enumerate(paras):
            lbl = ""
            sp = ps.split()
            if len(sp) > 0:
                coli = sp[0].find(":")
                if coli > 0:
                    lbl = sp[0][:coli]
                    sp = sp[1:]
            ev.Paras.append(sp)
            ev.ParaLabels.append(lbl)
            er = ev.CheckWords(sp)
            if er:
                err = True
        ev.InitOrder()
        return err

    def OpenWords(ev, fname):
        txt = read_as_string(fname)
        ev.ScanWords(txt)

    def ScanWords(ev, txt):
        ev.Words = go.Slice_string()
        lns = txt.splitlines()
        for ln in lns:
            sp = ln.split()
            for s in sp:
                ev.Words.append(s)
        ev.WordMapFmWords()

    def WordMapFmWords(ev):
        ev.WordMap = {}
        for i, wrd in enumerate(ev.Words):
            ev.WordMap[wrd] = i

    def WordsFmWordMap(ev):
        ev.Words = go.Slice_string()
        ctr = 0
        for wrd in ev.WordMap:
            ev.Words.append(wrd)
            ctr += 1
        ev.Words.sort()
        for i, wrd in enumerate(ev.Words):
            ev.WordMap[wrd] = i

    def WordsFmText(ev):
        ev.WordMap = {}
        for para in ev.Paras:
            for wrd in para:
                ev.WordMap[wrd] = -1
        ev.WordsFmWordMap()

    def Step(ev):
        ev.Epoch.Same()
        if ev.Trial.Incr():
            erand.PermuteInts(ev.Order)
            ev.Epoch.Incr()
        ev.SetParaState()
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

    def String(ev):
        """
        String returns the string rep of the LED env state
        """
        cpar = ev.CurPara()
        if len(cpar) == 0:
            return ""
        str = cpar[0]
        if len(cpar) > 1:
            str += " " + cpar[1]
            if len(cpar) > 2:
                str += " ... " + cpar[len(cpar) - 1]
        return str

    def ParaIndex(ev):
        """
        ParaIndex returns the current idx number in Paras, based on Sequential / perumuted Order
        """
        if ev.Trial.Cur < 0:
            return -1
        if ev.Sequential:
            return ev.Trial.Cur
        return ev.Order[ev.Trial.Cur]

    def CurPara(ev):
        """
        CurPara returns the current paragraph
        """
        pidx = ev.ParaIndex()
        if pidx >= 0 and pidx < len(ev.Paras):
            return ev.Paras[pidx]
        return []

    def SetParaState(ev):
        """
        SetParaState sets the para state from current para
        """
        cpar = ev.CurPara()
        ev.CurParaState.SetZeros()
        for wrd in cpar:
            widx = ev.WordMap[wrd]
            ev.CurParaState.SetFloat1D(widx, 1)
