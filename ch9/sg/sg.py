#!/usr/local/bin/pyleabra -i

# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# use:
# pyleabra -i ra25.py 
# to run in gui interactive mode from the command line (or pyleabra, import ra25)
# see main function at the end for startup args

# sg is the sentence gestalt model, which learns to encode both
# syntax and semantics of sentences in an integrated "gestalt"
# hidden layer. The sentences have simple agent-verb-patient
# structure with optional prepositional or adverb modifier
# phrase at the end, and can be either in the active or passive
# form (80% active, 20% passive). There are ambiguous terms that
# need to be resolved via context, showing a key interaction
# between syntax and semantics.

from leabra import go, leabra, emer, relpos, eplot, env, agg, patgen, prjn, etable, efile, split, etensor, params, netview, rand, erand, gi, giv, pygiv, pyparams, mat32, metric, simat, pca, clust, deep

import importlib as il  #il.reload(ra25) -- doesn't seem to work for reasons unknown
import io, sys, getopt
from datetime import datetime, timezone
from enum import Enum
import numpy as np

from sg_env import SentGenEnv
from sg_probe_env import ProbeEnv
import sg_words

# import matplotlib
# matplotlib.use('SVG')
# import matplotlib.pyplot as plt
# plt.rcParams['svg.fonttype'] = 'none'  # essential for not rendering fonts as paths

# note: pandas, xarray or pytorch TensorDataSet can be used for input / output
# patterns and recording of "log" data for plotting.  However, the etable.Table
# has better GUI and API support, and handles tensor columns directly unlike
# pandas.  Support for easy migration between these is forthcoming.
# import pandas as pd

# this will become Sim later.. 
TheSim = 1

# LogPrec is precision for saving float values in logs
LogPrec = 4

# note: we cannot use methods for callbacks from Go -- must be separate functions
# so below are all the callbacks from the GUI toolbar actions

def InitCB(recv, send, sig, data):
    TheSim.Init()
    TheSim.UpdateClassView()
    TheSim.vp.SetNeedsFullRender()

def TrainCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.ToolBar.UpdateActions()
        TheSim.Train()

def StopCB(recv, send, sig, data):
    TheSim.Stop()

def StepTrialCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.TrainTrial()
        TheSim.IsRunning = False
        TheSim.UpdateClassView()
        TheSim.vp.SetNeedsFullRender()

def StepSeqCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.ToolBar.UpdateActions()
        TheSim.TrainSeq()

def StepEpochCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.ToolBar.UpdateActions()
        TheSim.TrainEpoch()

def StepRunCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.ToolBar.UpdateActions()
        TheSim.TrainRun()

def InitTestCB(recv, send, sig, data):
    TheSim.InitTest()
    TheSim.UpdateClassView()
    TheSim.vp.SetNeedsFullRender()

def TestTrialCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.TestTrial(False)
        TheSim.IsRunning = False
        TheSim.UpdateClassView()
        TheSim.vp.SetNeedsFullRender()

def TestSeqCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.TestSeq()
        TheSim.IsRunning = False
        TheSim.UpdateClassView()
        TheSim.vp.SetNeedsFullRender()

def TestAllCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.ToolBar.UpdateActions()
        TheSim.RunTestAll()

def ProbeAllCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.ToolBar.UpdateActions()
        TheSim.ProbeAll()

def ResetTstTrlLogCB(recv, send, sig, data):
    TheSim.TstTrlLog.SetNumRows(0)
    TheSim.TstTrlPlot.Update()

def OpenWeightsCB(recv, send, sig, data):    
    TheSim.OpenWts()

def NewRndSeedCB(recv, send, sig, data):
    TheSim.NewRndSeed()

def ReadmeCB(recv, send, sig, data):
    gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch9/sg/README.md")

def UpdtFuncNotRunning(act):
    act.SetActiveStateUpdt(not TheSim.IsRunning)
    
def UpdtFuncRunning(act):
    act.SetActiveStateUpdt(TheSim.IsRunning)

def FilterTickEq5(et, row):
    return etable.Table(handle=et).CellFloat("Tick", row) == 5
    
#####################################################    
#     Sim

class Sim(pygiv.ClassViewObj):
    """
    Sim encapsulates the entire simulation model, and we define all the
    functionality as methods on this struct.  This structure keeps all relevant
    state information organized and available without having to pass everything around
    as arguments to methods, and provides the core GUI interface (note the view tags
    for the fields which provide hints to how things should be displayed).
    """

    def __init__(self):
        super(Sim, self).__init__()
        self.Net = deep.Network()
        self.SetTags("Net", 'view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"')
        self.TrnEpcLog = etable.Table()
        self.SetTags("TrnEpcLog", 'view:"no-inline" desc:"training epoch-level log data"')
        self.TstEpcLog = etable.Table()
        self.SetTags("TstEpcLog", 'view:"no-inline" desc:"testing epoch-level log data"')
        self.TrnTrlLog = etable.Table()
        self.SetTags("TrnTrlLog", 'view:"no-inline" desc:"training trial-level log data"')
        self.TstTrlLog = etable.Table()
        self.SetTags("TstTrlLog", 'view:"no-inline" desc:"testing trial-level log data"')
        self.SentProbeTrlLog = etable.Table()
        self.SetTags("SentProbeTrlLog", 'view:"no-inline" desc:"probing trial-level log data"')
        self.NounProbeTrlLog = etable.Table()
        self.SetTags("NounProbeTrlLog", 'view:"no-inline" desc:"probing trial-level log data"')
        self.TrnTrlAmbStats = etable.Table()
        self.SetTags("TrnTrlAmbStats", 'view:"no-inline" desc:"aggregate trl stats for last epc"')
        self.TrnTrlQTypStats = etable.Table()
        self.SetTags("TrnTrlQTypStats", 'view:"no-inline" desc:"aggregate trl stats for last epc"')
        self.RunLog = etable.Table()
        self.SetTags("RunLog", 'view:"no-inline" desc:"summary log of each run"')
        self.RunStats = etable.Table()
        self.SetTags("RunStats", 'view:"no-inline" desc:"aggregate stats on all runs"')
        self.Params = params.Sets()
        self.SetTags("Params", 'view:"no-inline" desc:"full collection of param sets"')
        self.ParamSet = str()
        self.SetTags("ParamSet", 'desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set -- can use multiple names separated by spaces (don\'t put spaces in ParamSet names!)"')
        self.Tag = str()
        self.SetTags("Tag", 'desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"')
        self.MaxRuns = int(1)
        self.SetTags("MaxRuns", 'desc:"maximum number of model runs to perform"')
        self.MaxEpcs = int(500)
        self.SetTags("MaxEpcs", 'desc:"maximum number of epochs to run per model run"')
        self.NZeroStop = int(5)
        self.SetTags("NZeroStop", 'desc:"if a positive number, training will stop after this many epochs with zero SSE"')
        self.TrainEnv = SentGenEnv()
        self.SetTags("TrainEnv", 'desc:"Training environment -- contains everything about iterating over input / output patterns over training"')
        self.TestEnv = SentGenEnv()
        self.SetTags("TestEnv", 'desc:"Testing environment -- manages iterating over testing"')
        self.SentProbeEnv = SentGenEnv()
        self.SetTags("SentProbeEnv", 'desc:"Probe environment -- manages iterating over testing"')
        self.NounProbeEnv = ProbeEnv()
        self.SetTags("NounProbeEnv", 'desc:"Probe environment -- manages iterating over testing"')
        self.Time = leabra.Time()
        self.SetTags("Time", 'desc:"leabra timing parameters and state"')
        self.ViewOn = True
        self.SetTags("ViewOn", 'desc:"whether to update the network view while running"')
        self.TrainUpdt = leabra.TimeScales.AlphaCycle
        self.SetTags("TrainUpdt", 'desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"')
        self.TestUpdt = leabra.TimeScales.AlphaCycle
        self.SetTags("TestUpdt", 'desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"')
        self.TestInterval = int(5000)
        self.SetTags("TestInterval", 'desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"')
        self.LayStatNms = go.Slice_string(["Encode", "EncodeCT", "Gestalt", "GestaltCT", "Decode"])
        self.SetTags("LayStatNms", 'view:"-" desc:"names of layers to collect more detailed stats on (avg act, etc)"')
        self.StatLayNms = go.Slice_string(["Filler", "EncodeP"])
        self.SetTags("StatLayNms", 'view:"-" desc:"stat layers"')
        self.StatNms = go.Slice_string(["Fill", "Inp"])
        self.SetTags("StatNms", 'view:"-" desc:"stat short names"')
        self.ProbeNms = go.Slice_string(["Gestalt", "GestaltCT"])
        self.SetTags("ProbeNms", 'view:"-" desc:"layers to probe"')

        # statistics: note use float64 as that is best for etable.Table
        self.TrlOut = str()
        self.SetTags("TrlOut", 'inactive:"+" desc:"output response(s) output units active > .2"')
        self.TrlPred = str()
        self.SetTags("TrlPred", 'inactive:"+" desc:"predicted word(s) active > .2"')
        self.TrlErr = [0.0] * 2
        self.SetTags("TrlErr", 'inactive:"+" desc:"1 if trial was error, 0 if correct -- based on SSE = 0 (subject to .5 unit-wise tolerance)"')
        self.TrlSSE = [0.0] * 2
        self.SetTags("TrlSSE", 'inactive:"+" desc:"current trial\'s sum squared error"')
        self.TrlAvgSSE = [0.0] * 2
        self.SetTags("TrlAvgSSE", 'inactive:"+" desc:"current trial\'s average sum squared error"')
        self.TrlCosDiff = [0.0] * 2
        self.SetTags("TrlCosDiff", 'inactive:"+" desc:"current trial\'s cosine difference"')
        self.EpcSSE = [0.0] * 2
        self.SetTags("EpcSSE", 'inactive:"+" desc:"last epoch\'s total sum squared error"')
        self.EpcAvgSSE = [0.0] * 2
        self.SetTags("EpcAvgSSE", 'inactive:"+" desc:"last epoch\'s average sum squared error (average over trials, and over units within layer)"')
        self.EpcPctErr = [0.0] * 2
        self.SetTags("EpcPctErr", 'inactive:"+" desc:"last epoch\'s average TrlErr"')
        self.EpcPctCor = [0.0] * 2
        self.SetTags("EpcPctCor", 'inactive:"+" desc:"1 - last epoch\'s average TrlErr"')
        self.EpcCosDiff = [0.0] * 2
        self.SetTags("EpcCosDiff", 'inactive:"+" desc:"last epoch\'s average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"')
        self.EpcPerTrlMSec = float()
        self.SetTags("EpcPerTrlMSec", 'inactive:"+" desc:"how long did the epoch take per trial in wall-clock milliseconds"')
        self.FirstZero = int()
        self.SetTags("FirstZero", 'inactive:"+" desc:"epoch at when SSE first went to zero"')
        self.NZero = int()
        self.SetTags("NZero", 'inactive:"+" desc:"number of epochs in a row with zero SSE"')

        # internal state - view:"-"
        self.SumN = [0.0] * 2
        self.SetTags("SumN", 'view:"-" inactive:"+" desc:"number of each stat"')
        self.SumErr = [0.0] * 2
        self.SetTags("SumErr", 'view:"-" inactive:"+" desc:"sum to increment as we go through epoch"')
        self.SumSSE = [0.0] * 2
        self.SetTags("SumSSE", 'view:"-" inactive:"+" desc:"sum to increment as we go through epoch"')
        self.SumAvgSSE = [0.0] * 2
        self.SetTags("SumAvgSSE", 'view:"-" inactive:"+" desc:"sum to increment as we go through epoch"')
        self.SumCosDiff = [0.0] * 2
        self.SetTags("SumCosDiff", 'view:"-" inactive:"+" desc:"sum to increment as we go through epoch"')
        self.Win = 0
        self.SetTags("Win", 'view:"-" desc:"main GUI window"')
        self.NetView = 0
        self.SetTags("NetView", 'view:"-" desc:"the network viewer"')
        self.ToolBar = 0
        self.SetTags("ToolBar", 'view:"-" desc:"the master toolbar"')
        self.TrnEpcPlot = 0
        self.SetTags("TrnEpcPlot", 'view:"-" desc:"the training epoch plot"')
        self.TstEpcPlot = 0
        self.SetTags("TstEpcPlot", 'view:"-" desc:"the testing epoch plot"')
        self.TrnTrlPlot = 0
        self.SetTags("TrnTrlPlot", 'view:"-" desc:"the train-trial plot"')
        self.TstTrlPlot = 0
        self.SetTags("TstTrlPlot", 'view:"-" desc:"the test-trial plot"')
        self.SentProbeClustPlot = 0
        self.SetTags("SentProbeClustPlot", 'view:"-" desc:"the probe cluster plot"')
        self.NounProbeClustPlot = 0
        self.SetTags("NounProbeClustPlot", 'view:"-" desc:"the probe cluster plot"')
        self.RunPlot = 0
        self.SetTags("RunPlot", 'view:"-" desc:"the run plot"')
        self.TrnEpcFile = 0
        self.SetTags("TrnEpcFile", 'view:"-" desc:"log file"')
        self.RunFile = 0
        self.SetTags("RunFile", 'view:"-" desc:"log file"')
        self.ValsTsrs = {}
        self.SetTags("ValsTsrs", 'view:"-" desc:"for holding layer values"')
        self.SaveWts = False
        self.SetTags("SaveWts", 'view:"-" desc:"for command-line run only, auto-save final weights after each run"')
        self.NoGui = False
        self.SetTags("NoGui", 'view:"-" desc:"if true, runing in no GUI mode"')
        self.LogSetParams = False
        self.SetTags("LogSetParams", 'view:"-" desc:"if true, print message for all params that are set"')
        self.IsRunning = False
        self.SetTags("IsRunning", 'view:"-" desc:"true if sim is running"')
        self.StopNow = False
        self.SetTags("StopNow", 'view:"-" desc:"flag to stop running"')
        self.NeedsNewRun = False
        self.SetTags("NeedsNewRun", 'view:"-" desc:"flag to initialize NewRun if last one finished"')
        self.RndSeed = int(10)
        self.SetTags("RndSeed", 'view:"-" desc:"the current random seed"')
        self.LastEpcTime = 0
        self.SetTags("LastEpcTime", 'view:"-" desc:"timer for last epoch"')
        self.vp  = 0
        self.SetTags("vp", 'view:"-" desc:"viewport"')

    def InitParams(ss):
        """
        Sets the default set of parameters -- Base is always applied, and others can be optionally
        selected to apply on top of that
        """
        ss.Params.OpenJSON("sg.params")

    def Config(ss):
        """
        Config configures all the elements using the standard functions
        """
        ss.InitParams()
        ss.ConfigEnv()
        ss.ConfigNet(ss.Net)
        ss.ConfigTrnEpcLog(ss.TrnEpcLog)
        ss.ConfigTstEpcLog(ss.TstEpcLog)
        ss.ConfigTrnTrlLog(ss.TrnTrlLog)
        ss.ConfigTstTrlLog(ss.TstTrlLog)
        ss.ConfigSentProbeTrlLog(ss.SentProbeTrlLog)
        ss.ConfigNounProbeTrlLog(ss.NounProbeTrlLog)
        ss.ConfigRunLog(ss.RunLog)

    def ConfigEnv(ss):
        if ss.MaxRuns == 0:
            ss.MaxRuns = 1
        if ss.MaxEpcs == 0: # allow user override
            ss.MaxEpcs = 500
            ss.NZeroStop = 5

        ss.TrainEnv.Nm = "TrainEnv"
        ss.TrainEnv.Dsc = "training params and state"
        ss.TrainEnv.Seq.Max = 100 # sequences per epoch training
        ss.TrainEnv.Rules.OpenRulesPy("sg_rules.txt")
        ss.TrainEnv.PPassive = 0.2
        ss.TrainEnv.Words = sg_words.SGWords
        ss.TrainEnv.Roles = sg_words.SGRoles
        ss.TrainEnv.Fillers = sg_words.SGFillers
        ss.TrainEnv.WordTrans = sg_words.SGWordTrans
        ss.TrainEnv.AmbigVerbs = sg_words.SGAmbigVerbs
        ss.TrainEnv.AmbigNouns = sg_words.SGAmbigNouns
        ss.TrainEnv.Validate()
        ss.TrainEnv.Run.Max = ss.MaxRuns # note: we are not setting epoch max -- do that manually

        ss.TestEnv.Nm = "TestEnv"
        ss.TestEnv.Dsc = "testing params and state"
        ss.TestEnv.Seq.Max = 14
        ss.TestEnv.Rules.OpenRulesPy("sg_tests.txt")
        ss.TestEnv.PPassive = 0 # passive explicitly marked
        ss.TestEnv.Words = sg_words.SGWords
        ss.TestEnv.Roles = sg_words.SGRoles
        ss.TestEnv.Fillers = sg_words.SGFillers
        ss.TestEnv.WordTrans = sg_words.SGWordTrans
        ss.TestEnv.AmbigVerbs = sg_words.SGAmbigVerbs
        ss.TestEnv.AmbigNouns = sg_words.SGAmbigNouns
        ss.TestEnv.Validate()

        ss.SentProbeEnv.Nm = "SentProbeEnv"
        ss.SentProbeEnv.Dsc = "probe params and state"
        ss.SentProbeEnv.Seq.Max = 17
        ss.SentProbeEnv.Rules.OpenRulesPy("sg_probes.txt")
        ss.SentProbeEnv.PPassive = 0 # passive explicitly marked
        ss.SentProbeEnv.Words = sg_words.SGWords
        ss.SentProbeEnv.Roles = sg_words.SGRoles
        ss.SentProbeEnv.Fillers = sg_words.SGFillers
        ss.SentProbeEnv.WordTrans = sg_words.SGWordTrans
        ss.SentProbeEnv.AmbigVerbs = sg_words.SGAmbigVerbs
        ss.SentProbeEnv.AmbigNouns = sg_words.SGAmbigNouns
        ss.SentProbeEnv.Validate()

        ss.NounProbeEnv.Nm = "NounProbeEnv"
        ss.NounProbeEnv.Dsc = "probe params and state"
        ss.NounProbeEnv.Words = sg_words.SGWords
        ss.NounProbeEnv.Validate()

        ss.TrainEnv.Init(0)
        ss.TestEnv.Init(0)
        ss.SentProbeEnv.Init(0)
        ss.NounProbeEnv.Init(0)

    def ConfigNet(ss, net):
        # overall strategy:
        #
        # Encode does pure prediction of next word, which remains about 60% correct at best
        # Gestalt gets direct word input, does full error-driven fill-role learning
        # via decoder.
        #
        # Gestalt can be entirely independent of encode, or recv encode -- testing value.
        # GestaltD depends *critically* on getting direct error signal from Decode!
        #
        # For pure predictive encoder, EncodeP -> Gestalt is bad.  if we leak Decode
        # error signal back to Encode, then it is actually useful, as is GestaltD -> EncodeP
        #
        # run notes:
        # 54 = no enc <-> gestalt -- not much diff..  probably just get rid of enc then?
        # 48 = enc -> gestalt still, no inp -> gest
        # 44 = gestd -> encd, otherwise same as 48 -- improves inp pred due to leak via gestd, else fill same
        # 43 = best perf overall -- 44 + gestd -> inp  -- inp a bit better
        #

        net.InitName(net, "SentGestalt")
        inl = net.AddLayer2D("Input", 10, 5, emer.Input)
        role = net.AddLayer2D("Role", 9, 1, emer.Input)
        fill = net.AddLayer2D("Filler", 11, 5, emer.Target)
        el = deep.AddDeep2DPy(net.AsLeabra(), "Encode", 12, 12) # 12x12 better..
        enc = el[0]
        encct = el[1]
        encp = el[2]
        enc.SetClass("Encode")
        encct.SetClass("Encode")
        dec = net.AddLayer2D("Decode", 12, 12, emer.Hidden)
        el = deep.AddDeepNoTRC2DPy(net.AsLeabra(), "Gestalt", 12, 12) # 12x12 def better with full
        gest = el[0]
        gestct = el[1]
        gest.SetClass("Gestalt")
        gestct.SetClass("Gestalt")

        encp.Shape().CopyShape(inl.Shape())
        deep.TRCLayer(encp).Drivers.AddOne("Input")

        encp.SetRelPos(relpos.Rel(Rel= relpos.RightOf, Other= "Input", YAlign= relpos.Front, Space= 2))
        role.SetRelPos(relpos.Rel(Rel= relpos.RightOf, Other= "EncodeP", YAlign= relpos.Front, Space= 4))
        fill.SetRelPos(relpos.Rel(Rel= relpos.RightOf, Other= "Role", YAlign= relpos.Front, Space= 4))
        enc.SetRelPos(relpos.Rel(Rel= relpos.Above, Other= "Input", YAlign= relpos.Front, XAlign= relpos.Left))
        encct.SetRelPos(relpos.Rel(Rel= relpos.RightOf, Other= "Encode", YAlign= relpos.Front, Space= 2))
        dec.SetRelPos(relpos.Rel(Rel= relpos.RightOf, Other= "EncodeCT", YAlign= relpos.Front, Space= 2))
        gest.SetRelPos(relpos.Rel(Rel= relpos.Above, Other= "Encode", YAlign= relpos.Front, XAlign= relpos.Left))
        gestct.SetRelPos(relpos.Rel(Rel= relpos.RightOf, Other= "Gestalt", YAlign= relpos.Front, Space= 2))

        full = prjn.NewFull()

        pj = net.ConnectLayers(inl, enc, full, emer.Forward)
        pj.SetClass("FmInput")

        pj = net.ConnectLayers(inl, gest, full, emer.Forward) # this is key -- skip encoder
        pj.SetClass("FmInput")

        pj = encct.RecvPrjns().SendName("EncodeP")
        pj.SetClass("EncodePToCT")
        pj = enc.RecvPrjns().SendName("EncodeP")
        pj.SetClass("EncodePToSuper")

        # gestd gets error from Filler, this communicates Filler to encd -> corrupts prediction
        # net.ConnectLayers(gestd, encd, full, emer.Forward)

        # testing no use of enc at all
        net.BidirConnectLayersPy(enc, gest, full)

        net.ConnectLayers(gestct, enc, full, emer.Back) # give enc the best of gestd
        # net.ConnectLayers(gestd, gest, full, emer.Back) // not essential?  todo retest

        # this allows current role info to propagate back to input prediction
        # does not seem to be important
        # net.ConnectLayers(gestd, inp, full, emer.Forward) // must be weaker..

        # if gestd not driving inp, then this is bad -- .005 MIGHT be tiny bit beneficial but not worth it
        # pj = net.ConnectLayers(inp, gestd, full, emer.Back) // these enable prediction
        # pj.SetClass("EncodePToGestalt")
        # pj = net.ConnectLayers(inp, gest, full, emer.Back)
        # pj.SetClass("EncodePToGestalt")

        net.BidirConnectLayersPy(gest, dec, full)
        net.BidirConnectLayersPy(gestct, dec, full) # bidir is essential here to get error signal
        # directly into context layer -- has rel of 0.2

        # net.BidirConnectLayersPy(enc, dec, full) // not beneficial

        net.BidirConnectLayersPy(dec, role, full)
        net.BidirConnectLayersPy(dec, fill, full)

        # add extra deep context
        pj = net.ConnectCtxtToCT(encct, encct, full)
        pj.SetClass("EncSelfCtxt")
        pj = net.ConnectCtxtToCT(inl, encct, full)
        pj.SetClass("CtxtFmInput")

        # add extra deep context
        pj = net.ConnectCtxtToCT(gestct, gestct, full)
        pj.SetClass("GestSelfCtxt")
        pj = net.ConnectCtxtToCT(inl, gestct, full) # yes better
        pj.SetClass("CtxtFmInput")

        net.Defaults()
        ss.SetParams("Network", ss.LogSetParams) # only set Network params
        net.Build()
        net.InitWts()

    def Init(ss):
        """
        Init restarts the run, and initializes everything, including network weights
        and resets the epoch log table
        """
        rand.Seed(ss.RndSeed)
        ss.ConfigEnv()

        ss.StopNow = False
        ss.SetParams("", ss.LogSetParams) # all sheets
        ss.NewRun()
        ss.UpdateView(True)

    def NewRndSeed(ss):
        """
        NewRndSeed gets a new random seed based on current time -- otherwise uses
        the same random seed for every run
        """
        ss.RndSeed = int(datetime.now(timezone.utc).timestamp())

    def Counters(ss, train):
        """
        Counters returns a string of the current counter state
        use tabs to achieve a reasonable formatting overall
        and add a few tabs at the end to allow for expansion..
        """
        if train:
            return "Run:\t%d\tEpoch:\t%d\tSeq:\t%d\tTick:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t" % (ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Seq.Cur, ss.TrainEnv.Tick.Cur, ss.TrainEnv.Trial.Cur, ss.Time.Cycle, ss.TrainEnv.String())
        else:
            return "Run:\t%d\tEpoch:\t%d\tSeq:\t%d\tTick:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t" % (ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TestEnv.Seq.Cur, ss.TestEnv.Tick.Cur, ss.TestEnv.Trial.Cur, ss.Time.Cycle, ss.TestEnv.String())

    def UpdateView(ss, train):
        if ss.NetView != 0 and ss.NetView.IsVisible():
            ss.NetView.Record(ss.Counters(train))

            ss.NetView.GoUpdate()

    def AlphaCyc(ss, train):
        """
        AlphaCyc runs one alpha-cycle (100 msec, 4 quarters)             of processing.
        External inputs must have already been applied prior to calling,
        using ApplyExt method on relevant layers (see TrainTrial, TestTrial).

        If train is true, then learning DWt or WtFmDWt calls are made.
        Handles netview updating within scope of AlphaCycle
        """

        if ss.Win != 0:
            ss.Win.PollEvents() # this is essential for GUI responsiveness while running
        viewUpdt = ss.TrainUpdt.value
        if not train:
            viewUpdt = ss.TestUpdt.value

        if train:
            ss.Net.WtFmDWt()

        ss.Net.AlphaCycInit()
        ss.Time.AlphaCycStart()
        for qtr in range(4):
            for cyc in range(ss.Time.CycPerQtr):
                ss.Net.Cycle(ss.Time)
                ss.Time.CycleInc()
                if ss.ViewOn:
                    if viewUpdt == leabra.Cycle:
                        if cyc != ss.Time.CycPerQtr-1: # will be updated by quarter
                            ss.UpdateView(train)
                    if viewUpdt == leabra.FastSpike:
                        if (cyc+1)%10 == 0:
                            ss.UpdateView(train)
            ss.Net.QuarterFinal(ss.Time)
            ss.Time.QuarterInc()
            if ss.ViewOn:
                if viewUpdt <= leabra.Quarter:
                    ss.UpdateView(train)
                if viewUpdt == leabra.Phase:
                    if qtr >= 2:
                        ss.UpdateView(train)

        if train:
            # if ss.TrainEnv.Tick.Cur > 0 { // first unlearnable
            ss.Net.DWt()
            # }

        if ss.ViewOn and viewUpdt == leabra.AlphaCycle:
            ss.UpdateView(train)

    def ApplyInputs(ss, en):
        """
        ApplyInputs applies input patterns from given environment.
        It is good practice to have this be a separate method with appropriate
        args so that it can be used for various different contexts
        (training, testing, etc).
        """
        ss.Net.InitExt()

        lays = go.Slice_string(["Input", "Role", "Filler"])
        for lnm in lays :
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            pats = en.State(ly.Nm)
            if pats != 0:
                ly.ApplyExt(pats)

    def ApplyInputsProbe(ss, en):
        """
        ApplyInputsProbe applies input patterns from given environment.
        It is good practice to have this be a separate method with appropriate
        args so that it can be used for various different contexts
        (training, testing, etc).
        """
        ss.Net.InitExt()

        lays = go.Slice_string(["Input"])
        for lnm in lays :
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            pats = en.State(ly.Nm)
            if pats != 0:
                ly.ApplyExt(pats)

    def RunEnd(ss):
        """
        RunEnd is called at the end of a run -- save weights, record final log, etc here
        """
        ss.LogRun(ss.RunLog)
        if ss.SaveWts:
            fnm = ss.WeightsFileName()
            print("Saving Weights to: %s\n" % fnm)
            ss.Net.SaveWtsJSON(gi.FileName(fnm))

    def NewRun(ss):
        """
        NewRun intializes a new run of the model, using the TrainEnv.Run counter
        for the new run value
        """
        run = ss.TrainEnv.Run.Cur
        ss.TrainEnv.Init(run)
        ss.TestEnv.Init(run)
        ss.Time.Reset()
        ss.Net.InitWts()
        ss.InitStats()
        ss.TrnEpcLog.SetNumRows(0)
        ss.TstEpcLog.SetNumRows(0)
        ss.NeedsNewRun = False

    def InitStats(ss):
        """
        InitStats initializes all the statistics, especially important for the
        cumulative epoch stats -- called at start of new run
        """

        ss.FirstZero = -1
        ss.NZero = 0
        for i in range(2):
            ss.SumN[i] = 0
            ss.SumErr[i] = 0
            ss.SumSSE[i] = 0
            ss.SumAvgSSE[i] = 0
            ss.SumCosDiff[i] = 0

            ss.TrlErr[i] = 0
            ss.TrlSSE[i] = 0
            ss.TrlAvgSSE[i] = 0
            ss.EpcSSE[i] = 0
            ss.EpcAvgSSE[i] = 0
            ss.EpcPctErr[i] = 0
            ss.EpcCosDiff[i] = 0

    def ActiveUnitNames(ss, lnm, nms, thr):
        """
        ActiveUnitNames reports names of units ActM active > thr, using list of names for units
        """
        acts = []
        ly = leabra.Layer(ss.Net.LayerByName(lnm))
        nn = len(ly.Neurons)
        for ni in range(nn):
            nrn = ly.Neurons[ni]
            if nrn.ActM > thr:
                acts.append(nms[ni])
        return acts

    def TrialStats(ss, accum):
        """
        TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
        accum is true.  Note that we're accumulating stats here on the Sim side so the
        core algorithm side remains as simple as possible, and doesn't need to worry about
        different time-scales over which stats could be accumulated etc.
        You can also aggregate directly from log data, as is done for testing stats
        """
        for li, lnm in enumerate(ss.StatLayNms):
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            ss.TrlCosDiff[li] = float(ly.CosDiff.Cos)
            sse = ly.SSE(0.5)
            avgsse = sse / len(ly.Neurons)
            ss.TrlSSE[li] = sse
            ss.TrlAvgSSE[li] = avgsse
            if ss.TrlSSE[li] > 0:
                ss.TrlErr[li] = 1
            else:
                ss.TrlErr[li] = 0
            if lnm == "EncodeP" and ss.TrainEnv.Tick.Cur == 1:
                ss.TrlCosDiff[li] = 0
                ss.TrlSSE[li] = 0
                ss.TrlAvgSSE[li] = 0
                ss.TrlErr[li] = 0
            if accum:
                if ss.TrainEnv.Tick.Cur == 0 and li == 0:
                    continue
                ss.SumN[li] += 1
                ss.SumErr[li] += ss.TrlErr[li]
                ss.SumSSE[li] += ss.TrlSSE[li]
                ss.SumAvgSSE[li] += ss.TrlAvgSSE[li]
                ss.SumCosDiff[li] += ss.TrlCosDiff[li]
            if lnm == "Filler":
                ss.TrlOut = ", ".join(ss.ActiveUnitNames(lnm, ss.TrainEnv.Fillers, .2))
            if lnm == "EncodeP":
                ss.TrlPred = ", ".join(ss.ActiveUnitNames(lnm, ss.TrainEnv.Words, .2))

    def TrainTrial(ss):
        """
        TrainTrial runs one trial of training using TrainEnv
        """
        if ss.NeedsNewRun:
            ss.NewRun()

        ss.TrainEnv.Step()

        epc = ss.TrainEnv.CounterCur(env.Epoch)
        chg = ss.TrainEnv.CounterChg(env.Epoch)
        if chg:
            ss.LogTrnEpc(ss.TrnEpcLog)
            ss.LrateSched(epc)
            ss.TrainEnv.Trial.Cur = 0
            if ss.ViewOn and ss.TrainUpdt.value > leabra.AlphaCycle:
                ss.UpdateView(True)
            if ss.TestInterval > 0 and epc%ss.TestInterval == 0: # note: epc is *next* so won't trigger first time
                ss.TestAll()
            if epc >= ss.MaxEpcs or (ss.NZeroStop > 0 and ss.NZero >= ss.NZeroStop):
                # done with training..
                ss.RunEnd()
                if ss.TrainEnv.Run.Incr(): # we are done!
                    ss.StopNow = True
                    return
                else:
                    ss.NeedsNewRun = True
                    return

        fill = leabra.Layer(ss.Net.LayerByName("Filler"))
        fill.SetType(emer.Target)

        ss.ApplyInputs(ss.TrainEnv)
        ss.AlphaCyc(True)   # train
        ss.TrialStats(True) # accumulate
        ss.LogTrnTrl(ss.TrnTrlLog)

    def TrainSeq(ss):
        """
        TrainSeq runs training trials for remainder of this sequence
        """
        ss.StopNow = False
        curSeq = ss.TrainEnv.Seq.Cur
        while True:
            ss.TrainTrial()
            if ss.StopNow or ss.TrainEnv.Seq.Cur != curSeq:
                break
        ss.Stopped()

    def TrainEpoch(ss):
        """
        TrainEpoch runs training trials for remainder of this epoch
        """
        ss.StopNow = False
        curEpc = ss.TrainEnv.Epoch.Cur
        while True:
            ss.TrainTrial()
            if ss.StopNow or ss.TrainEnv.Epoch.Cur != curEpc:
                break
        ss.Stopped()

    def TrainRun(ss):
        """
        TrainRun runs training trials for remainder of run
        """
        ss.StopNow = False
        curRun = ss.TrainEnv.Run.Cur
        while True:
            ss.TrainTrial()
            if ss.StopNow or ss.TrainEnv.Run.Cur != curRun:
                break
        ss.Stopped()

    def LrateSched(ss, epc):
        """
        LrateSched implements the learning rate schedule
        also CT self-context strength schedule!
        """
        if epc == 200:
            ss.Net.LrateMult(0.5)
            print("dropped lrate 0.5 at epoch: %d\n" % epc)
        if epc == 300:
            ss.Net.LrateMult(0.2)
            print("dropped lrate 0.2 at epoch: %d\n" % epc)
        if epc == 400:
            ss.Net.LrateMult(0.1)
            print("dropped lrate 0.1 at epoch: %d\n" % epc)

    def Train(ss):
        """
        Train runs the full training from this point onward
        """
        ss.StopNow = False
        while True:
            ss.TrainTrial()
            if ss.StopNow:
                break
        ss.Stopped()

    def Stop(ss):
        """
        Stop tells the sim to stop running
        """
        ss.StopNow = True

    def Stopped(ss):
        """
        Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
        """
        ss.IsRunning = False
        if ss.Win != 0:
            vp = ss.Win.WinViewport2D()
            if ss.ToolBar != 0:
                ss.ToolBar.UpdateActions()
            vp.SetNeedsFullRender()
            ss.UpdateClassView()

    def OpenWts(ss):
        """
        OpenWts opens trained weights
        """
        ss.Net.OpenWtsJSON("trained.wts")

    def SaveWeights(ss, filename):
        """
        SaveWeights saves the network weights -- when called with giv.CallMethod
        it will auto-prompt for filename
        """
        ss.Net.SaveWtsJSON(filename)

    def InitTest(ss):
        """
        InitTest initializes testing state
        """
        rand.Seed(ss.RndSeed)
        ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
        ss.TstTrlLog.SetNumRows(0)
        ss.Net.InitActs()
        ss.UpdateView(False)

    def TestTrial(ss, returnOnChg):
        """
        TestTrial runs one trial of testing -- always sequentially presented inputs
        """
        ss.TestEnv.Step()

        chg = ss.TestEnv.CounterChg(env.Epoch)
        if chg:
            if ss.ViewOn and ss.TestUpdt.value > leabra.AlphaCycle:
                ss.UpdateView(False)
            ss.LogTstEpc(ss.TstEpcLog)
            if returnOnChg:
                return

        fill = leabra.Layer(ss.Net.LayerByName("Filler"))
        fill.SetType(emer.Compare)

        ss.ApplyInputs(ss.TestEnv)
        ss.AlphaCyc(False)
        ss.TrialStats(False)
        ss.LogTstTrl(ss.TstTrlLog)

    def TestSeq(ss):
        """
        TestSeq runs testing trials for remainder of this sequence
        """
        ss.StopNow = False
        curSeq = ss.TestEnv.Seq.Cur
        while True:
            ss.TestTrial(True)
            if ss.StopNow or ss.TestEnv.Seq.Cur != curSeq:
                break
        ss.Stopped()

    def TestAll(ss):
        """
        TestAll runs through the full set of testing items
        """
        ss.InitTest()
        while True:
            ss.TestTrial(True)
            chg = ss.TestEnv.CounterChg(env.Epoch)
            if chg or ss.StopNow:
                break

    def RunTestAll(ss):
        """
        RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
        """
        ss.StopNow = False
        ss.TestAll()
        ss.Stopped()

    def ProbeAll(ss):
        """
        ProbeAll runs all probes
        """
        ss.InitTest()
        ss.SentProbeEnv.Init(ss.TrainEnv.Run.Cur)
        ss.SentProbeTrlLog.SetNumRows(0)

        fill = leabra.Layer(ss.Net.LayerByName("Filler"))
        fill.SetType(emer.Compare)

        while True:
            ss.SentProbeEnv.Step()
            if ss.SentProbeEnv.Seq.Cur == 0:
                break
            ss.ApplyInputs(ss.SentProbeEnv)
            ss.AlphaCyc(False)
            ss.TrialStats(False)
            ss.LogSentProbeTrl(ss.SentProbeTrlLog)

        ss.NounProbeEnv.Init(ss.TrainEnv.Run.Cur)
        ss.NounProbeTrlLog.SetNumRows(0)
        epc = ss.NounProbeEnv.Epoch.Cur
        while True:
            ss.NounProbeEnv.Step()
            if ss.NounProbeEnv.Epoch.Cur != epc:
                break
            ss.Net.InitActs()
            ss.ApplyInputsProbe(ss.NounProbeEnv)
            ss.AlphaCyc(False)
            ss.TrialStats(False)
            ss.LogNounProbeTrl(ss.NounProbeTrlLog)
        ss.ProbeClusterPlot()

    def ParamsName(ss):
        """
        ParamsName returns name of current set of parameters
        """
        if ss.ParamSet == "":
            return "Base"
        return ss.ParamSet

    def SetParams(ss, sheet, setMsg):
        """
        SetParams sets the params for "Base" and then current ParamSet.
        If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
        otherwise just the named sheet
        if setMsg = true then we output a message for each param that was set.
        """
        if sheet == "":

            ss.Params.ValidateSheets(go.Slice_string(["Network", "Sim"]))
        ss.SetParamsSet("Base", sheet, setMsg)
        if ss.ParamSet != "" and ss.ParamSet != "Base":
            sps = ss.ParamSet.split()
            for ps in sps:
                ss.SetParamsSet(ps, sheet, setMsg)

    def SetParamsSet(ss, setNm, sheet, setMsg):
        """
        SetParamsSet sets the params for given params.Set name.
        If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
        otherwise just the named sheet
        if setMsg = true then we output a message for each param that was set.
        """
        pset = ss.Params.SetByNameTry(setNm)
        if sheet == "" or sheet == "Network":
            if "Network" in pset.Sheets:
                netp = pset.SheetByNameTry("Network")
                ss.Net.ApplyParams(netp, setMsg)
        if sheet == "" or sheet == "Sim":
            if "Sim" in pset.Sheets:
                simp= pset.SheetByNameTry("Sim")
                pyparams.ApplyParams(ss, simp, setMsg)

    def ValsTsr(ss, name):
        """
        ValsTsr gets value tensor of given name, creating if not yet made
        """
        if name in ss.ValsTsrs:
            return ss.ValsTsrs[name]
        tsr = etensor.Float32()
        ss.ValsTsrs[name] = tsr
        return tsr

    def RunName(ss):
        """
        RunName returns a name for this run that combines Tag and Params -- add this to
        any file names that are saved.
        """
        if ss.Tag != "":
            return ss.Tag + "_" + ss.ParamsName()
        else:
            return ss.ParamsName()

    def RunEpochName(ss, run, epc):
        """
        RunEpochName returns a string with the run and epoch numbers with leading zeros, suitable
        for using in weights file names.  Uses 3, 5 digits for each.
        """
        return "%03d_%05d" % (run, epc)

    def WeightsFileName(ss):
        """
        WeightsFileName returns default current weights file name
        """
        return ss.Net.Nm + "_" + ss.RunName() + "_" + ss.RunEpochName(ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur) + ".wts.gz"

    def LogFileName(ss, lognm):
        """
        LogFileName returns default log file name
        """
        return ss.Net.Nm + "_" + ss.RunName() + "_" + lognm + ".tsv"

    def LogTrnTrl(ss, dt):
        """
        LogTrnTrl adds data from current trial to the TrnTrlLog table.
        log always contains number of testing items
        """
        epc = ss.TrainEnv.Epoch.Prv
        trl = ss.TrainEnv.Trial.Cur
        row = trl

        if trl == 0:
            dt.SetNumRows(0)

        if dt.Rows <= row:
            dt.SetNumRows(row + 1)

        cur = ss.TrainEnv.CurInputs()

        dt.SetCellFloat("Run", row, float(ss.TrainEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellFloat("Seq", row, float(ss.TrainEnv.Seq.Prv))
        dt.SetCellFloat("Tick", row, float(ss.TrainEnv.Tick.Cur))
        dt.SetCellFloat("Trial", row, float(trl))
        dt.SetCellString("TrialName", row, ss.TrainEnv.String())
        dt.SetCellString("Input", row, cur[0])
        dt.SetCellString("Pred", row, ss.TrlPred)
        dt.SetCellString("Role", row, cur[1])
        dt.SetCellString("Filler", row, cur[2])
        dt.SetCellString("Output", row, ss.TrlOut)
        dt.SetCellString("QType", row, cur[3])
        dt.SetCellFloat("AmbigVerb", row, float(ss.TrainEnv.NAmbigVerbs))
        dt.SetCellFloat("AmbigNouns", row, min(float(ss.TrainEnv.NAmbigNouns), 1))
        for li, lnm in enumerate(ss.StatNms):
            dt.SetCellFloat(lnm+"Err", row, ss.TrlErr[li])
            dt.SetCellFloat(lnm+"SSE", row, ss.TrlSSE[li])
            dt.SetCellFloat(lnm+"AvgSSE", row, ss.TrlAvgSSE[li])
            dt.SetCellFloat(lnm+"CosDiff", row, ss.TrlCosDiff[li])

        ss.TrnTrlPlot.GoUpdate()

    def ConfigTrnTrlLog(ss, dt):
        dt.SetMetaData("name", "TrnTrlLog")
        dt.SetMetaData("desc", "Record of training per input pattern")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        sch = etable.Schema(
            [etable.Column("Run", etensor.INT64, go.nil, go.nil),
            etable.Column("Epoch", etensor.INT64, go.nil, go.nil),
            etable.Column("Seq", etensor.INT64, go.nil, go.nil),
            etable.Column("Tick", etensor.INT64, go.nil, go.nil),
            etable.Column("Trial", etensor.INT64, go.nil, go.nil),
            etable.Column("TrialName", etensor.STRING, go.nil, go.nil),
            etable.Column("Input", etensor.STRING, go.nil, go.nil),
            etable.Column("Pred", etensor.STRING, go.nil, go.nil),
            etable.Column("Role", etensor.STRING, go.nil, go.nil),
            etable.Column("Filler", etensor.STRING, go.nil, go.nil),
            etable.Column("Output", etensor.STRING, go.nil, go.nil),
            etable.Column("QType", etensor.STRING, go.nil, go.nil),
            etable.Column("AmbigVerb", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("AmbigNouns", etensor.FLOAT64, go.nil, go.nil)]
        )
        for lnm in ss.StatNms :
            sch.append( etable.Column(lnm + "Err", etensor.FLOAT64, go.nil, go.nil))
            sch.append( etable.Column(lnm + "SSE", etensor.FLOAT64, go.nil, go.nil))
            sch.append( etable.Column(lnm + "AvgSSE", etensor.FLOAT64, go.nil, go.nil))
            sch.append( etable.Column(lnm + "CosDiff", etensor.FLOAT64, go.nil, go.nil))
        dt.SetFromSchema(sch, 0)

    def ConfigTrnTrlPlot(ss, plt, dt):
        plt.Params.Title = "Sentence Gestalt Train Trial Plot"
        plt.Params.XAxisCol = "Trial"

        plt.SetTable(dt)

        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Seq", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Tick", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Input", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Pred", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Role", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Filler", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Output", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("QType", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("AmbigVerb", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("AmbigNouns", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)

        for lnm in ss.StatNms :
            plt.SetColParams(lnm+"Err", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
            plt.SetColParams(lnm+"SSE", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
            plt.SetColParams(lnm+"AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
            plt.SetColParams(lnm+"CosDiff", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        return plt

    def HogDead(ss, lnm):
        """
        HogDead computes the proportion of units in given layer name with ActAvg over hog thr
        and under dead threshold
        """
        ly = leabra.Layer(ss.Net.LayerByName(lnm))
        n = float(len(ly.Neurons))
        if n == 0:
            return
        for ni in ly.Neurons :
            nrn = ly.Neurons[ni]
            if nrn.ActAvg > 0.3:
                hog += 1
            elif nrn.ActAvg < 0.01:
                dead += 1
        hog /= n
        dead /= n
        return

    def LogTrnEpc(ss, dt):
        """
        LogTrnEpc adds data from current epoch to the TrnEpcLog table.
        computes epoch averages prior to logging.
        """
        row = dt.Rows
        dt.SetNumRows(row + 1)

        epc = ss.TrainEnv.Epoch.Prv
        nt = float(ss.TrnTrlLog.Rows)

        dt.SetCellFloat("Run", row, float(ss.TrainEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))

        if ss.LastEpcTime.IsZero():
            ss.EpcPerTrlMSec = 0
        else:
            iv = time.Now().Sub(ss.LastEpcTime)
            ss.EpcPerTrlMSec = float(iv) / (nt * float(time.Millisecond))
        ss.LastEpcTime = time.Now()
        dt.SetCellFloat("PerTrlMSec", row, ss.EpcPerTrlMSec)

        for li, lnm in ss.StatNms :
            ss.EpcSSE[li] = ss.SumSSE[li] / ss.SumN[li]
            ss.SumSSE[li] = 0
            ss.EpcAvgSSE[li] = ss.SumAvgSSE[li] / ss.SumN[li]
            ss.SumAvgSSE[li] = 0
            ss.EpcPctErr[li] = float(ss.SumErr[li]) / ss.SumN[li]
            ss.SumErr[li] = 0
            ss.EpcPctCor[li] = 1 - ss.EpcPctErr[li]
            ss.EpcCosDiff[li] = ss.SumCosDiff[li] / ss.SumN[li]
            ss.SumCosDiff[li] = 0
            ss.SumN[li] = 0
            dt.SetCellFloat(lnm+"SSE", row, ss.EpcSSE[li])
            dt.SetCellFloat(lnm+"AvgSSE", row, ss.EpcAvgSSE[li])
            dt.SetCellFloat(lnm+"PctErr", row, ss.EpcPctErr[li])
            dt.SetCellFloat(lnm+"PctCor", row, ss.EpcPctCor[li])
            dt.SetCellFloat(lnm+"CosDiff", row, ss.EpcCosDiff[li])

        ss.LogEpcStats(dt, ss.TrnTrlLog)

        for lnm in ss.LayStatNms :
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            dt.SetCellFloat(ly.Nm+" ActAvg", row, float(ly.Pools[0].ActAvg.ActPAvgEff))
            hog, dead = ss.HogDead(lnm)
            dt.SetCellFloat(ly.Nm+" Hog", row, hog)
            dt.SetCellFloat(ly.Nm+" Dead", row, dead)

        ss.TrnEpcPlot.GoUpdate()
        if ss.TrnEpcFile != 0:
            if ss.TrainEnv.Run.Cur == 0 and epc == 0:
                dt.WriteCSVHeaders(ss.TrnEpcFile, etable.Tab)
            dt.WriteCSVRow(ss.TrnEpcFile, row, etable.Tab)

    def LogEpcStats(ss, dt, trlog):
        """
        LogEpcStats does extra SG epoch-level stats on a given trial table
        """
        row = dt.Rows - 1
        tix = etable.NewIdxView(trlog)
        ambspl = split.GroupBy(tix, go.Slice_string(["AmbigNouns"]))
        qtspl = split.GroupBy(tix, go.Slice_string(["QType"]))

        cols = go.Slice_string(["Err", "SSE", "CosDiff"])
        for lnm in ss.StatNms :
            for cl in cols :
                split.Agg(ambspl, lnm+cl, agg.AggMean)
                split.Agg(qtspl, lnm+cl, agg.AggMean)
        ambst = ambspl.AggsToTable(etable.ColNameOnly)
        ss.TrnTrlAmbStats = ambst
        # rolest := rolespl.AggsToTable(etable.ColNameOnly)
        # ss.TrnTrlRoleStats = rolest

        if ambst != 0 and ambst.Rows == 2:
            for cl in cols :
                dt.SetCellFloat("UnAmbFill"+cl, row, ambst.CellFloat("Fill"+cl, 0))
                dt.SetCellFloat("AmbFill"+cl, row, ambst.CellFloat("Fill"+cl, 1))

        qtst = qtspl.AggsToTable(etable.ColNameOnly)
        ss.TrnTrlQTypStats = qtst
        if qtst != 0 and qtst.Rows == 2:
            for cl in cols :
                dt.SetCellFloat("CurQFill"+cl, row, qtst.CellFloat("Fill"+cl, 0))
                dt.SetCellFloat("RevQFill"+cl, row, qtst.CellFloat("Fill"+cl, 1))

    # for lnm := range ss.StatNms {
    #     for ri, rl := range SGRoles {
    #         for cl := range cols {
    #             dt.SetCellFloat(rl+lnm+cl, row, rolest.CellFloat(lnm+cl, ri))
    #         }
    #     }
    # }

    def ConfigTrnEpcLog(ss, dt):
        dt.SetMetaData("name", "TrnEpcLog")
        dt.SetMetaData("desc", "Record of performance over epochs of training")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        sch = etable.Schema(
            [etable.Column("Run", etensor.INT64, go.nil, go.nil),
            etable.Column("Epoch", etensor.INT64, go.nil, go.nil),
            etable.Column("PerTrlMSec", etensor.FLOAT64, go.nil, go.nil)]
        )

        for lnm in ss.StatNms :
            sch.append( etable.Column(lnm + "SSE", etensor.FLOAT64, go.nil, go.nil))
            sch.append( etable.Column(lnm + "AvgSSE", etensor.FLOAT64, go.nil, go.nil))
            sch.append( etable.Column(lnm + "PctErr", etensor.FLOAT64, go.nil, go.nil))
            sch.append( etable.Column(lnm + "PctCor", etensor.FLOAT64, go.nil, go.nil))
            sch.append( etable.Column(lnm + "CosDiff", etensor.FLOAT64, go.nil, go.nil))

        cols = go.Slice_string(["Err", "SSE", "CosDiff"])
        for cl in cols :
            sch.append( etable.Column("UnAmbFill" + cl, etensor.FLOAT64, go.nil, go.nil))
            sch.append( etable.Column("AmbFill" + cl, etensor.FLOAT64, go.nil, go.nil))
            sch.append( etable.Column("CurQFill" + cl, etensor.FLOAT64, go.nil, go.nil))
            sch.append( etable.Column("RevQFill" + cl, etensor.FLOAT64, go.nil, go.nil))

        # for lnm := range ss.StatNms {
        #     for rl := range SGRoles {
        #         for cl := range cols {
        #             sch.append( etable.Column{rl + lnm + cl, etensor.FLOAT64, nil, nil})
        #         }
        #     }
        # }

        for lnm in ss.LayStatNms :
            sch.append( etable.Column(lnm + " ActAvg", etensor.FLOAT64, go.nil, go.nil))
            sch.append( etable.Column(lnm + " Hog", etensor.FLOAT64, go.nil, go.nil))
            sch.append( etable.Column(lnm + " Dead", etensor.FLOAT64, go.nil, go.nil))
        dt.SetFromSchema(sch, 0)

    def ConfigTrnEpcPlot(ss, plt, dt):
        plt.Params.Title = "Sentence Gestalt Epoch Plot"
        plt.Params.XAxisCol = "Epoch"
        plt.SetTable(dt)

        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("PerTrlMSec", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

        for lnm in ss.StatNms :
            plt.SetColParams(lnm+"SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
            plt.SetColParams(lnm+"AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
            plt.SetColParams(lnm+"PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) # default plot
            plt.SetColParams(lnm+"PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
            plt.SetColParams(lnm+"CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

        cols = go.Slice_string(["Err", "SSE", "CosDiff"])
        for cl in cols :
            plt.SetColParams("UnAmbFill"+cl, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
            plt.SetColParams("AmbFill"+cl, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
            plt.SetColParams("CurQFill"+cl, (cl == "Err"), eplot.FixMin, 0, eplot.FixMax, 1)
            plt.SetColParams("RevQFill"+cl, (cl == "Err"), eplot.FixMin, 0, eplot.FixMax, 1)

        # for lnm := range ss.StatNms {
        #     for rl := range SGRoles {
        #         for cl := range cols {
        #             plt.SetColParams(rl+lnm+cl, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        #         }
        #     }
        # }

        for lnm in ss.LayStatNms :
            plt.SetColParams(lnm+" ActAvg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, .5)
            plt.SetColParams(lnm+" Hog", eplot.Off, eplot.FixMin, 0, eplot.FixMax, .5)
            plt.SetColParams(lnm+" Dead", eplot.Off, eplot.FixMin, 0, eplot.FixMax, .5)
        return plt

    def LogTstTrl(ss, dt):
        """
        LogTstTrl adds data from current trial to the TstTrlLog table.
        log always contains number of testing items
        """
        epc = ss.TrainEnv.Epoch.Prv
        trl = ss.TestEnv.Trial.Cur
        row = dt.Rows

        if dt.Rows <= row:
            dt.SetNumRows(row + 1)

        cur = ss.TestEnv.CurInputs()

        st = ""
        for n in ss.TestEnv.Rules.Fired:
            if n[0] != "Sentences":
                st = n[0]

        dt.SetCellFloat("Run", row, float(ss.TrainEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellFloat("Seq", row, float(ss.TestEnv.Seq.Prv))
        dt.SetCellString("SentType", row, st)
        dt.SetCellFloat("Tick", row, float(ss.TestEnv.Tick.Cur))
        dt.SetCellFloat("Trial", row, float(trl))
        dt.SetCellString("TrialName", row, ss.TestEnv.String())
        dt.SetCellString("Input", row, cur[0])
        dt.SetCellString("Pred", row, ss.TrlPred)
        dt.SetCellString("Role", row, cur[1])
        dt.SetCellString("Filler", row, cur[2])
        dt.SetCellString("Output", row, ss.TrlOut)
        dt.SetCellString("QType", row, cur[3])
        dt.SetCellFloat("AmbigVerb", row, float(ss.TestEnv.NAmbigVerbs))
        dt.SetCellFloat("AmbigNouns", row, min(float(ss.TestEnv.NAmbigNouns), 1))
        for li, lnm in enumerate(ss.StatNms):
            dt.SetCellFloat(lnm+"Err", row, ss.TrlErr[li])
            dt.SetCellFloat(lnm+"SSE", row, ss.TrlSSE[li])
            dt.SetCellFloat(lnm+"AvgSSE", row, ss.TrlAvgSSE[li])
            dt.SetCellFloat(lnm+"CosDiff", row, ss.TrlCosDiff[li])

        # note: essential to use Go version of update when called from another goroutine
        ss.TstTrlPlot.GoUpdate()

    def ConfigTstTrlLog(ss, dt):
        dt.SetMetaData("name", "TstTrlLog")
        dt.SetMetaData("desc", "Record of testing per input pattern")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        sch = etable.Schema(
            [etable.Column("Run", etensor.INT64, go.nil, go.nil),
            etable.Column("Epoch", etensor.INT64, go.nil, go.nil),
            etable.Column("Seq", etensor.INT64, go.nil, go.nil),
            etable.Column("SentType", etensor.STRING, go.nil, go.nil),
            etable.Column("Tick", etensor.INT64, go.nil, go.nil),
            etable.Column("Trial", etensor.INT64, go.nil, go.nil),
            etable.Column("TrialName", etensor.STRING, go.nil, go.nil),
            etable.Column("Input", etensor.STRING, go.nil, go.nil),
            etable.Column("Pred", etensor.STRING, go.nil, go.nil),
            etable.Column("Role", etensor.STRING, go.nil, go.nil),
            etable.Column("Filler", etensor.STRING, go.nil, go.nil),
            etable.Column("Output", etensor.STRING, go.nil, go.nil),
            etable.Column("QType", etensor.STRING, go.nil, go.nil),
            etable.Column("AmbigVerb", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("AmbigNouns", etensor.FLOAT64, go.nil, go.nil)]
        )
        for lnm in ss.StatNms :
            sch.append( etable.Column(lnm + "Err", etensor.FLOAT64, go.nil, go.nil))
            sch.append( etable.Column(lnm + "SSE", etensor.FLOAT64, go.nil, go.nil))
            sch.append( etable.Column(lnm + "AvgSSE", etensor.FLOAT64, go.nil, go.nil))
            sch.append( etable.Column(lnm + "CosDiff", etensor.FLOAT64, go.nil, go.nil))
        dt.SetFromSchema(sch, 0)

    def ConfigTstTrlPlot(ss, plt, dt):
        plt.Params.Title = "Sentence Gestalt Test Trial Plot"
        plt.Params.XAxisCol = "TrialName"
        plt.Params.Type = eplot.Bar
        plt.SetTable(dt)
        plt.Params.XAxisRot = 45

        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Seq", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("SentType", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Tick", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Input", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Pred", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Role", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Filler", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Output", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("QType", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("AmbigVerb", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("AmbigNouns", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

        for lnm in ss.StatNms :
            plt.SetColParams(lnm+"Err", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
            plt.SetColParams(lnm+"SSE", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
            plt.SetColParams(lnm+"AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
            plt.SetColParams(lnm+"CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

        plt.SetColParams("FillErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)

        return plt

    def LogTstEpc(ss, dt):
        row = dt.Rows
        dt.SetNumRows(row + 1)

        trl = ss.TstTrlLog
        tix = etable.NewIdxView(trl)
        epc = ss.TrainEnv.Epoch.Prv # ?

        # note: this shows how to use agg methods to compute summary data from another
        # data table, instead of incrementing on the Sim
        dt.SetCellFloat("Run", row, float(ss.TrainEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))

        for lnm in ss.StatNms :
            dt.SetCellFloat(lnm+"SSE", row, agg.Mean(tix, lnm+"SSE")[0])
            dt.SetCellFloat(lnm+"AvgSSE", row, agg.Mean(tix, lnm+"AvgSSE")[0])
            dt.SetCellFloat(lnm+"PctErr", row, agg.Mean(tix, lnm+"Err")[0])
            dt.SetCellFloat(lnm+"CosDiff", row, agg.Mean(tix, lnm+"CosDiff")[0])

        ss.LogEpcStats(dt, ss.TstTrlLog)

        # note: essential to use Go version of update when called from another goroutine
        ss.TstEpcPlot.GoUpdate()

    def ConfigTstEpcLog(ss, dt):
        dt.SetMetaData("name", "TstEpcLog")
        dt.SetMetaData("desc", "Summary stats for testing trials")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        sch = etable.Schema(
            [etable.Column("Run", etensor.INT64, go.nil, go.nil),
            etable.Column("Epoch", etensor.INT64, go.nil, go.nil)]
        )

        for lnm in ss.StatNms :
            sch.append( etable.Column(lnm + "SSE", etensor.FLOAT64, go.nil, go.nil))
            sch.append( etable.Column(lnm + "AvgSSE", etensor.FLOAT64, go.nil, go.nil))
            sch.append( etable.Column(lnm + "PctErr", etensor.FLOAT64, go.nil, go.nil))
            sch.append( etable.Column(lnm + "CosDiff", etensor.FLOAT64, go.nil, go.nil))

        cols = go.Slice_string(["Err", "SSE", "CosDiff"])
        for cl in cols :
            sch.append( etable.Column("UnAmbFill" + cl, etensor.FLOAT64, go.nil, go.nil))
            sch.append( etable.Column("AmbFill" + cl, etensor.FLOAT64, go.nil, go.nil))
            sch.append( etable.Column("CurQFill" + cl, etensor.FLOAT64, go.nil, go.nil))
            sch.append( etable.Column("RevQFill" + cl, etensor.FLOAT64, go.nil, go.nil))
        dt.SetFromSchema(sch, 0)

    def ConfigTstEpcPlot(ss, plt, dt):
        plt.Params.Title = "Sentence Gestalt Testing Epoch Plot"
        plt.Params.XAxisCol = "Epoch"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

        for lnm in ss.StatNms :
            plt.SetColParams(lnm+"SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
            plt.SetColParams(lnm+"AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
            plt.SetColParams(lnm+"PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) # default plot
            plt.SetColParams(lnm+"CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

        cols = go.Slice_string(["Err", "SSE", "CosDiff"])
        for cl in cols :
            plt.SetColParams("UnAmbFill"+cl, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
            plt.SetColParams("AmbFill"+cl, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
            plt.SetColParams("CurQFill"+cl, (cl == "Err"), eplot.FixMin, 0, eplot.FixMax, 1)
            plt.SetColParams("RevQFill"+cl, (cl == "Err"), eplot.FixMin, 0, eplot.FixMax, 1)
        return plt

    def LogSentProbeTrl(ss, dt):
        """
        LogSentProbeTrl adds data from current trial to the SentProbeTrlLog table.
    # this is triggered by increment so use previous value
        log always contains number of testing items
        """
        epc = ss.TrainEnv.Epoch.Prv
        trl = ss.SentProbeEnv.Trial.Cur
        row = dt.Rows

        if dt.Rows <= row:
            dt.SetNumRows(row + 1)

        cur = ss.SentProbeEnv.CurInputs()
        st = ""
        for n, _ in ss.SentProbeEnv.Rules.Fired :
            if n != "Sentences":
                st = n

        dt.SetCellFloat("Run", row, float(ss.SentProbeEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellFloat("Seq", row, float(ss.SentProbeEnv.Seq.Prv))
        dt.SetCellString("SentType", row, st)
        dt.SetCellFloat("Tick", row, float(ss.SentProbeEnv.Tick.Cur))
        dt.SetCellFloat("Trial", row, float(trl))
        dt.SetCellString("TrialName", row, ss.SentProbeEnv.String())
        dt.SetCellString("Input", row, cur[0])
        dt.SetCellString("Pred", row, ss.TrlPred)
        dt.SetCellString("Role", row, cur[1])
        dt.SetCellString("Filler", row, cur[2])
        dt.SetCellString("Output", row, ss.TrlOut)
        dt.SetCellString("QType", row, cur[3])

        for lnm in ss.ProbeNms :
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            vt = ss.ValsTsr(lnm)
            ly.UnitValsTensor(vt, "ActM")
            dt.SetCellTensor(lnm, row, vt)

    def ConfigSentProbeTrlLog(ss, dt):
        dt.SetMetaData("name", "SentProbeTrlLog")
        dt.SetMetaData("desc", "Record of testing per input pattern")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        sch = etable.Schema(
            [etable.Column("Run", etensor.INT64, go.nil, go.nil),
            etable.Column("Epoch", etensor.INT64, go.nil, go.nil),
            etable.Column("Seq", etensor.INT64, go.nil, go.nil),
            etable.Column("SentType", etensor.STRING, go.nil, go.nil),
            etable.Column("Tick", etensor.INT64, go.nil, go.nil),
            etable.Column("Trial", etensor.INT64, go.nil, go.nil),
            etable.Column("TrialName", etensor.STRING, go.nil, go.nil),
            etable.Column("Input", etensor.STRING, go.nil, go.nil),
            etable.Column("Pred", etensor.STRING, go.nil, go.nil),
            etable.Column("Role", etensor.STRING, go.nil, go.nil),
            etable.Column("Filler", etensor.STRING, go.nil, go.nil),
            etable.Column("Output", etensor.STRING, go.nil, go.nil),
            etable.Column("QType", etensor.STRING, go.nil, go.nil)]
        )
        for lnm in ss.ProbeNms :
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            sch.append( etable.Column(lnm, etensor.FLOAT64, ly.Shp.Shp, go.nil))

        dt.SetFromSchema(sch, 0)

    def LogNounProbeTrl(ss, dt):
        """
        LogNounProbeTrl adds data from current trial to the NounProbeTrlLog table.
    # this is triggered by increment so use previous value
        log always contains number of testing items
        """
        epc = ss.TrainEnv.Epoch.Prv
        trl = ss.NounProbeEnv.Trial.Cur
        row = dt.Rows

        if dt.Rows <= row:
            dt.SetNumRows(row + 1)

        dt.SetCellFloat("Run", row, float(ss.NounProbeEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellFloat("Trial", row, float(trl))
        dt.SetCellString("TrialName", row, ss.NounProbeEnv.String())

        for lnm in ss.ProbeNms :
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            vt = ss.ValsTsr(lnm)
            ly.UnitValsTensor(vt, "ActM")
            dt.SetCellTensor(lnm, row, vt)

    def ConfigNounProbeTrlLog(ss, dt):
        dt.SetMetaData("name", "NounProbeTrlLog")
        dt.SetMetaData("desc", "Record of testing per input pattern")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        sch = etable.Schema(
            [etable.Column("Run", etensor.INT64, go.nil, go.nil),
            etable.Column("Epoch", etensor.INT64, go.nil, go.nil),
            etable.Column("Trial", etensor.INT64, go.nil, go.nil),
            etable.Column("TrialName", etensor.STRING, go.nil, go.nil)]
        )
        for lnm in ss.ProbeNms :
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            sch.append( etable.Column(lnm, etensor.FLOAT64, ly.Shp.Shp, go.nil))

        dt.SetFromSchema(sch, 0)

    def ProbeClusterPlot(ss):
        """
        ProbeClustPlot does cluster plotting of probe data
        """
        stix = etable.NewIdxView(ss.SentProbeTrlLog)
        stix.Filter(FilterTickEq5)
        ss.ClustPlot(ss.SentProbeClustPlot, stix, "Gestalt", "SentType", clust.Contrast)
        ss.SentProbeClustPlot.Update()

        ntix = etable.NewIdxView(ss.NounProbeTrlLog)
        ss.ClustPlot(ss.NounProbeClustPlot, ntix, "Gestalt", "TrialName", clust.Max)
        ss.NounProbeClustPlot.Update()

    def ClustPlot(ss, plt, ix, colNm, lblNm, dfunc):
        """
        ClustPlot does one cluster plot on given table column
        """
        nm = ix.Table.MetaData["name"]
        smat = simat.SimMat()
        smat.TableColStd(ix, colNm, lblNm, False, metric.Euclidean)
        pt = etable.Table()
        clust.Plot(pt, clust.GlomStd(smat, dfunc), smat)
        plt.InitName(plt, colNm)
        plt.Params.Title = "Cluster Plot of: " + nm + " " + colNm
        plt.Params.XAxisCol = "X"
        plt.SetTable(pt)

        plt.SetColParams("X", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Y", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Label", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)

    def LogRun(ss, dt):
        """
        LogRun adds data from current run to the RunLog table.

    # this is NOT triggered by increment yet -- use Cur
        """
        return
        run = ss.TrainEnv.Run.Cur
        row = dt.Rows
        dt.SetNumRows(row + 1)

        epclog = ss.TrnEpcLog
        epcix = etable.NewIdxView(epclog)
        # compute mean over last N epochs for run level
        nlast = 5
        if nlast > epcix.Len()-1:
            nlast = epcix.Len() - 1
        epcix.Idxs = epcix.Idxs[epcix.Len()-nlast:]

        params = ss.RunName() # includes tag

        dt.SetCellFloat("Run", row, float(run))
        dt.SetCellString("Params", row, params)
        dt.SetCellFloat("FirstZero", row, float(ss.FirstZero))
        dt.SetCellFloat("SSE", row, agg.Mean(epcix, "SSE")[0])
        dt.SetCellFloat("AvgSSE", row, agg.Mean(epcix, "AvgSSE")[0])
        dt.SetCellFloat("PctErr", row, agg.Mean(epcix, "PctErr")[0])
        dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
        dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])

        runix = etable.NewIdxView(dt)
        spl = split.GroupBy(runix, go.Slice_string(["Params"]))
        split.Desc(spl, "FirstZero")
        split.Desc(spl, "PctCor")
        ss.RunStats = spl.AggsToTable(etable.AddAggName)

        # note: essential to use Go version of update when called from another goroutine
        ss.RunPlot.GoUpdate()
        if ss.RunFile != 0:
            if row == 0:
                dt.WriteCSVHeaders(ss.RunFile, etable.Tab)
            dt.WriteCSVRow(ss.RunFile, row, etable.Tab)

    def ConfigRunLog(ss, dt):
        dt.SetMetaData("name", "RunLog")
        dt.SetMetaData("desc", "Record of performance at end of training")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        sch = etable.Schema(
            [etable.Column("Run", etensor.INT64, go.nil, go.nil),
            etable.Column("Params", etensor.STRING, go.nil, go.nil),
            etable.Column("FirstZero", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("SSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("AvgSSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("PctErr", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("PctCor", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("CosDiff", etensor.FLOAT64, go.nil, go.nil)]
        )
        dt.SetFromSchema(sch, 0)

    def ConfigRunPlot(ss, plt, dt):
        plt.Params.Title = "Sentence Gestalt Run Plot"
        plt.Params.XAxisCol = "Run"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("FirstZero", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0) # default plot
        plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("PctErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        return plt

    def ConfigNetView(ss, nv):
        nv.ViewDefaults()
        nv.Scene().Camera.Pose.Pos.Set(0, 1.2, 3.0) # more "head on" than default which is more "top down"
        nv.Scene().Camera.LookAt(mat32.Vec3(0, 0, 0), mat32.Vec3(0, 1, 0))

    def ConfigGui(ss):
        """
        ConfigGui configures the GoGi gui interface for this simulation,
        """
        width = 1600
        height = 1200

        gi.SetAppName("SG")
        gi.SetAppAbout('This is the sentence gestalt model, which learns to encode both syntax and semantics of sentences in an integrated "gestalt" hidden layer. The sentences have simple agent-verb-patient structure with optional prepositional or adverb modifier phrase at the end, and can be either in the active or passive form (80% active, 20% passive). There are ambiguous terms that need to be resolved via context, showing a key interaction between syntax and semantics. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch9/sg/README.md">README.md on GitHub</a>.</p>')

        win = gi.NewMainWindow("SG", "Sentence Gestalt", width, height)
        ss.Win = win

        vp = win.WinViewport2D()
        ss.vp = vp
        updt = vp.UpdateStart()

        mfr = win.SetMainFrame()

        tbar = gi.AddNewToolBar(mfr, "tbar")
        tbar.SetStretchMaxWidth()
        ss.ToolBar = tbar

        split = gi.AddNewSplitView(mfr, "split")
        split.Dim = mat32.X
        split.SetStretchMax()

        cv = ss.NewClassView("sv")
        cv.AddFrame(split)
        cv.Config()

        tv = gi.AddNewTabView(split, "tv")

        nv = netview.NetView()
        tv.AddTab(nv, "NetView")
        nv.Var = "Act"
        nv.SetNet(ss.Net)
        ss.NetView = nv
        ss.ConfigNetView(nv)

        plt = eplot.Plot2D()
        tv.AddTab(plt, "TrnEpcPlot")
        ss.TrnEpcPlot = ss.ConfigTrnEpcPlot(plt, ss.TrnEpcLog)

        plt = eplot.Plot2D()
        tv.AddTab(plt, "TrnTrlPlot")
        ss.TrnTrlPlot = ss.ConfigTrnTrlPlot(plt, ss.TrnTrlLog)

        plt = eplot.Plot2D()
        tv.AddTab(plt, "TstTrlPlot")
        ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

        plt = eplot.Plot2D()
        tv.AddTab(plt, "TstEpcPlot")
        ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.TstEpcLog)

        plt = eplot.Plot2D()
        tv.AddTab(plt, "SentProbeClustPlot")
        ss.SentProbeClustPlot = plt

        plt = eplot.Plot2D()
        tv.AddTab(plt, "NounProbeClustPlot")
        ss.NounProbeClustPlot = plt

        plt = eplot.Plot2D()
        tv.AddTab(plt, "RunPlot")
        ss.RunPlot = ss.ConfigRunPlot(plt, ss.RunLog)

        split.SetSplitsList(go.Slice_float32([.2, .8]))
        recv = win.This()

        tbar.AddAction(gi.ActOpts(Label="Init", Icon="update", Tooltip="Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc=UpdtFuncNotRunning), recv, InitCB)

        tbar.AddAction(gi.ActOpts(Label="Train", Icon="run", Tooltip="Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.", UpdateFunc=UpdtFuncNotRunning), recv, TrainCB)
        
        tbar.AddAction(gi.ActOpts(Label="Stop", Icon="stop", Tooltip="Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc=UpdtFuncRunning), recv, StopCB)
        
        tbar.AddAction(gi.ActOpts(Label="Step Trial", Icon="step-fwd", Tooltip="Advances one training trial at a time.", UpdateFunc=UpdtFuncNotRunning), recv, StepTrialCB)
        
        tbar.AddAction(gi.ActOpts(Label="Step Epoch", Icon="fast-fwd", Tooltip="Advances one epoch (complete set of training patterns) at a time.", UpdateFunc=UpdtFuncNotRunning), recv, StepEpochCB)

        tbar.AddAction(gi.ActOpts(Label= "Step Seq", Icon= "fast-fwd", Tooltip= "Advances one sequence (sentence) at a time.", UpdateFunc=UpdtFuncNotRunning), recv, StepSeqCB)
        
        tbar.AddAction(gi.ActOpts(Label="Step Run", Icon="fast-fwd", Tooltip="Advances one full training Run at a time.", UpdateFunc=UpdtFuncNotRunning), recv, StepRunCB)
        
        tbar.AddSeparator("test")
        
        tbar.AddAction(gi.ActOpts(Label= "Open Weights", Icon= "update", Tooltip= "Open weights trained on first phase of training (excluding 'novel' objects)", UpdateFunc=UpdtFuncNotRunning), recv, OpenWeightsCB)

        tbar.AddAction(gi.ActOpts(Label= "Init Test", Icon= "update", Tooltip= "Initialize to start of testing items.", UpdateFunc=UpdtFuncNotRunning), recv, InitTestCB)
        
        tbar.AddAction(gi.ActOpts(Label="Test Trial", Icon="step-fwd", Tooltip="Runs the next testing trial.", UpdateFunc=UpdtFuncNotRunning), recv, TestTrialCB)
        
        tbar.AddAction(gi.ActOpts(Label= "Test Seq", Icon= "fast-fwd", Tooltip= "Advances one sequence (sentence) at a time.", UpdateFunc=UpdtFuncNotRunning), recv, TestSeqCB)
        
        tbar.AddAction(gi.ActOpts(Label="Test All", Icon="fast-fwd", Tooltip="Tests all of the testing trials.", UpdateFunc=UpdtFuncNotRunning), recv, TestAllCB)

        tbar.AddSeparator("log")
        
        tbar.AddAction(gi.ActOpts(Label="Reset TstTrlLog", Icon="reset", Tooltip="Resets the testing trial log, so it is easier to read"), recv, ResetTstTrlLogCB)

        tbar.AddAction(gi.ActOpts(Label= "Probe All", Icon= "fast-fwd", Tooltip= "probe inputs.", UpdateFunc=UpdtFuncNotRunning), recv, ProbeAllCB)
        
        tbar.AddSeparator("misc")
        
        tbar.AddAction(gi.ActOpts(Label="New Seed", Icon="new", Tooltip="Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."), recv, NewRndSeedCB)

        tbar.AddAction(gi.ActOpts(Label="README", Icon="file-markdown", Tooltip="Opens your browser on the README file that contains instructions for how to run this model."), recv, ReadmeCB)

        # main menu
        appnm = gi.AppName()
        mmen = win.MainMenu
        mmen.ConfigMenus(go.Slice_string([appnm, "File", "Edit", "Window"]))

        amen = gi.Action(win.MainMenu.ChildByName(appnm, 0))
        amen.Menu.AddAppMenu(win)

        emen = gi.Action(win.MainMenu.ChildByName("Edit", 1))
        emen.Menu.AddCopyCutPaste(win)

        # note: Command in shortcuts is automatically translated into Control for
        # Linux, Windows or Meta for MacOS
        # fmen := win.MainMenu.ChildByName("File", 0).(*gi.Action)
        # fmen.Menu.AddAction(gi.ActOpts{Label: "Open", Shortcut: "Command+O"},
        #   win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
        #       FileViewOpenSVG(vp)
        #   })
        # fmen.Menu.AddSeparator("csep")
        # fmen.Menu.AddAction(gi.ActOpts{Label: "Close Window", Shortcut: "Command+W"},
        #   win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
        #       win.Close()
        #   })

        win.MainMenuUpdated()
        vp.UpdateEndNoSig(updt)
        win.GoStartEventLoop()

# TheSim is the overall state for this simulation
TheSim = Sim()
 
def main(argv):
    TheSim.Config()
    TheSim.ConfigGui()
    TheSim.Init()
    
main(sys.argv[1:])


