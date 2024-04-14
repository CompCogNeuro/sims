#!/usr/local/bin/pyleabra -i

# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# use:
# pyleabra -i ra25.py
# to run in gui interactive mode from the command line (or pyleabra, import ra25)
# see main function at the end for startup args

# stroop illustrates how the PFC can produce top-down biasing for executive control,
# in the context of the widely-studied Stroop task.

from leabra import (
    go,
    leabra,
    emer,
    relpos,
    eplot,
    env,
    agg,
    patgen,
    prjn,
    etable,
    efile,
    split,
    etensor,
    params,
    netview,
    rand,
    erand,
    gi,
    giv,
    pygiv,
    pyparams,
    math32,
    metric,
    simat,
    pca,
    clust,
)

import importlib as il  # il.reload(ra25) -- doesn't seem to work for reasons unknown
import io, sys, getopt
from datetime import datetime, timezone
from enum import Enum
import numpy as np

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


def TestTrialCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.TestTrial(False)
        TheSim.IsRunning = False
        TheSim.UpdateClassView()
        TheSim.vp.SetNeedsFullRender()


def TestAllCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.ToolBar.UpdateActions()
        TheSim.RunTestAll()


def SOATestTrialCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.SOATestTrial(False)
        TheSim.IsRunning = False
        TheSim.UpdateClassView()
        TheSim.vp.SetNeedsFullRender()


def SOATestAllCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.ToolBar.UpdateActions()
        TheSim.RunSOATestAll()


def ResetTstTrlLogCB(recv, send, sig, data):
    TheSim.TstTrlLog.SetNumRows(0)
    TheSim.TstTrlPlot.Update()


def DefaultsCB(recv, send, sig, data):
    TheSim.Defaults()
    TheSim.Init()
    TheSim.UpdateClassView()
    TheSim.vp.SetNeedsFullRender()


def NewRndSeedCB(recv, send, sig, data):
    TheSim.NewRndSeed()


def ReadmeCB(recv, send, sig, data):
    core.OpenURL(
        "https://github.com/CompCogNeuro/sims/blob/master/ch10/stroop/README.md"
    )


def UpdateFuncNotRunning(act):
    act.SetActiveStateUpdate(not TheSim.IsRunning)


def UpdateFuncRunning(act):
    act.SetActiveStateUpdate(TheSim.IsRunning)


#####################################################
#     Sim


class Sim(pyviews.ClassViewObj):
    """
    Sim encapsulates the entire simulation model, and we define all the
    functionality as methods on this struct.  This structure keeps all relevant
    state information organized and available without having to pass everything around
    as arguments to methods, and provides the core GUI interface (note the view tags
    for the fields which provide hints to how things should be displayed).
    """

    def __init__(self):
        super(Sim, self).__init__()
        self.FmPFC = float(0.3)
        self.SetTags(
            "FmPFC",
            'def:"0.3" step:"0.01" desc:"strength of projection from PFC to Hidden -- reduce to simulate PFC damage"',
        )
        self.DtVmTau = float(30)
        self.SetTags(
            "DtVmTau",
            'def:"30" step:"5" desc:"time constant for updating the network "',
        )
        self.Net = leabra.Network()
        self.SetTags(
            "Net",
            'view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"',
        )
        self.TrainPats = etable.Table()
        self.SetTags("TrainPats", 'view:"no-inline" desc:"training patterns"')
        self.TestPats = etable.Table()
        self.SetTags("TestPats", 'view:"no-inline" desc:"testing patterns"')
        self.SOAPats = etable.Table()
        self.SetTags("SOAPats", 'view:"no-inline" desc:"SOA testing patterns"')
        self.TrnEpcLog = etable.Table()
        self.SetTags(
            "TrnEpcLog", 'view:"no-inline" desc:"training epoch-level log data"'
        )
        self.TstEpcLog = etable.Table()
        self.SetTags(
            "TstEpcLog", 'view:"no-inline" desc:"testing epoch-level log data"'
        )
        self.TstTrlLog = etable.Table()
        self.SetTags(
            "TstTrlLog", 'view:"no-inline" desc:"testing trial-level log data"'
        )
        self.SOATrlLog = etable.Table()
        self.SetTags(
            "SOATrlLog", 'view:"no-inline" desc:"SOA testing trial-level log data"'
        )
        self.RunLog = etable.Table()
        self.SetTags("RunLog", 'view:"no-inline" desc:"summary log of each run"')
        self.RunStats = etable.Table()
        self.SetTags("RunStats", 'view:"no-inline" desc:"aggregate stats on all runs"')
        self.Params = params.Sets()
        self.SetTags("Params", 'view:"no-inline" desc:"full collection of param sets"')
        self.ParamSet = str()
        self.SetTags(
            "ParamSet",
            'view:"-" desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set -- can use multiple names separated by spaces (don\'t put spaces in ParamSet names!)"',
        )
        self.MaxRuns = int(1)
        self.SetTags("MaxRuns", 'desc:"maximum number of model runs to perform"')
        self.MaxEpcs = int(55)
        self.SetTags("MaxEpcs", 'desc:"maximum number of epochs to run per model run"')
        self.TrainEnv = env.FreqTable()
        self.SetTags(
            "TrainEnv",
            'desc:"Training environment -- contains everything about iterating over input / output patterns over training"',
        )
        self.TestEnv = env.FixedTable()
        self.SetTags(
            "TestEnv",
            'desc:"Testing environment for std strooop -- manages iterating over testing"',
        )
        self.SOATestEnv = env.FixedTable()
        self.SetTags(
            "SOATestEnv",
            'desc:"Testing environment for SOA tests -- manages iterating over testing"',
        )
        self.Time = leabra.Time()
        self.SetTags("Time", 'desc:"leabra timing parameters and state"')
        self.ViewOn = True
        self.SetTags(
            "ViewOn", 'desc:"whether to update the network view while running"'
        )
        self.TrainUpdate = leabra.TimeScales.Quarter
        self.SetTags(
            "TrainUpdate",
            'desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"',
        )
        self.TestUpdate = leabra.TimeScales.Cycle
        self.SetTags(
            "TestUpdate",
            'desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"',
        )
        self.TestInterval = int(5)
        self.SetTags(
            "TestInterval",
            'desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"',
        )
        self.TstRecLays = go.Slice_string(
            ["Colors", "Words", "PFC", "Hidden", "Output"]
        )
        self.SetTags(
            "TstRecLays",
            'desc:"names of layers to record activations etc of during testing"',
        )

        # statistics: note use float64 as that is best for etable.Table
        self.TrlErr = float()
        self.SetTags(
            "TrlErr",
            'inactive:"+" desc:"1 if trial was error, 0 if correct -- based on SSE = 0 (subject to .5 unit-wise tolerance)"',
        )
        self.TrlSSE = float()
        self.SetTags("TrlSSE", 'inactive:"+" desc:"current trial\'s sum squared error"')
        self.TrlAvgSSE = float()
        self.SetTags(
            "TrlAvgSSE",
            'inactive:"+" desc:"current trial\'s average sum squared error"',
        )
        self.TrlCosDiff = float()
        self.SetTags(
            "TrlCosDiff", 'inactive:"+" desc:"current trial\'s cosine difference"'
        )
        self.SOA = int()
        self.SetTags("SOA", 'inactive:"+" desc:"current SOA value"')
        self.SOAMaxCyc = int()
        self.SetTags(
            "SOAMaxCyc", 'inactive:"+" desc:"current max cycles value for SOA"'
        )
        self.SOATrlTyp = int()
        self.SetTags("SOATrlTyp", 'inactive:"+" desc:"current trial type for SOA"')
        self.EpcSSE = float()
        self.SetTags(
            "EpcSSE", 'inactive:"+" desc:"last epoch\'s total sum squared error"'
        )
        self.EpcAvgSSE = float()
        self.SetTags(
            "EpcAvgSSE",
            'inactive:"+" desc:"last epoch\'s average sum squared error (average over trials, and over units within layer)"',
        )
        self.EpcPctErr = float()
        self.SetTags(
            "EpcPctErr",
            'inactive:"+" desc:"last epoch\'s percent of trials that had SSE > 0 (subject to .5 unit-wise tolerance)"',
        )
        self.EpcPctCor = float()
        self.SetTags(
            "EpcPctCor",
            'inactive:"+" desc:"last epoch\'s percent of trials that had SSE == 0 (subject to .5 unit-wise tolerance)"',
        )
        self.EpcCosDiff = float()
        self.SetTags(
            "EpcCosDiff",
            'inactive:"+" desc:"last epoch\'s average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"',
        )

        # internal state - view:"-"
        self.SumErr = float()
        self.SetTags(
            "SumErr",
            'view:"-" inactive:"+" desc:"sum to increment as we go through epoch"',
        )
        self.SumSSE = float()
        self.SetTags(
            "SumSSE",
            'view:"-" inactive:"+" desc:"sum to increment as we go through epoch"',
        )
        self.SumAvgSSE = float()
        self.SetTags(
            "SumAvgSSE",
            'view:"-" inactive:"+" desc:"sum to increment as we go through epoch"',
        )
        self.SumCosDiff = float()
        self.SetTags(
            "SumCosDiff",
            'view:"-" inactive:"+" desc:"sum to increment as we go through epoch"',
        )
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
        self.TstTrlPlot = 0
        self.SetTags("TstTrlPlot", 'view:"-" desc:"the test-trial plot"')
        self.SOATrlPlot = 0
        self.SetTags("SOATrlPlot", 'view:"-" desc:"the SOA test-trial plot"')
        self.RunPlot = 0
        self.SetTags("RunPlot", 'view:"-" desc:"the run plot"')
        self.TrnEpcFile = 0
        self.SetTags("TrnEpcFile", 'view:"-" desc:"log file"')
        self.RunFile = 0
        self.SetTags("RunFile", 'view:"-" desc:"log file"')
        self.ValuesTsrs = {}
        self.SetTags("ValuesTsrs", 'view:"-" desc:"for holding layer values"')
        self.IsRunning = False
        self.SetTags("IsRunning", 'view:"-" desc:"true if sim is running"')
        self.StopNow = False
        self.SetTags("StopNow", 'view:"-" desc:"flag to stop running"')
        self.NeedsNewRun = False
        self.SetTags(
            "NeedsNewRun",
            'view:"-" desc:"flag to initialize NewRun if last one finished"',
        )
        self.RndSeed = int(1)
        self.SetTags("RndSeed", 'view:"-" desc:"the current random seed"')
        self.vp = 0
        self.SetTags("vp", 'view:"-" desc:"viewport"')

    def InitParams(ss):
        """
        Sets the default set of parameters -- Base is always applied, and others can be optionally
        selected to apply on top of that
        """
        ss.Params.OpenJSON("stroop.params")
        ss.Defaults()

    def Defaults(ss):
        ss.FmPFC = 0.3
        ss.DtVmTau = 30

    def Config(ss):
        """
        Config configures all the elements using the standard functions
        """
        ss.InitParams()
        ss.OpenPats()
        ss.ConfigEnv()
        ss.ConfigNet(ss.Net)
        ss.ConfigTrnEpcLog(ss.TrnEpcLog)
        ss.ConfigTstEpcLog(ss.TstEpcLog)
        ss.ConfigTstTrlLog(ss.TstTrlLog)
        ss.ConfigSOATrlLog(ss.SOATrlLog)
        ss.ConfigRunLog(ss.RunLog)

    def ConfigEnv(ss):
        if ss.MaxRuns == 0:
            ss.MaxRuns = 1
        if ss.MaxEpcs == 0:  # allow user override
            ss.MaxEpcs = 55

        ss.TrainEnv.Nm = "TrainEnv"
        ss.TrainEnv.Dsc = "training params and state"
        ss.TrainEnv.Table = etable.NewIndexView(ss.TrainPats)
        ss.TrainEnv.NSamples = 1
        # ss.TrainEnv.Validate()
        ss.TrainEnv.Run.Max = (
            ss.MaxRuns
        )  # note: we are not setting epoch max -- do that manually

        ss.TestEnv.Nm = "TestEnv"
        ss.TestEnv.Dsc = "testing params and state"
        ss.TestEnv.Table = etable.NewIndexView(ss.TestPats)
        ss.TestEnv.Sequential = True
        # ss.TestEnv.Validate()

        ss.SOATestEnv.Nm = "SOATestEnv"
        ss.SOATestEnv.Dsc = "test all params and state"
        ss.SOATestEnv.Table = etable.NewIndexView(ss.SOAPats)
        ss.SOATestEnv.Sequential = True
        # ss.SOATestEnv.Validate()

        ss.TrainEnv.Init(0)
        ss.TestEnv.Init(0)
        ss.SOATestEnv.Init(0)

    def ConfigNet(ss, net):
        net.InitName(net, "Stroop")
        clr = net.AddLayer2D("Colors", 1, 2, emer.Input)
        wrd = net.AddLayer2D("Words", 1, 2, emer.Input)
        hid = net.AddLayer4D("Hidden", 1, 2, 1, 2, emer.Hidden)
        pfc = net.AddLayer2D("PFC", 1, 2, emer.Input)
        out = net.AddLayer2D("Output", 1, 2, emer.Target)

        full = prjn.NewFull()
        clr2hid = prjn.NewOneToOne()
        wrd2hid = prjn.NewOneToOne()
        wrd2hid.RecvStart = 2

        pfc2hid = prjn.NewRect()
        pfc2hid.Scale.Set(0.5, 0.5)
        pfc2hid.Size.Set(1, 1)

        net.ConnectLayers(clr, hid, clr2hid, emer.Forward)
        net.ConnectLayers(wrd, hid, wrd2hid, emer.Forward)
        net.ConnectLayers(pfc, hid, pfc2hid, emer.Back)
        net.BidirConnectLayersPy(hid, out, full)

        wrd.SetRelPos(
            relpos.Rel(Rel=relpos.RightOf, Other="Colors", YAlign=relpos.Front, Space=1)
        )
        out.SetRelPos(
            relpos.Rel(Rel=relpos.RightOf, Other="Words", YAlign=relpos.Front, Space=1)
        )
        hid.SetRelPos(
            relpos.Rel(
                Rel=relpos.Above,
                Other="Colors",
                YAlign=relpos.Front,
                XAlign=relpos.Left,
                YOffset=1,
            )
        )
        pfc.SetRelPos(
            relpos.Rel(Rel=relpos.RightOf, Other="Hidden", YAlign=relpos.Front, Space=1)
        )

        net.Defaults()
        ss.SetParams("Network", False)  # only set Network params
        net.Build()
        net.InitWts()

    def Init(ss):
        """
            Init restarts the run, and initializes everything, including network weights

        # all sheets
            and resets the epoch log table
        """
        rand.Seed(ss.RndSeed)
        ss.StopNow = False
        ss.SetParams("", False)
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
            return "Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t" % (
                ss.TrainEnv.Run.Cur,
                ss.TrainEnv.Epoch.Cur,
                ss.TrainEnv.Trial.Cur,
                ss.Time.Cycle,
                ss.TrainEnv.TrialName.Cur,
            )
        else:
            return "Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t" % (
                ss.TrainEnv.Run.Cur,
                ss.TrainEnv.Epoch.Cur,
                ss.TestEnv.Trial.Cur,
                ss.Time.Cycle,
                ss.TestEnv.TrialName.Cur,
            )

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
            ss.Win.PollEvents()  # this is essential for GUI responsiveness while running
        viewUpdate = ss.TrainUpdate.value
        if not train:
            viewUpdate = ss.TestUpdate.value

        if train:
            ss.Net.WtFmDWt()

        ss.Net.AlphaCycInit(train)
        ss.Time.AlphaCycStart()
        for qtr in range(4):
            for cyc in range(ss.Time.CycPerQtr):
                ss.Net.Cycle(ss.Time)
                ss.Time.CycleInc()
                if ss.ViewOn:
                    if viewUpdate == leabra.Cycle:
                        if cyc != ss.Time.CycPerQtr - 1:  # will be updated by quarter
                            ss.UpdateView(train)
                    if viewUpdate == leabra.FastSpike:
                        if (cyc + 1) % 10 == 0:
                            ss.UpdateView(train)
            ss.Net.QuarterFinal(ss.Time)
            ss.Time.QuarterInc()
            if ss.ViewOn:
                if viewUpdate <= leabra.Quarter:
                    ss.UpdateView(train)
                if viewUpdate == leabra.Phase:
                    if qtr >= 2:
                        ss.UpdateView(train)

        if train:
            ss.Net.DWt()
        if ss.ViewOn and viewUpdate == leabra.AlphaCycle:
            ss.UpdateView(train)

    def AlphaCycTest(ss):
        """
        AlphaCycTest is for testing -- uses threshold stopping and longer quarters
        """

        viewUpdate = ss.TestUpdate.value
        train = False

        out = leabra.Layer(ss.Net.LayerByName("Output"))

        ss.Net.AlphaCycInit(train)
        ss.Time.AlphaCycStart()
        overThresh = False
        for qtr in range(4):
            for cyc in range(75):  # note: fixed 75 per quarter = 200 total
                ss.Net.Cycle(ss.Time)
                ss.Time.CycleInc()
                if ss.ViewOn:
                    if viewUpdate == leabra.Cycle:
                        ss.UpdateView(train)
                    if viewUpdate == leabra.FastSpike:
                        if (cyc + 1) % 10 == 0:
                            ss.UpdateView(train)
                outact = out.Pools[0].Inhib.Act.Max
                if outact > 0.51:
                    overThresh = True
                    break
            ss.Net.QuarterFinal(ss.Time)
            ss.Time.QuarterInc()
            if ss.ViewOn:
                if viewUpdate <= leabra.Quarter:
                    ss.UpdateView(train)
                if viewUpdate == leabra.Phase:
                    if qtr >= 2:
                        ss.UpdateView(train)
            if overThresh:
                break

        ss.UpdateView(False)

    def AlphaCycTestCyc(ss, cycs):
        """
        AlphaCycTestCyc test with specified number of cycles
        """

        viewUpdate = ss.TestUpdate.value
        train = False

        out = leabra.Layer(ss.Net.LayerByName("Output"))

        ss.Net.AlphaCycInit(train)
        ss.Time.AlphaCycStart()
        for cyc in range(cycs):  # just fixed cycles, no quarters
            ss.Net.Cycle(ss.Time)
            ss.Time.CycleInc()
            if ss.ViewOn:
                if viewUpdate == leabra.Cycle:
                    ss.UpdateView(train)
                if viewUpdate == leabra.FastSpike:
                    if (cyc + 1) % 10 == 0:
                        ss.UpdateView(train)
            outact = out.Pools[0].Inhib.Act.Max
            if cycs > 100 and outact > 0.51:  # only for long trials
                break
        ss.Net.QuarterFinal(ss.Time)
        ss.Time.QuarterInc()
        if ss.ViewOn:
            if viewUpdate <= leabra.Quarter:
                ss.UpdateView(train)
            if viewUpdate == leabra.Phase:
                ss.UpdateView(train)

        ss.UpdateView(False)

    def ApplyInputs(ss, en):
        """
            ApplyInputs applies input patterns from given envirbonment.
            It is good practice to have this be a separate method with appropriate

        # going to the same layers, but good practice and cheap anyway
            args so that it can be used for various different contexts
            (training, testing, etc).
        """
        ss.Net.InitExt()

        lays = go.Slice_string(["Colors", "Words", "Output", "PFC"])
        for lnm in lays:
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            pats = en.State(ly.Nm)
            if pats != 0:
                ly.ApplyExt(pats)

    def TrainTrial(ss):
        """
        TrainTrial runs one trial of training using TrainEnv
        """
        if ss.NeedsNewRun:
            ss.NewRun()

        ss.TrainEnv.Step()

        # Key to query counters FIRST because current state is in NEXT epoch
        # if epoch counter has changed
        epc = env.CounterCur(ss.TrainEnv, env.Epoch)
        chg = env.CounterChg(ss.TrainEnv, env.Epoch)
        if chg:
            ss.LogTrnEpc(ss.TrnEpcLog)
            if ss.ViewOn and ss.TrainUpdate.value > leabra.AlphaCycle:
                ss.UpdateView(True)
            if (
                ss.TestInterval > 0 and epc % ss.TestInterval == 0
            ):  # note: epc is *next* so won't trigger first time
                ss.TestAll()
            if epc >= ss.MaxEpcs:
                # done with training..
                ss.RunEnd()
                if ss.TrainEnv.Run.Incr():  # we are done!
                    ss.StopNow = True
                    return
                else:
                    ss.NeedsNewRun = True
                    return

        ss.SetParamsSet("Training", "Network", False)
        out = leabra.Layer(ss.Net.LayerByName("Output"))
        out.SetType(emer.Target)

        ss.ApplyInputs(ss.TrainEnv)
        ss.AlphaCyc(True)  # train
        ss.TrialStats(True)  # accumulate

    def RunEnd(ss):
        """
        RunEnd is called at the end of a run -- save weights, record final log, etc here
        """
        ss.LogRun(ss.RunLog)

    def NewRun(ss):
        """
        NewRun intializes a new run of the model, using the TrainEnv.Run counter
        for the new run value
        """
        run = ss.TrainEnv.Run.Cur
        ss.TrainEnv.Init(run)
        ss.TestEnv.Init(run)
        ss.SOATestEnv.Init(run)
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

        ss.SumErr = 0
        ss.SumSSE = 0
        ss.SumAvgSSE = 0
        ss.SumCosDiff = 0

        ss.TrlErr = 0
        ss.TrlSSE = 0
        ss.TrlAvgSSE = 0
        ss.EpcSSE = 0
        ss.EpcAvgSSE = 0
        ss.EpcPctErr = 0
        ss.EpcCosDiff = 0

    def TrialStats(ss, accum):
        """
        TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
        accum is true.  Note that we're accumulating stats here on the Sim side so the
        core algorithm side remains as simple as possible, and doesn't need to worry about
        different time-scales over which stats could be accumulated etc.
        You can also aggregate directly from log data, as is done for testing stats
        """
        out = leabra.Layer(ss.Net.LayerByName("Output"))
        ss.TrlCosDiff = float(out.CosDiff.Cos)
        ss.TrlSSE = out.SSE(0.5)  # 0.5 = per-unit tolerance -- right side of .5
        ss.TrlAvgSSE = ss.TrlSSE / len(out.Neurons)
        if ss.TrlSSE > 0:
            ss.TrlErr = 1
        else:
            ss.TrlErr = 0
        if accum:
            ss.SumErr += ss.TrlErr
            ss.SumSSE += ss.TrlSSE
            ss.SumAvgSSE += ss.TrlAvgSSE
            ss.SumCosDiff += ss.TrlCosDiff
        return

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

    def SaveWeights(ss, filename):
        """
        SaveWeights saves the network weights -- when called with views.CallMethod
        it will auto-prompt for filename
        """
        ss.Net.SaveWtsJSON(filename)

    def TestTrial(ss, returnOnChg):
        """
        TestTrial runs one trial of testing -- always sequentially presented inputs
        """
        ss.TestEnv.Step()

        chg = env.CounterChg(ss.TestEnv, env.Epoch)
        if chg:
            if ss.ViewOn and ss.TestUpdate.value > leabra.AlphaCycle:
                ss.UpdateView(False)
            ss.LogTstEpc(ss.TstEpcLog)
            if returnOnChg:
                return

        ss.SetParamsSet("Testing", "Network", False)
        out = leabra.Layer(ss.Net.LayerByName("Output"))
        out.SetType(emer.Compare)

        ss.ApplyInputs(ss.TestEnv)
        ss.AlphaCycTest()
        ss.TrialStats(False)
        ss.LogTstTrl(ss.TstTrlLog, ss.TestEnv.Trial.Cur, ss.TestEnv.TrialName.Cur)

    def TestAll(ss):
        """
        TestAll runs through the full set of testing items
        """

        ss.SetParamsSet("Testing", "Network", False)
        ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
        while True:
            ss.TestTrial(True)
            chg = env.CounterChg(ss.TestEnv, env.Epoch)
            if chg or ss.StopNow:
                break

    def RunTestAll(ss):
        """
        RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
        """
        ss.StopNow = False
        ss.TestAll()
        ss.Stopped()

    def SOATestTrial(ss, returnOnChg):
        """
        SOATestTrial runs one trial of testing -- always sequentially presented inputs
        """
        ss.SOATestEnv.Step()

        chg = env.CounterChg(ss.SOATestEnv, env.Epoch)
        if chg:
            if ss.ViewOn and ss.TestUpdate.value > leabra.AlphaCycle:
                ss.UpdateView(False)
            if returnOnChg:
                return

        trl = ss.SOATestEnv.Trial.Cur
        ss.SOA = int(ss.SOAPats.CellFloat("SOA", trl))
        ss.SOAMaxCyc = int(ss.SOAPats.CellFloat("MaxCycles", trl))
        ss.SOATrlTyp = int(ss.SOAPats.CellFloat("TrialType", trl))

        ss.SetParamsSet("Testing", "Network", False)
        ss.SetParamsSet("SOATesting", "Network", False)
        out = leabra.Layer(ss.Net.LayerByName("Output"))
        out.SetType(emer.Compare)

        islate = "latestim" in ss.SOATestEnv.TrialName.Cur
        if not islate or ss.SOA == 0:
            ss.Net.InitActs()
        ss.ApplyInputs(ss.SOATestEnv)
        ss.AlphaCycTestCyc(ss.SOAMaxCyc)
        if "latestim" in ss.SOATestEnv.TrialName.Cur:
            ss.TrialStats(False)
            ss.LogSOATrl(ss.SOATrlLog, ss.SOATestEnv.Trial.Cur)

    def SOATestAll(ss):
        """
        SOATestAll runs through the full set of testing items
        """
        ss.SOATestEnv.Init(ss.TrainEnv.Run.Cur)
        ss.SOATrlLog.SetNumRows(0)
        while True:
            ss.SOATestTrial(True)
            chg = env.CounterChg(ss.SOATestEnv, env.Epoch)
            if chg or ss.StopNow:
                break

    def RunSOATestAll(ss):
        """
        RunSOATestAll runs through the full set of testing items, has stop running = false at end -- for gui
        """
        ss.StopNow = False
        ss.SOATestAll()
        ss.Stopped()

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

            spo = (
                ss.Params.SetByName("Testing").SheetByName("Network").SelByName("Layer")
            )
            spo.Params.SetParamByName("Layer.Act.Dt.VmTau", ("%g" % (ss.DtVmTau)))

            if "Network" in pset.Sheets:
                netp = pset.SheetByNameTry("Network")
                ss.Net.ApplyParams(netp, setMsg)

            hid = leabra.Layer(ss.Net.LayerByName("Hidden"))
            fmpfc = leabra.Prjn(hid.RcvPrjns.SendName("PFC"))
            fmpfc.WtScale.Rel = ss.FmPFC

        if sheet == "" or sheet == "Sim":
            if "Sim" in pset.Sheets:
                simp = pset.SheetByNameTry("Sim")
                pyparams.ApplyParams(ss, simp, setMsg)

        if sheet == "" or sheet == "Sim":
            if "Sim" in pset.Sheets:
                ss = pset.Sheets["Sim"]
                simp.Apply(ss, setMsg)

    def OpenPat(ss, dt, fname, name, desc):
        dt.OpenCSV(fname, etable.Tab)
        dt.SetMetaData("name", name)
        dt.SetMetaData("desc", desc)

    def OpenPats(ss):
        ss.OpenPat(
            ss.TrainPats, "stroop_train.tsv", "Stroop Train", "Stroop Training patterns"
        )
        ss.OpenPat(
            ss.TestPats, "stroop_test.tsv", "Stroop Test", "Stroop Testing patterns"
        )
        ss.OpenPat(
            ss.SOAPats, "stroop_soa.tsv", "Stroop SOA", "Stroop SOA Testing patterns"
        )

    def ValuesTsr(ss, name):
        """
        ValuesTsr gets value tensor of given name, creating if not yet made
        """
        if name in ss.ValuesTsrs:
            return ss.ValuesTsrs[name]
        tsr = etensor.Float32()
        ss.ValuesTsrs[name] = tsr
        return tsr

    def LogTrnEpc(ss, dt):
        """
        LogTrnEpc adds data from current epoch to the TrnEpcLog table.
        computes epoch averages prior to logging.
        """
        row = dt.Rows
        dt.SetNumRows(row + 1)

        epc = ss.TrainEnv.Epoch.Prv
        nt = float(len(ss.TrainEnv.Order))

        ss.EpcSSE = ss.SumSSE / nt
        ss.SumSSE = 0
        ss.EpcAvgSSE = ss.SumAvgSSE / nt
        ss.SumAvgSSE = 0
        ss.EpcPctErr = float(ss.SumErr) / nt
        ss.SumErr = 0
        ss.EpcPctCor = 1 - ss.EpcPctErr
        ss.EpcCosDiff = ss.SumCosDiff / nt
        ss.SumCosDiff = 0

        dt.SetCellFloat("Run", row, float(ss.TrainEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellFloat("SSE", row, ss.EpcSSE)
        dt.SetCellFloat("AvgSSE", row, ss.EpcAvgSSE)
        dt.SetCellFloat("PctErr", row, ss.EpcPctErr)
        dt.SetCellFloat("PctCor", row, ss.EpcPctCor)
        dt.SetCellFloat("CosDiff", row, ss.EpcCosDiff)

        ss.TrnEpcPlot.GoUpdate()
        if ss.TrnEpcFile != 0:
            if ss.TrainEnv.Run.Cur == 0 and epc == 0:
                dt.WriteCSVHeaders(ss.TrnEpcFile, etable.Tab)
            dt.WriteCSVRow(ss.TrnEpcFile, row, etable.Tab)

    def ConfigTrnEpcLog(ss, dt):
        dt.SetMetaData("name", "TrnEpcLog")
        dt.SetMetaData("desc", "Record of performance over epochs of training")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        sch = etable.Schema(
            [
                etable.Column("Run", etensor.INT64, go.nil, go.nil),
                etable.Column("Epoch", etensor.INT64, go.nil, go.nil),
                etable.Column("SSE", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("AvgSSE", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("PctErr", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("PctCor", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("CosDiff", etensor.FLOAT64, go.nil, go.nil),
            ]
        )
        dt.SetFromSchema(sch, 0)

    def ConfigTrnEpcPlot(ss, plt, dt):
        plt.Params.Title = "Stroop Epoch Plot"
        plt.Params.XAxisCol = "Epoch"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams(
            "SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0
        )  # default plot
        plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

        return plt

    def LogTstTrl(ss, dt, trl, trlnm):
        """
            LogTstTrl adds data from current trial to the TstTrlLog table.
        # this is triggered by increment so use previous value
            log always contains number of testing items
        """
        epc = ss.TrainEnv.Epoch.Prv

        row = dt.Rows
        if dt.Rows <= row:
            dt.SetNumRows(row + 1)

        dt.SetCellFloat("Run", row, float(ss.TrainEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellFloat("Trial", row, float(trl % 3))
        dt.SetCellString("TrialName", row, trlnm)
        dt.SetCellFloat("Cycle", row, float(ss.Time.Cycle))
        dt.SetCellFloat("Err", row, ss.TrlErr)
        dt.SetCellFloat("SSE", row, ss.TrlSSE)
        dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
        dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

        for lnm in ss.TstRecLays:
            tsr = ss.ValuesTsr(lnm)
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            ly.UnitValuesTensor(tsr, "ActM")  # get minus phase act
            dt.SetCellTensor(lnm, row, tsr)

        # note: essential to use Go version of update when called from another goroutine
        ss.TstTrlPlot.GoUpdate()

    def ConfigTstTrlLog(ss, dt):
        dt.SetMetaData("name", "TstTrlLog")
        dt.SetMetaData("desc", "Record of testing per input pattern")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        nt = ss.TestEnv.Table.Len()  # number in view
        sch = etable.Schema(
            [
                etable.Column("Run", etensor.INT64, go.nil, go.nil),
                etable.Column("Epoch", etensor.INT64, go.nil, go.nil),
                etable.Column("Trial", etensor.INT64, go.nil, go.nil),
                etable.Column("TrialName", etensor.STRING, go.nil, go.nil),
                etable.Column("Cycle", etensor.INT64, go.nil, go.nil),
                etable.Column("Err", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("SSE", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("AvgSSE", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("CosDiff", etensor.FLOAT64, go.nil, go.nil),
            ]
        )
        for lnm in ss.TstRecLays:
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            sch.append(etable.Column(lnm, etensor.FLOAT64, ly.Shp.Shp, go.nil))
        dt.SetFromSchema(sch, nt)

    def ConfigTstTrlPlot(ss, plt, dt):
        plt.Params.Title = "Stroop Test Trial Plot"
        plt.Params.XAxisCol = "Trial"
        plt.SetTable(dt)
        plt.Params.Points = True
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("TrialName", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams(
            "Cycle", eplot.On, eplot.FixMin, 0, eplot.FixMax, 250
        )  # default plot
        plt.SetColParams("Err", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

        for lnm in ss.TstRecLays:
            cp = plt.SetColParams(lnm, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
            cp.TensorIndex = -1  # plot all

        return plt

    def LogSOATrl(ss, dt, trl):
        """
            LogSOATrl adds data from current trial to the SOATrlLog table.
        # this is triggered by increment so use previous value
            log always contains number of testing items
        """
        epc = ss.TrainEnv.Epoch.Prv

        row = dt.Rows
        if dt.Rows <= row:
            dt.SetNumRows(row + 1)

        conds = go.Slice_string(["Color_Conf", "Color_Cong", "Word_Conf", "Word_Cong"])

        dt.SetCellFloat("Run", row, float(ss.TrainEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellFloat("Trial", row, float(ss.SOATrlTyp))
        dt.SetCellFloat("SOA", row, float(ss.SOA))
        dt.SetCellString("TrialName", row, conds[ss.SOATrlTyp])
        dt.SetCellFloat("Cycle", row, float(ss.Time.Cycle))
        dt.SetCellFloat("Err", row, ss.TrlErr)
        dt.SetCellFloat("SSE", row, ss.TrlSSE)
        dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
        dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

        for lnm in ss.TstRecLays:
            tsr = ss.ValuesTsr(lnm)
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            ly.UnitValuesTensor(tsr, "ActM")  # get minus phase act
            dt.SetCellTensor(lnm, row, tsr)

        # note: essential to use Go version of update when called from another goroutine
        ss.SOATrlPlot.GoUpdate()

    def ConfigSOATrlLog(ss, dt):
        dt.SetMetaData("name", "SOATrlLog")
        dt.SetMetaData("desc", "Record of testing per input pattern")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        nt = ss.SOATestEnv.Table.Len()  # number in view
        sch = etable.Schema(
            [
                etable.Column("Run", etensor.INT64, go.nil, go.nil),
                etable.Column("Epoch", etensor.INT64, go.nil, go.nil),
                etable.Column("Trial", etensor.INT64, go.nil, go.nil),
                etable.Column("SOA", etensor.INT64, go.nil, go.nil),
                etable.Column("TrialName", etensor.STRING, go.nil, go.nil),
                etable.Column("Cycle", etensor.INT64, go.nil, go.nil),
                etable.Column("Err", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("SSE", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("AvgSSE", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("CosDiff", etensor.FLOAT64, go.nil, go.nil),
            ]
        )
        for lnm in ss.TstRecLays:
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            sch.append(etable.Column(lnm, etensor.FLOAT64, ly.Shp.Shp, go.nil))
        dt.SetFromSchema(sch, nt)

    def ConfigSOATrlPlot(ss, plt, dt):
        plt.Params.Title = "Stroop SOA Test Trial Plot"
        plt.Params.XAxisCol = "SOA"
        plt.Params.LegendCol = "TrialName"
        plt.SetTable(dt)
        plt.Params.Points = True
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("SOA", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams(
            "Cycle", eplot.On, eplot.FixMin, 0, eplot.FixMax, 220
        )  # default plot
        plt.SetColParams("Err", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

        for lnm in ss.TstRecLays:
            cp = plt.SetColParams(lnm, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
            cp.TensorIndex = -1  # plot all

        return plt

    def LogTstEpc(ss, dt):
        row = dt.Rows
        dt.SetNumRows(row + 1)

        trl = ss.TstTrlLog
        tix = etable.NewIndexView(trl)
        epc = ss.TrainEnv.Epoch.Prv  # ?

        # note: this shows how to use agg methods to compute summary data from another
        # data table, instead of incrementing on the Sim
        dt.SetCellFloat("Run", row, float(ss.TrainEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellFloat("SSE", row, agg.Sum(tix, "SSE")[0])
        dt.SetCellFloat("AvgSSE", row, agg.Mean(tix, "AvgSSE")[0])
        dt.SetCellFloat("PctErr", row, agg.Mean(tix, "Err")[0])
        dt.SetCellFloat("PctCor", row, 1 - agg.Mean(tix, "Err")[0])
        dt.SetCellFloat("CosDiff", row, agg.Mean(tix, "CosDiff")[0])

        # note: essential to use Go version of update when called from another goroutine
        ss.TstEpcPlot.GoUpdate()

    def ConfigTstEpcLog(ss, dt):
        dt.SetMetaData("name", "TstEpcLog")
        dt.SetMetaData("desc", "Summary stats for testing trials")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        sch = etable.Schema(
            [
                etable.Column("Run", etensor.INT64, go.nil, go.nil),
                etable.Column("Epoch", etensor.INT64, go.nil, go.nil),
                etable.Column("SSE", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("AvgSSE", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("PctErr", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("PctCor", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("CosDiff", etensor.FLOAT64, go.nil, go.nil),
            ]
        )
        dt.SetFromSchema(sch, 0)

    def ConfigTstEpcPlot(ss, plt, dt):
        plt.Params.Title = "Stroop Testing Epoch Plot"
        plt.Params.XAxisCol = "Epoch"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams(
            "SSE", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0
        )  # default plot
        plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("PctErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        return plt

    def LogRun(ss, dt):
        """
            LogRun adds data from current run to the RunLog table.
        # this is NOT triggered by increment yet -- use Cur
        """
        run = ss.TrainEnv.Run.Cur
        row = dt.Rows
        dt.SetNumRows(row + 1)

        epclog = ss.TrnEpcLog
        epcix = etable.NewIndexView(epclog)
        # compute mean over last N epochs for run level
        nlast = 5
        if nlast > epcix.Len() - 1:
            nlast = epcix.Len() - 1
        epcix.Indexes = epcix.Indexes[epcix.Len() - nlast :]

        params = ""

        dt.SetCellFloat("Run", row, float(run))
        dt.SetCellString("Params", row, params)
        dt.SetCellFloat("SSE", row, agg.Mean(epcix, "SSE")[0])
        dt.SetCellFloat("AvgSSE", row, agg.Mean(epcix, "AvgSSE")[0])
        dt.SetCellFloat("PctErr", row, agg.Mean(epcix, "PctErr")[0])
        dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
        dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])

        runix = etable.NewIndexView(dt)
        spl = split.GroupBy(runix, go.Slice_string(["Params"]))
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
            [
                etable.Column("Run", etensor.INT64, go.nil, go.nil),
                etable.Column("Params", etensor.STRING, go.nil, go.nil),
                etable.Column("SSE", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("AvgSSE", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("PctErr", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("PctCor", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("CosDiff", etensor.FLOAT64, go.nil, go.nil),
            ]
        )
        dt.SetFromSchema(sch, 0)

    def ConfigRunPlot(ss, plt, dt):
        plt.Params.Title = "Stroop Run Plot"
        plt.Params.XAxisCol = "Run"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("SSE", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("PctErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        return plt

    def ConfigNetView(ss, nv):
        nv.ViewDefaults()
        nv.Scene().Camera.Pose.Pos.Set(0.1, 1.8, 3.5)
        nv.Scene().Camera.LookAt(math32.Vector3(0.1, 0.15, 0), math32.Vector3(0, 1, 0))

        labs = go.Slice_string(
            [
                "     g      r",
                "   G       R",
                "  gr      rd",
                "     g      r         G      R",
                "  cn     wr",
            ]
        )
        nv.ConfigLabels(labs)

        lays = go.Slice_string(["Colors", "Words", "Output", "Hidden", "PFC"])

        for li, lnm in enumerate(lays):
            ly = nv.LayerByName(lnm)
            lbl = nv.LabelByName(labs[li])
            lbl.Pose = ly.Pose
            lbl.Pose.Pos.Y += 0.2
            lbl.Pose.Pos.Z += 0.02
            lbl.Pose.Scale.SetMul(math32.Vector3(0.4, 0.06, 0.5))

    def ConfigGui(ss):
        """
        ConfigGui configures the GoGi gui interface for this simulation,
        """
        width = 1600
        height = 1200

        core.SetAppName("stroop")
        core.SetAppAbout(
            'illustrates how the PFC can produce top-down biasing for executive control, in the context of the widely-studied Stroop task. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch10/stroop/README.md">README.md on GitHub</a>.</p>'
        )

        win = core.NewMainWindow("stroop", "Stroop", width, height)
        ss.Win = win

        vp = win.WinViewport2D()
        ss.vp = vp
        updt = vp.UpdateStart()

        mfr = win.SetMainFrame()

        tbar = core.AddNewToolBar(mfr, "tbar")
        tbar.SetStretchMaxWidth()
        ss.ToolBar = tbar

        split = core.AddNewSplitView(mfr, "split")
        split.Dim = math32.X
        split.SetStretchMax()

        cv = ss.NewClassView("sv")
        cv.AddFrame(split)
        cv.Config()

        tv = core.AddNewTabView(split, "tv")

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
        tv.AddTab(plt, "TstTrlPlot")
        ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

        plt = eplot.Plot2D()
        tv.AddTab(plt, "SOATrlPlot")
        ss.SOATrlPlot = ss.ConfigSOATrlPlot(plt, ss.SOATrlLog)

        plt = eplot.Plot2D()
        tv.AddTab(plt, "TstEpcPlot")
        ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.TstEpcLog)

        plt = eplot.Plot2D()
        tv.AddTab(plt, "RunPlot")
        ss.RunPlot = ss.ConfigRunPlot(plt, ss.RunLog)

        split.SetSplitsList(go.Slice_float32([0.2, 0.8]))
        recv = win.This()

        tbar.AddAction(
            core.ActOpts(
                Label="Init",
                Icon="update",
                Tooltip="Initialize everything including network weights, and start over.  Also applies current params.",
                UpdateFunc=UpdateFuncNotRunning,
            ),
            recv,
            InitCB,
        )

        tbar.AddAction(
            core.ActOpts(
                Label="Train",
                Icon="run",
                Tooltip="Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.",
                UpdateFunc=UpdateFuncNotRunning,
            ),
            recv,
            TrainCB,
        )

        tbar.AddAction(
            core.ActOpts(
                Label="Stop",
                Icon="stop",
                Tooltip="Interrupts running.  Hitting Train again will pick back up where it left off.",
                UpdateFunc=UpdateFuncRunning,
            ),
            recv,
            StopCB,
        )

        tbar.AddAction(
            core.ActOpts(
                Label="Step Trial",
                Icon="step-fwd",
                Tooltip="Advances one training trial at a time.",
                UpdateFunc=UpdateFuncNotRunning,
            ),
            recv,
            StepTrialCB,
        )

        tbar.AddAction(
            core.ActOpts(
                Label="Step Epoch",
                Icon="fast-fwd",
                Tooltip="Advances one epoch (complete set of training patterns) at a time.",
                UpdateFunc=UpdateFuncNotRunning,
            ),
            recv,
            StepEpochCB,
        )

        tbar.AddAction(
            core.ActOpts(
                Label="Step Run",
                Icon="fast-fwd",
                Tooltip="Advances one full training Run at a time.",
                UpdateFunc=UpdateFuncNotRunning,
            ),
            recv,
            StepRunCB,
        )

        tbar.AddSeparator("test")

        tbar.AddAction(
            core.ActOpts(
                Label="Test Trial",
                Icon="step-fwd",
                Tooltip="Runs the next testing trial.",
                UpdateFunc=UpdateFuncNotRunning,
            ),
            recv,
            TestTrialCB,
        )

        tbar.AddAction(
            core.ActOpts(
                Label="Test All",
                Icon="fast-fwd",
                Tooltip="Tests all of the testing trials.",
                UpdateFunc=UpdateFuncNotRunning,
            ),
            recv,
            TestAllCB,
        )

        tbar.AddAction(
            core.ActOpts(
                Label="SOA Test Trial",
                Icon="step-fwd",
                Tooltip="Runs the next testing trial.",
                UpdateFunc=UpdateFuncNotRunning,
            ),
            recv,
            SOATestTrialCB,
        )

        tbar.AddAction(
            core.ActOpts(
                Label="SOA Test All",
                Icon="fast-fwd",
                Tooltip="Tests all of the testing trials.",
                UpdateFunc=UpdateFuncNotRunning,
            ),
            recv,
            SOATestAllCB,
        )

        tbar.AddSeparator("misc")

        tbar.AddAction(
            core.ActOpts(
                Label="Reset TstTrlLog",
                Icon="reset",
                Tooltip="Reset the test trial log -- otherwise it accumulates to compare across parameters etc.",
            ),
            recv,
            ResetTstTrlLogCB,
        )

        tbar.AddAction(
            core.ActOpts(
                Label="New Seed",
                Icon="new",
                Tooltip="Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
            ),
            recv,
            NewRndSeedCB,
        )

        tbar.AddAction(
            core.ActOpts(
                Label="Defaults",
                Icon="update",
                Tooltip="Restore initial default parameters.",
                UpdateFunc=UpdateFuncNotRunning,
            ),
            recv,
            DefaultsCB,
        )

        tbar.AddAction(
            core.ActOpts(
                Label="README",
                Icon="file-markdown",
                Tooltip="Opens your browser on the README file that contains instructions for how to run this model.",
            ),
            recv,
            ReadmeCB,
        )

        # main menu
        appnm = core.AppName()
        mmen = win.MainMenu
        mmen.ConfigMenus(go.Slice_string([appnm, "File", "Edit", "Window"]))

        amen = core.Action(win.MainMenu.ChildByName(appnm, 0))
        amen.Menu.AddAppMenu(win)

        emen = core.Action(win.MainMenu.ChildByName("Edit", 1))
        emen.Menu.AddCopyCutPaste(win)

        # note: Command in shortcuts is automatically translated into Control for
        # Linux, Windows or Meta for MacOS
        # fmen := win.MainMenu.ChildByName("File", 0).(*core.Action)
        # fmen.Menu.AddAction(core.ActOpts{Label: "Open", Shortcut: "Command+O"},
        #   win.This(), func(recv, send tree.Ki, sig int64, data interface{}) {
        #       FileViewOpenSVG(vp)
        #   })
        # fmen.Menu.AddSeparator("csep")
        # fmen.Menu.AddAction(core.ActOpts{Label: "Close Window", Shortcut: "Command+W"},
        #   win.This(), func(recv, send tree.Ki, sig int64, data interface{}) {
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
