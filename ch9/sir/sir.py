#!/usr/local/bin/pyleabra -i

# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# use:
# pyleabra -i ra25.py
# to run in gui interactive mode from the command line (or pyleabra, import ra25)
# see main function at the end for startup args

# sir illustrates the dynamic gating of information into PFC active maintenance,
# by the basal ganglia (BG). It uses a simple Store-Ignore-Recall (SIR) task,
# where the BG system learns via phasic dopamine signals and trial-and-error exploration,
# discovering what needs to be stored, ignored, and recalled as a function of
# reinforcement of correct behavior, and learned reinforcement of useful
# working memory representations.

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
    pbwm,
    rl,
)

import importlib as il  # il.reload(ra25) -- doesn't seem to work for reasons unknown
import io, sys, getopt
from datetime import datetime, timezone
from enum import Enum
import numpy as np

from sir_env import SIREnv, Actions

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


def ResetRunLogCB(recv, send, sig, data):
    TheSim.RunLog.SetNumRows(0)
    TheSim.RunPlot.Update()


def DefaultsCB(recv, send, sig, data):
    TheSim.Defaults()
    TheSim.Init()
    TheSim.UpdateClassView()
    TheSim.vp.SetNeedsFullRender()


def NewRndSeedCB(recv, send, sig, data):
    TheSim.NewRndSeed()


def ReadmeCB(recv, send, sig, data):
    core.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch10/sir/README.md")


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
        self.BurstDaGain = float(1)
        self.SetTags(
            "BurstDaGain",
            'desc:"strength of dopamine bursts: 1 default -- reduce for PD OFF, increase for PD ON"',
        )
        self.DipDaGain = float(1)
        self.SetTags(
            "DipDaGain",
            'desc:"strength of dopamine dips: 1 default -- reduce to siulate D2 agonists"',
        )
        self.Net = pbwm.Network()
        self.SetTags(
            "Net",
            'view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"',
        )
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
        self.RunLog = etable.Table()
        self.SetTags("RunLog", 'view:"no-inline" desc:"summary log of each run"')
        self.RunStats = etable.Table()
        self.SetTags("RunStats", 'view:"no-inline" desc:"aggregate stats on all runs"')
        self.SimMat = simat.SimMat()
        self.SetTags("SimMat", 'view:"no-inline" desc:"similarity matrix"')
        self.Params = params.Sets()
        self.SetTags("Params", 'view:"no-inline" desc:"full collection of param sets"')
        self.ParamSet = str()
        self.SetTags(
            "ParamSet",
            'desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set -- can use multiple names separated by spaces (don\'t put spaces in ParamSet names!)"',
        )
        self.Tag = str()
        self.SetTags(
            "Tag",
            'desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"',
        )
        self.MaxRuns = int(10)
        self.SetTags("MaxRuns", 'desc:"maximum number of model runs to perform"')
        self.MaxEpcs = int(100)
        self.SetTags("MaxEpcs", 'desc:"maximum number of epochs to run per model run"')
        self.MaxTrls = int(100)
        self.SetTags("MaxTrls", 'desc:"maximum number of training trials per epoch"')
        self.NZeroStop = int(5)
        self.SetTags(
            "NZeroStop",
            'desc:"if a positive number, training will stop after this many epochs with zero SSE"',
        )
        self.TrainEnv = SIREnv()
        self.SetTags("TrainEnv", 'desc:"Training environment -- SIR environment"')
        self.TestEnv = SIREnv()
        self.SetTags("TestEnv", 'desc:"Testing nvironment -- SIR environment"')
        self.Time = leabra.Time()
        self.SetTags("Time", 'desc:"leabra timing parameters and state"')
        self.ViewOn = True
        self.SetTags(
            "ViewOn", 'desc:"whether to update the network view while running"'
        )
        self.TrainUpdate = leabra.TimeScales.AlphaCycle
        self.SetTags(
            "TrainUpdate",
            'desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"',
        )
        self.TestUpdate = leabra.TimeScales.AlphaCycle
        self.SetTags(
            "TestUpdate",
            'desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"',
        )
        self.TstRecLays = go.Slice_string(
            ["Input", "Output", "GPiThal", "PFCmntD", "PFCoutD"]
        )
        self.SetTags(
            "TstRecLays",
            'desc:"names of layers to record activations etc of during testing"',
        )

        self.TrlDA = float()
        self.SetTags("TrlDA", 'inactive:"+" desc:"dopamine level on this trial"')
        self.TrlAbsDA = float()
        self.SetTags(
            "TrlAbsDA", 'inactive:"+" desc:"absolute value of dopamine on this trial"'
        )
        self.TrlRewPred = float()
        self.SetTags(
            "TrlRewPred", 'inactive:"+" desc:"reward prediction level on this trial"'
        )
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
        self.EpcDA = float()
        self.SetTags("EpcDA", 'inactive:"+" desc:"last epoch\'s average dopamine"')
        self.EpcAbsDA = float()
        self.SetTags("EpcAbsDA", 'inactive:"+" desc:"last epoch\'s avg abs dopamine"')
        self.EpcRewPred = float()
        self.SetTags(
            "EpcRewPred", 'inactive:"+" desc:"last epoch\'s avg rew pred dopamine"'
        )
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
        self.SetTags("EpcPctErr", 'inactive:"+" desc:"last epoch\'s average TrlErr"')
        self.EpcPctCor = float()
        self.SetTags(
            "EpcPctCor", 'inactive:"+" desc:"1 - last epoch\'s average TrlErr"'
        )
        self.EpcCosDiff = float()
        self.SetTags(
            "EpcCosDiff",
            'inactive:"+" desc:"last epoch\'s average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"',
        )
        self.EpcPerTrlMSec = float()
        self.SetTags(
            "EpcPerTrlMSec",
            'inactive:"+" desc:"how long did the epoch take per trial in wall-clock milliseconds"',
        )
        self.FirstZero = int()
        self.SetTags(
            "FirstZero", 'inactive:"+" desc:"epoch at when SSE first went to zero"'
        )
        self.NZero = int()
        self.SetTags(
            "NZero", 'inactive:"+" desc:"number of epochs in a row with zero SSE"'
        )

        # internal state - view:"-"
        self.SumDA = float()
        self.SetTags(
            "SumDA",
            'view:"-" inactive:"+" desc:"sum to increment as we go through epoch"',
        )
        self.SumAbsDA = float()
        self.SetTags(
            "SumAbsDA",
            'view:"-" inactive:"+" desc:"sum to increment as we go through epoch"',
        )
        self.SumRewPred = float()
        self.SetTags(
            "SumRewPred",
            'view:"-" inactive:"+" desc:"sum to increment as we go through epoch"',
        )
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
        self.WtsGrid = 0
        self.SetTags("WtsGrid", 'view:"-" desc:"the weights grid view"')
        self.TrnEpcPlot = 0
        self.SetTags("TrnEpcPlot", 'view:"-" desc:"the training epoch plot"')
        self.TstEpcPlot = 0
        self.SetTags("TstEpcPlot", 'view:"-" desc:"the testing epoch plot"')
        self.TstTrlPlot = 0
        self.SetTags("TstTrlPlot", 'view:"-" desc:"the test-trial plot"')
        self.RunPlot = 0
        self.SetTags("RunPlot", 'view:"-" desc:"the run plot"')
        self.TrnEpcFile = 0
        self.SetTags("TrnEpcFile", 'view:"-" desc:"log file"')
        self.RunFile = 0
        self.SetTags("RunFile", 'view:"-" desc:"log file"')
        self.ValuesTsrs = {}
        self.SetTags("ValuesTsrs", 'view:"-" desc:"for holding layer values"')
        self.SaveWts = False
        self.SetTags(
            "SaveWts",
            'view:"-" desc:"for command-line run only, auto-save final weights after each run"',
        )
        self.NoGui = False
        self.SetTags("NoGui", 'view:"-" desc:"if true, runing in no GUI mode"')
        self.LogSetParams = False
        self.SetTags(
            "LogSetParams",
            'view:"-" desc:"if true, print message for all params that are set"',
        )
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
        self.LastEpcTime = 0
        self.SetTags("LastEpcTime", 'view:"-" desc:"timer for last epoch"')
        self.vp = 0
        self.SetTags("vp", 'view:"-" desc:"viewport"')

    def InitParams(ss):
        """
        Sets the default set of parameters -- Base is always applied, and others can be optionally
        selected to apply on top of that
        """
        ss.Params.OpenJSON("sir.params")
        ss.Defaults()

    def Defaults(ss):
        ss.BurstDaGain = 1
        ss.DipDaGain = 1

    def Config(ss):
        """
        Config configures all the elements using the standard functions
        """
        ss.InitParams()
        ss.ConfigEnv()
        ss.ConfigNet(ss.Net)
        ss.ConfigTrnEpcLog(ss.TrnEpcLog)
        ss.ConfigTstEpcLog(ss.TstEpcLog)
        ss.ConfigTstTrlLog(ss.TstTrlLog)
        ss.ConfigRunLog(ss.RunLog)

    def ConfigEnv(ss):
        if ss.MaxRuns == 0:  # allow user override
            ss.MaxRuns = 10
        if ss.MaxEpcs == 0:  # allow user override
            ss.MaxEpcs = 100
            ss.NZeroStop = 5
        if ss.MaxTrls == 0:  # allow user override
            ss.MaxTrls = 100

        ss.TrainEnv.Nm = "TrainEnv"
        ss.TrainEnv.Dsc = "training params and state"
        ss.TrainEnv.SetNStim(4)
        ss.TrainEnv.RewVal = 1
        ss.TrainEnv.NoRewVal = 0
        ss.TrainEnv.Validate()
        ss.TrainEnv.Run.Max = (
            ss.MaxRuns
        )  # note: we are not setting epoch max -- do that manually
        ss.TrainEnv.Trial.Max = ss.MaxTrls

        ss.TestEnv.Nm = "TestEnv"
        ss.TestEnv.Dsc = "testing params and state"
        ss.TestEnv.SetNStim(4)
        ss.TestEnv.RewVal = 1
        ss.TestEnv.NoRewVal = 0
        ss.TestEnv.Validate()
        ss.TestEnv.Run.Max = (
            ss.MaxRuns
        )  # note: we are not setting epoch max -- do that manually
        ss.TestEnv.Trial.Max = 20  # good amount for testing

        ss.TrainEnv.Init(0)
        ss.TestEnv.Init(0)

    def ConfigNet(ss, net):
        net.InitName(net, "SIR")
        rc = rl.AddRWLayersPy(net.AsLeabra(), "", relpos.Behind, 2)
        rew = rc[0]
        rp = rc[1]
        da = rc[2]
        snc = rl.RWDaLayer(da)
        snc.SetName("SNc")

        inp = net.AddLayer2D("Input", 1, 4, emer.Input)
        ctrl = net.AddLayer2D("CtrlInput", 1, 3, emer.Input)
        out = net.AddLayer2D("Output", 1, 4, emer.Target)
        hid = net.AddLayer2D("Hidden", 7, 7, emer.Hidden)
        inp.SetRelPos(
            relpos.Rel(
                Rel=relpos.Above,
                Other=rew.Name(),
                YAlign=relpos.Front,
                XAlign=relpos.Left,
            )
        )
        out.SetRelPos(
            relpos.Rel(Rel=relpos.RightOf, Other="Input", YAlign=relpos.Front, Space=1)
        )
        ctrl.SetRelPos(
            relpos.Rel(Rel=relpos.Behind, Other="Input", XAlign=relpos.Left, Space=2)
        )
        hid.SetRelPos(
            relpos.Rel(
                Rel=relpos.Behind, Other="CtrlInput", XAlign=relpos.Left, Space=2
            )
        )

        # args: nY, nMaint, nOut, nNeurBgY, nNeurBgX, nNeurPfcY, nNeurPfcX
        # returns: mtxGo, mtxNoGo, gpe, gpi, cini, pfcMnt, pfcMntD, pfcOut, pfcOutD =
        nl = pbwm.AddPBWMPy(net.AsLeabra(), "", 1, 1, 1, 1, 3, 1, 4)
        mtxGo = nl[0]
        mtxNoGo = nl[1]
        cin = pbwm.CINLayer(nl[4])
        pfcMnt = nl[5]
        pfcMntD = nl[6]
        pfcOutD = nl[8]

        cin.RewLays.AddOne(rew.Name(), rp.Name())

        mtxGo.SetRelPos(
            relpos.Rel(Rel=relpos.RightOf, Other="Rew", YAlign=relpos.Front, Space=14)
        )

        full = prjn.NewFull()
        fmin = prjn.NewRect()
        fmin.Size.Set(1, 1)
        fmin.Scale.Set(1, 1)
        fmin.Wrap = True

        net.ConnectLayersPrjn(ctrl, rp, full, emer.Forward, rl.RWPrjn())
        net.ConnectLayersPrjn(pfcMntD, rp, full, emer.Forward, rl.RWPrjn())
        net.ConnectLayersPrjn(pfcOutD, rp, full, emer.Forward, rl.RWPrjn())

        pj = net.ConnectLayersPrjn(
            ctrl, mtxGo, fmin, emer.Forward, pbwm.MatrixTracePrjn()
        )
        pj.SetClass("MatrixPrjn")
        pj = net.ConnectLayersPrjn(
            ctrl, mtxNoGo, fmin, emer.Forward, pbwm.MatrixTracePrjn()
        )
        pj.SetClass("MatrixPrjn")
        pj = net.ConnectLayers(inp, pfcMnt, fmin, emer.Forward)
        pj.SetClass("PFCFixed")

        net.ConnectLayers(inp, hid, full, emer.Forward)
        net.BidirConnectLayersPy(hid, out, full)
        pj = net.ConnectLayers(pfcOutD, hid, full, emer.Forward)
        pj.SetClass("FmPFCOutD")
        pj = net.ConnectLayers(pfcOutD, out, full, emer.Forward)
        pj.SetClass("FmPFCOutD")
        net.ConnectLayers(inp, out, full, emer.Forward)

        snc.SendDA.AddAllBut(net, go.nil)  # send dopamine to all layers..

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
                ss.TrainEnv.String(),
            )
        else:
            return "Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t" % (
                ss.TrainEnv.Run.Cur,
                ss.TrainEnv.Epoch.Cur,
                ss.TestEnv.Trial.Cur,
                ss.Time.Cycle,
                ss.TestEnv.String(),
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
            if qtr == 2:
                ss.ApplyReward(train)

        if train:
            ss.Net.DWt()
        if ss.ViewOn and viewUpdate == leabra.AlphaCycle:
            ss.UpdateView(train)

    def ApplyInputs(ss, en):
        """
            ApplyInputs applies input patterns from given envirbonment.
            It is good practice to have this be a separate method with appropriate

        # going to the same layers, but good practice and cheap anyway
            args so that it can be used for various different contexts
            (training, testing, etc).
        """
        ss.Net.InitExt()

        lays = go.Slice_string(["Input", "CtrlInput", "Output"])
        for lnm in lays:
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            pats = en.State(ly.Nm)
            if pats == 0:
                continue
            ly.ApplyExt(pats)

    def ApplyReward(ss, train):
        """
        ApplyReward computes reward based on network output and applies it -- call
        at start of 3rd quarter (plus phase)
        """
        en = 0
        if train:
            en = ss.TrainEnv
        else:
            en = ss.TestEnv
        if en.Act != Actions.Recall:
            return
        out = leabra.Layer(ss.Net.LayerByName("Output"))
        mxi = out.Pools[0].Inhib.Act.MaxIndex
        en.SetReward(mxi)
        pats = en.State("Reward")
        ly = leabra.Layer(ss.Net.LayerByName("Rew"))
        ly.ApplyExt1DTsr(pats)

    def TrainTrial(ss):
        """
        TrainTrial runs one trial of training using TrainEnv
        """

        if ss.NeedsNewRun:
            ss.NewRun()

        ss.TrainEnv.Step()

        # Key to query counters FIRST because current state is in NEXT epoch
        # if epoch counter has changed
        epc = ss.TrainEnv.CounterCur(env.Epoch)
        chg = ss.TrainEnv.CounterChg(env.Epoch)
        if chg:
            if ss.ViewOn and ss.TrainUpdate.value > leabra.AlphaCycle:
                ss.UpdateView(True)
            ss.LogTrnEpc(ss.TrnEpcLog)
            if ss.ViewOn and ss.TrainUpdate.value > leabra.AlphaCycle:
                ss.UpdateView(True)
            if epc >= ss.MaxEpcs or (ss.NZeroStop > 0 and ss.NZero >= ss.NZeroStop):
                # done with training..
                ss.RunEnd()
                if ss.TrainEnv.Run.Incr():  # we are done!
                    ss.StopNow = True
                    return
                else:
                    ss.NeedsNewRun = True
                    return

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

        ss.SumDA = 0
        ss.SumAbsDA = 0
        ss.SumRewPred = 0
        ss.SumSSE = 0
        ss.SumAvgSSE = 0
        ss.SumCosDiff = 0
        ss.SumErr = 0
        ss.FirstZero = -1
        ss.NZero = 0

        ss.TrlDA = 0
        ss.TrlAbsDA = 0
        ss.TrlRewPred = 0
        ss.TrlErr = 0
        ss.TrlSSE = 0
        ss.TrlAvgSSE = 0
        ss.EpcDA = 0
        ss.EpcAbsDA = 0
        ss.EpcRewPred = 0
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
        snc = leabra.Layer(ss.Net.LayerByName("SNc"))
        rp = leabra.Layer(ss.Net.LayerByName("RWPred"))
        ss.TrlDA = float(snc.Neurons[0].Act)
        ss.TrlAbsDA = abs(ss.TrlDA)
        ss.TrlRewPred = float(rp.Neurons[0].Act)
        ss.TrlCosDiff = float(out.CosDiff.Cos)
        ss.TrlSSE = out.SSE(0.5)  # 0.5 = per-unit tolerance -- right side of .5
        ss.TrlAvgSSE = ss.TrlSSE / len(out.Neurons)
        if ss.TrlSSE > 0:
            ss.TrlErr = 1
        else:
            ss.TrlErr = 0
        if accum:
            ss.SumDA += ss.TrlDA
            ss.SumAbsDA += ss.TrlAbsDA
            ss.SumRewPred += ss.TrlRewPred
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

        chg = ss.TestEnv.CounterChg(env.Epoch)
        if chg:
            if ss.ViewOn and ss.TestUpdate.value > leabra.AlphaCycle:
                ss.UpdateView(False)
            ss.LogTstEpc(ss.TstEpcLog)
            if returnOnChg:
                return
        ss.ApplyInputs(ss.TestEnv)
        ss.AlphaCyc(False)
        ss.TrialStats(False)
        ss.LogTstTrl(ss.TstTrlLog)

    def TestAll(ss):
        """
        TestAll runs through the full set of testing items
        """
        ss.TestEnv.Init(ss.TestEnv.Run.Cur)
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
        matg = pbwm.MatrixLayer(ss.Net.LayerByName("MatrixGo"))
        matn = pbwm.MatrixLayer(ss.Net.LayerByName("MatrixNoGo"))

        matg.Matrix.BurstGain = ss.BurstDaGain
        matg.Matrix.DipGain = ss.DipDaGain
        matn.Matrix.BurstGain = ss.BurstDaGain
        matn.Matrix.DipGain = ss.DipDaGain

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
                simp = pset.SheetByNameTry("Sim")
                pyparams.ApplyParams(ss, simp, setMsg)

    def ValuesTsr(ss, name):
        """
        ValuesTsr gets value tensor of given name, creating if not yet made
        """
        if name in ss.ValuesTsrs:
            return ss.ValuesTsrs[name]
        tsr = etensor.Float32()
        ss.ValuesTsrs[name] = tsr
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
        return (
            ss.Net.Nm
            + "_"
            + ss.RunName()
            + "_"
            + ss.RunEpochName(ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur)
            + ".wts.gz"
        )

    def LogFileName(ss, lognm):
        """
        LogFileName returns default log file name
        """
        return ss.Net.Nm + "_" + ss.RunName() + "_" + lognm + ".csv"

    def LogTrnEpc(ss, dt):
        """
        LogTrnEpc adds data from current epoch to the TrnEpcLog table.
        computes epoch averages prior to logging.
        """
        row = dt.Rows
        dt.SetNumRows(row + 1)

        epc = ss.TrainEnv.Epoch.Prv
        nt = float(ss.TrainEnv.Trial.Max)

        ss.EpcDA = ss.SumDA / nt
        ss.SumDA = 0
        ss.EpcAbsDA = ss.SumAbsDA / nt
        ss.SumAbsDA = 0
        ss.EpcRewPred = ss.SumRewPred / nt
        ss.SumRewPred = 0
        ss.EpcSSE = ss.SumSSE / nt
        ss.SumSSE = 0
        ss.EpcAvgSSE = ss.SumAvgSSE / nt
        ss.SumAvgSSE = 0
        ss.EpcPctErr = float(ss.SumErr) / nt
        ss.SumErr = 0
        ss.EpcPctCor = 1 - ss.EpcPctErr
        ss.EpcCosDiff = ss.SumCosDiff / nt
        ss.SumCosDiff = 0
        if ss.FirstZero < 0 and ss.EpcPctErr == 0:
            ss.FirstZero = epc
        if ss.EpcPctErr == 0:
            ss.NZero += 1
        else:
            ss.NZero = 0

        # if ss.LastEpcTime.IsZero():
        #     ss.EpcPerTrlMSec = 0
        # else:
        #     iv = time.Now().Sub(ss.LastEpcTime)
        #     ss.EpcPerTrlMSec = float(iv) / (nt * float(time.Millisecond))
        # ss.LastEpcTime = time.Now()

        dt.SetCellFloat("Run", row, float(ss.TrainEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellFloat("SSE", row, ss.EpcSSE)
        dt.SetCellFloat("AvgSSE", row, ss.EpcAvgSSE)
        dt.SetCellFloat("PctErr", row, ss.EpcPctErr)
        dt.SetCellFloat("PctCor", row, ss.EpcPctCor)
        dt.SetCellFloat("CosDiff", row, ss.EpcCosDiff)
        dt.SetCellFloat("DA", row, ss.EpcDA)
        dt.SetCellFloat("AbsDA", row, ss.EpcAbsDA)
        dt.SetCellFloat("RewPred", row, ss.EpcRewPred)
        dt.SetCellFloat("PerTrlMSec", row, ss.EpcPerTrlMSec)

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
                etable.Column("DA", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("AbsDA", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("RewPred", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("PerTrlMSec", etensor.FLOAT64, go.nil, go.nil),
            ]
        )
        dt.SetFromSchema(sch, 0)

    def ConfigTrnEpcPlot(ss, plt, dt):
        plt.Params.Title = "SIR Epoch Plot"
        plt.Params.XAxisCol = "Epoch"
        plt.SetTable(dt)

        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("DA", eplot.Off, eplot.FixMin, -1, eplot.FixMax, 1)
        plt.SetColParams("AbsDA", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("RewPred", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("PerTrlMSec", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

        return plt

    def LogTstTrl(ss, dt):
        """
        LogTstTrl adds data from current trial to the TstTrlLog table.
        log always contains number of testing items
        """
        epc = ss.TestEnv.Epoch.Prv

        trl = ss.TestEnv.Trial.Cur
        row = trl

        if dt.Rows <= row:
            dt.SetNumRows(row + 1)

        dt.SetCellFloat("Run", row, float(ss.TestEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellFloat("Trial", row, float(trl))
        dt.SetCellString("TrialName", row, ss.TestEnv.String())
        dt.SetCellFloat("Err", row, ss.TrlErr)
        dt.SetCellFloat("SSE", row, ss.TrlSSE)
        dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
        dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)
        dt.SetCellFloat("DA", row, ss.TrlDA)
        dt.SetCellFloat("AbsDA", row, ss.TrlAbsDA)
        dt.SetCellFloat("RewPred", row, ss.TrlRewPred)

        for lnm in ss.TstRecLays:
            tsr = ss.ValuesTsr(lnm)
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            ly.UnitValuesTensor(tsr, "ActM")
            dt.SetCellTensor(lnm, row, tsr)

        ss.TstTrlPlot.GoUpdate()

    def ConfigTstTrlLog(ss, dt):
        dt.SetMetaData("name", "TstTrlLog")
        dt.SetMetaData("desc", "Record of testing per input pattern")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        nt = ss.TestEnv.Trial.Max
        sch = etable.Schema(
            [
                etable.Column("Run", etensor.INT64, go.nil, go.nil),
                etable.Column("Epoch", etensor.INT64, go.nil, go.nil),
                etable.Column("Trial", etensor.INT64, go.nil, go.nil),
                etable.Column("TrialName", etensor.STRING, go.nil, go.nil),
                etable.Column("Err", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("SSE", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("AvgSSE", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("CosDiff", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("DA", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("AbsDA", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("RewPred", etensor.FLOAT64, go.nil, go.nil),
            ]
        )
        for lnm in ss.TstRecLays:
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            sch.append(etable.Column(lnm, etensor.FLOAT64, ly.Shp.Shp, go.nil))
        dt.SetFromSchema(sch, nt)

    def ConfigTstTrlPlot(ss, plt, dt):
        plt.Params.Title = "SIR Test Trial Plot"
        plt.Params.XAxisCol = "Trial"
        plt.SetTable(dt)

        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams(
            "Err", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1
        )  # default plot
        plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("DA", eplot.On, eplot.FixMin, -1, eplot.FixMax, 1)
        plt.SetColParams("AbsDA", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("RewPred", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)

        for lnm in ss.TstRecLays:
            plt.SetColParams(lnm, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        return plt

    def LogTstEpc(ss, dt):
        row = dt.Rows
        dt.SetNumRows(row + 1)

        # trl := ss.TstTrlLog
        # tix := etable.NewIndexView(trl)
        epc = ss.TestEnv.Epoch.Prv  # ?

        # note: this shows how to use agg methods to compute summary data from another
        # data table, instead of incrementing on the Sim
        dt.SetCellFloat("Run", row, float(ss.TestEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))

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
            ]
        )
        dt.SetFromSchema(sch, 0)

    def ConfigTstEpcPlot(ss, plt, dt):
        plt.Params.Title = "SIR Testing Epoch Plot"
        plt.Params.XAxisCol = "Epoch"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
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
        nlast = 1
        if nlast > epcix.Len() - 1:
            nlast = epcix.Len() - 1
        epcix.Indexes = epcix.Indexes[epcix.Len() - nlast :]

        params = ss.RunName()

        dt.SetCellFloat("Run", row, float(run))
        dt.SetCellString("Params", row, params)
        dt.SetCellFloat("FirstZero", row, float(ss.FirstZero))
        dt.SetCellFloat("SSE", row, agg.Mean(epcix, "SSE")[0])
        dt.SetCellFloat("AvgSSE", row, agg.Mean(epcix, "AvgSSE")[0])
        dt.SetCellFloat("PctErr", row, agg.Mean(epcix, "PctErr")[0])
        dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
        dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])

        runix = etable.NewIndexView(dt)
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
            [
                etable.Column("Run", etensor.INT64, go.nil, go.nil),
                etable.Column("Params", etensor.STRING, go.nil, go.nil),
                etable.Column("FirstZero", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("SSE", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("AvgSSE", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("PctErr", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("PctCor", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("CosDiff", etensor.FLOAT64, go.nil, go.nil),
            ]
        )
        dt.SetFromSchema(sch, 0)

    def ConfigRunPlot(ss, plt, dt):
        plt.Params.Title = "SIR Run Plot"
        plt.Params.XAxisCol = "Run"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams(
            "FirstZero", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0
        )  # default plot
        plt.SetColParams("SSE", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("PctErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        return plt

    def ConfigNetView(ss, nv):
        nv.ViewDefaults()

        labs = go.Slice_string(
            [
                "  A B C D ",
                " A B C D",
                " A B C D",
                "A B C D",
                "A B C D",
                " A B C D ",
                "  S I R ",
            ]
        )
        nv.ConfigLabels(labs)

        lays = go.Slice_string(
            ["Input", "PFCmnt", "PFCmntD", "PFCout", "PFCoutD", "Output", "CtrlInput"]
        )

        for li, lnm in enumerate(lays):
            ly = nv.LayerByName(lnm)
            lbl = nv.LabelByName(labs[li])
            lbl.Pose = ly.Pose
            lbl.Pose.Pos.Y += 0.08
            lbl.Pose.Pos.Z += 0.02
            lbl.Pose.Scale.SetMul(math32.Vector3(1, 0.3, 0.5))

    def ConfigGui(ss):
        """
        ConfigGui configures the GoGi gui interface for this simulation,
        """
        width = 1600
        height = 1200

        core.SetAppName("sir")
        core.SetAppAbout(
            'illustrates the dynamic gating of information into PFC active maintenance, by the basal ganglia (BG). It uses a simple Store-Ignore-Recall (SIR) task, where the BG system learns via phasic dopamine signals and trial-and-error exploration, discovering what needs to be stored, ignored, and recalled as a function of reinforcement of correct behavior, and learned reinforcement of useful working memory representations. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch10/sir/README.md">README.md on GitHub</a>.</p>'
        )

        win = core.NewMainWindow("sir", "SIR: PBWM", width, height)
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

        tbar.AddSeparator("log")

        tbar.AddAction(
            core.ActOpts(
                Label="Reset RunLog",
                Icon="reset",
                Tooltip="Resets the accumulated log of all Runs, which are tagged with the ParamSet used",
            ),
            recv,
            ResetRunLogCB,
        )

        tbar.AddSeparator("misc")

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
        #   win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
        #       FileViewOpenSVG(vp)
        #   })
        # fmen.Menu.AddSeparator("csep")
        # fmen.Menu.AddAction(core.ActOpts{Label: "Close Window", Shortcut: "Command+W"},
        #   win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
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
