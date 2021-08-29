#!/usr/local/bin/pyleabra -i

# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# use:
# pyleabra -i ra25.py 
# to run in gui interactive mode from the command line (or pyleabra, import ra25)
# see main function at the end for startup args

# abac explores the classic paired associates learning task in a
# cortical-like network, which exhibits catastrophic levels of
# interference.

from leabra import go, leabra, emer, relpos, eplot, env, agg, patgen, prjn, etable, efile, split, etensor, params, netview, rand, erand, gi, giv, pygiv, pyparams, mat32, metric, simat, pca, clust

import importlib as il  #il.reload(ra25) -- doesn't seem to work for reasons unknown
import io, sys, getopt
from datetime import datetime, timezone

# import numpy as np
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

def TestItemCB2(recv, send, sig, data):
    win = gi.Window(handle=recv)
    vp = win.WinViewport2D()
    dlg = gi.Dialog(handle=send)
    if sig != gi.DialogAccepted:
        return
    val = gi.StringPromptDialogValue(dlg)
    idxs = TheSim.TestEnv.Table.RowsByString("Name", val, True, True) # contains, ignoreCase
    if len(idxs) == 0:
        gi.PromptDialog(vp, gi.DlgOpts(Title="Name Not Found", Prompt="No patterns found containing: " + val), True, False, go.nil, go.nil)
    else:
        if not TheSim.IsRunning:
            TheSim.IsRunning = True
            print("testing index: %s" % idxs[0])
            TheSim.TestItem(idxs[0])
            TheSim.IsRunning = False
            vp.SetNeedsFullRender()

def TestItemCB(recv, send, sig, data):
    win = gi.Window(handle=recv)
    gi.StringPromptDialog(win.WinViewport2D(), "", "Test Item",
        gi.DlgOpts(Title="Test Item", Prompt="Enter the Name of a given input pattern to test (case insensitive, contains given string."), win, TestItemCB2)

def TestAllCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.ToolBar.UpdateActions()
        TheSim.RunTestAll()

def RepsAnalysisCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.RepsAnalysis()

def BuildNetCB(recv, send, sig, data):
    TheSim.Net.Build()
    TheSim.NetView.Config()
    TheSim.NetView.Update()
    TheSim.vp.SetNeedsFullRender()

def DefaultsCB(recv, send, sig, data):
    TheSim.Defaults()
    TheSim.Init()
    TheSim.UpdateClassView()
    TheSim.vp.SetNeedsFullRender()

def ResetRunLogCB(recv, send, sig, data):
    TheSim.RunLog.SetNumRows(0)
    TheSim.RunPlot.Update()

def NewRndSeedCB(recv, send, sig, data):
    TheSim.NewRndSeed()

def ReadmeCB(recv, send, sig, data):
    gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch8/abac/README.md")

def UpdtFuncNotRunning(act):
    act.SetActiveStateUpdt(not TheSim.IsRunning)
    
def UpdtFuncRunning(act):
    act.SetActiveStateUpdt(TheSim.IsRunning)

    
class Reps(pygiv.ClassViewObj):
    """
    Reps contains standard analysis of representations
    """

    def __init__(self):
        super(Reps, self).__init__()
        self.SimMat = simat.SimMat()
        self.SimMat.Init()
        self.SetTags("SimMat", 'view:"no-inline" desc:"similarity matrix"')
        self.PCAPlot = eplot.Plot2D()
        self.PCAPlot.InitName(self.PCAPlot, "PCAPlot") # any Ki obj needs this
        self.SetTags("PCAPlot", 'view:"no-inline" desc:"plot of pca data"')
        self.ClustPlot = eplot.Plot2D()
        self.ClustPlot.InitName(self.ClustPlot, "ClustPlot") # any Ki obj needs this
        self.SetTags("ClustPlot", 'view:"no-inline" desc:"cluster plot"')
        self.PCA = pca.PCA()
        self.PCA.Init()
        self.SetTags("PCA", 'view:"-" desc:"pca results"')
        self.PCAPrjn = etable.Table()
        self.SetTags("PCAPrjn", 'view:"-" desc:"pca projections onto eigenvectors"')

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
        self.HiddenInhibGi = float(1.8)
        self.SetTags("HiddenInhibGi", 'def:"1.8" desc:"hidden layer inhibition -- increase to make sparser"')
        self.WtInitVar = float(0.25)
        self.SetTags("WtInitVar", 'def:"0.25" desc:"random initial weight variance -- increase to make more random"')
        self.FmContext = float(1)
        self.SetTags("FmContext", 'def:"1" desc:"relative WtScale.Rel from Context layer"')
        self.XCalLLrn = float(0.0003)
        self.SetTags("XCalLLrn", 'min:"0" step:"0.0001" def:"0.0003" desc:"amount of Hebbian BCM learning based on AvgL long-term average activity -- increase to increase amount of hebbian"')
        self.Lrate = float(0.04)
        self.SetTags("Lrate", 'def:"0.04" desc:"learning rate"')
        self.Net = leabra.Network()
        self.SetTags("Net", 'view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"')
        self.ABPats = etable.Table()
        self.SetTags("ABPats", 'view:"no-inline" desc:"AB paired associate training patterns"')
        self.ACPats = etable.Table()
        self.SetTags("ACPats", 'view:"no-inline" desc:"AC paired associate training patterns"')
        self.TrnEpcLog = etable.Table()
        self.SetTags("TrnEpcLog", 'view:"no-inline" desc:"training epoch-level log data"')
        self.TstEpcLog = etable.Table()
        self.SetTags("TstEpcLog", 'view:"no-inline" desc:"testing epoch-level log data"')
        self.TstTrlLog = etable.Table()
        self.SetTags("TstTrlLog", 'view:"no-inline" desc:"testing trial-level log data"')
        self.RunLog = etable.Table()
        self.SetTags("RunLog", 'view:"no-inline" desc:"summary log of each run"')
        self.RunStats = etable.Table()
        self.SetTags("RunStats", 'view:"no-inline" desc:"aggregate stats on all runs"')
        self.TstStats = etable.Table()
        self.SetTags("TstStats", 'view:"no-inline" desc:"testing stats"')
        self.Params = params.Sets()
        self.SetTags("Params", 'view:"no-inline" desc:"full collection of param sets"')
        self.ParamSet = str()
        self.SetTags("ParamSet", 'view:"-" desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set -- can use multiple names separated by spaces (don\'t put spaces in ParamSet names!)"')
        self.MaxRuns = int(10)
        self.SetTags("MaxRuns", 'desc:"maximum number of model runs to perform"')
        self.MaxEpcs = int(100)
        self.SetTags("MaxEpcs", 'desc:"maximum number of epochs to run per model run"')
        self.NZeroStop = int(1)
        self.SetTags("NZeroStop", 'desc:"if a positive number, training will stop after this many epochs with zero SSE"')
        self.TrainEnv = env.FixedTable()
        self.SetTags("TrainEnv", 'desc:"Training environment -- contains everything about iterating over input / output patterns over training"')
        self.TestEnv = env.FixedTable()
        self.SetTags("TestEnv", 'desc:"Training environment -- contains everything about iterating over input / output patterns over training"')
        self.Time = leabra.Time()
        self.SetTags("Time", 'desc:"leabra timing parameters and state"')
        self.ViewOn = True
        self.SetTags("ViewOn", 'desc:"whether to update the network view while running"')
        self.TrainUpdt = leabra.TimeScales.Quarter
        self.SetTags("TrainUpdt", 'desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"')
        self.TestUpdt = leabra.TimeScales.Quarter
        self.SetTags("TestUpdt", 'desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"')
        self.TestInterval = int(1)
        self.SetTags("TestInterval", 'desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"')
        self.TstRecLays = go.Slice_string(["Hidden"])
        self.SetTags("TstRecLays", 'desc:"names of layers to record activations etc of during testing"')
        self.HiddenReps = Reps()
        self.SetTags("HiddenReps", 'view:"inline" desc:"representational analysis of Hidden layer, sorted by relationship"')

        # statistics: note use float64 as that is best for etable.Table
        self.TestNm = str()
        self.SetTags("TestNm", 'inactive:"+" desc:"what set of patterns are we currently testing"')
        self.TrlErr = float()
        self.SetTags("TrlErr", 'inactive:"+" desc:"1 if trial was error, 0 if correct -- based on SSE = 0 (subject to .5 unit-wise tolerance)"')
        self.TrlSSE = float()
        self.SetTags("TrlSSE", 'inactive:"+" desc:"current trial\'s sum squared error"')
        self.TrlAvgSSE = float()
        self.SetTags("TrlAvgSSE", 'inactive:"+" desc:"current trial\'s average sum squared error"')
        self.TrlCosDiff = float()
        self.SetTags("TrlCosDiff", 'inactive:"+" desc:"current trial\'s cosine difference"')
        self.EpcSSE = float()
        self.SetTags("EpcSSE", 'inactive:"+" desc:"last epoch\'s total sum squared error"')
        self.EpcAvgSSE = float()
        self.SetTags("EpcAvgSSE", 'inactive:"+" desc:"last epoch\'s average sum squared error (average over trials, and over units within layer)"')
        self.EpcPctErr = float()
        self.SetTags("EpcPctErr", 'inactive:"+" desc:"last epoch\'s average TrlErr"')
        self.EpcPctCor = float()
        self.SetTags("EpcPctCor", 'inactive:"+" desc:"1 - last epoch\'s average TrlErr"')
        self.EpcCosDiff = float()
        self.SetTags("EpcCosDiff", 'inactive:"+" desc:"last epoch\'s average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"')
        self.FirstZero = int()
        self.SetTags("FirstZero", 'inactive:"+" desc:"epoch at when SSE first went to zero"')
        self.NZero = int()
        self.SetTags("NZero", 'inactive:"+" desc:"number of epochs in a row with zero SSE"')

        # internal state - view:"-"
        self.SumErr = float()
        self.SetTags("SumErr", 'view:"-" inactive:"+" desc:"sum to increment as we go through epoch"')
        self.SumSSE = float()
        self.SetTags("SumSSE", 'view:"-" inactive:"+" desc:"sum to increment as we go through epoch"')
        self.SumAvgSSE = float()
        self.SetTags("SumAvgSSE", 'view:"-" inactive:"+" desc:"sum to increment as we go through epoch"')
        self.SumCosDiff = float()
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
        self.TstTrlPlot = 0
        self.SetTags("TstTrlPlot", 'view:"-" desc:"the test-trial plot"')
        self.RunPlot = 0
        self.SetTags("RunPlot", 'view:"-" desc:"the run plot"')
        self.TrnEpcFile = 0
        self.SetTags("TrnEpcFile", 'view:"-" desc:"log file"')
        self.RunFile = 0
        self.SetTags("RunFile", 'view:"-" desc:"log file"')
        self.TstNms = go.Slice_string(["AB", "AC"])
        self.SetTags("TstNms", 'view:"-" desc:"names of test tables"')
        self.TstStatNms = go.Slice_string(["Err"])
        self.SetTags("TstStatNms", 'view:"-" desc:"names of test stats"')
        self.ValsTsrs = {}
        self.SetTags("ValsTsrs", 'view:"-" desc:"for holding layer values"')
        self.IsRunning = False
        self.SetTags("IsRunning", 'view:"-" desc:"true if sim is running"')
        self.StopNow = False
        self.SetTags("StopNow", 'view:"-" desc:"flag to stop running"')
        self.NeedsNewRun = False
        self.SetTags("NeedsNewRun", 'view:"-" desc:"flag to initialize NewRun if last one finished"')
        self.RndSeed = int(10)
        self.SetTags("RndSeed", 'view:"-" desc:"the current random seed"')
        self.vp  = 0
        self.SetTags("vp", 'view:"-" desc:"viewport"')

    def InitParams(ss):
        """
        Sets the default set of parameters -- Base is always applied, and others can be optionally
        selected to apply on top of that
        """
        ss.Params.OpenJSON("abac.params")

    def Defaults(ss):
        ss.HiddenInhibGi = 1.8
        ss.WtInitVar = 0.25
        ss.FmContext = 1
        ss.XCalLLrn = 0.0003
        ss.Lrate = 0.04

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
        ss.ConfigRunLog(ss.RunLog)

    def ConfigEnv(ss):
        if ss.MaxRuns == 0:
            ss.MaxRuns = 10
        if ss.MaxEpcs == 0: # allow user override
            ss.MaxEpcs = 100
            ss.NZeroStop = 1

        ss.TrainEnv.Nm = "TrainEnv"
        ss.TrainEnv.Dsc = "training params and state"
        ss.TrainEnv.Table = etable.NewIdxView(ss.ABPats)
        ss.TrainEnv.Validate()
        ss.TrainEnv.Run.Max = ss.MaxRuns # note: we are not setting epoch max -- do that manually

        ss.TestEnv.Nm = "TestEnv"
        ss.TestEnv.Dsc = "testing params and state"
        ss.TestEnv.Table = etable.NewIdxView(ss.ABPats)
        ss.TestEnv.Validate()
        ss.TestEnv.Run.Max = ss.MaxRuns # note: we are not setting epoch max -- do that manually

        ss.TrainEnv.Init(0)
        ss.TestEnv.Init(0)

    def ConfigNet(ss, net):
        net.InitName(net, "ABAC")
        inp = net.AddLayer2D("Input", 5, 5, emer.Input)
        ctx = net.AddLayer2D("Context", 5, 5, emer.Input)
        hid = net.AddLayer2D("Hidden", 10, 15, emer.Hidden)
        out = net.AddLayer2D("Output", 5, 5, emer.Target)

        full = prjn.NewFull()
        net.ConnectLayers(inp, hid, full, emer.Forward)
        net.ConnectLayers(ctx, hid, full, emer.Forward)
        net.BidirConnectLayersPy(hid, out, full)

        ctx.SetRelPos(relpos.Rel(Rel= relpos.RightOf, Other= "Input", YAlign= relpos.Front, Space= 2))
        hid.SetRelPos(relpos.Rel(Rel= relpos.Above, Other= "Input", YAlign= relpos.Front, XAlign= relpos.Left))
        out.SetRelPos(relpos.Rel(Rel= relpos.Above, Other= "Hidden", YAlign= relpos.Front, XAlign= relpos.Middle))

        net.Defaults()
        ss.SetParams("Network", False) # only set Network params
        net.Build()
        net.InitWts()

    def Init(ss):
        """
        Init restarts the run, and initializes everything, including network weights
        and resets the epoch log table
        """
        rand.Seed(ss.RndSeed)
        ss.TrainEnv.Init(0)
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
            return "Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t" % (ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Trial.Cur, ss.Time.Cycle, ss.TrainEnv.TrialName.Cur)
        else:
            return "Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t" % (ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TestEnv.Trial.Cur, ss.Time.Cycle, ss.TestEnv.TrialName.Cur)

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

        ss.Net.AlphaCycInit(train)
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
            ss.Net.DWt()
        if ss.ViewOn and viewUpdt == leabra.AlphaCycle:
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

        lays = go.Slice_string(["Input", "Context", "Output"])
        for lnm in lays :
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
            if ss.ViewOn and ss.TrainUpdt.value > leabra.AlphaCycle:
                ss.UpdateView(True)
            if ss.TestInterval > 0 and epc%ss.TestInterval == 0: # note: epc is *next* so won't trigger first time
                ss.TestAll()
            learned = (ss.NZeroStop > 0 and ss.NZero >= ss.NZeroStop)
            if ss.TrainEnv.Table.Table.MetaData["name"] == "AB" and (learned or epc == 50):
                ss.TrainEnv.Table = etable.NewIdxView(ss.ACPats)
                learned = False
            if learned or epc >= ss.MaxEpcs:
                # done with training..
                ss.RunEnd()
                if ss.TrainEnv.Run.Incr(): # we are done!
                    ss.StopNow = True
                    return
                else:
                    ss.NeedsNewRun = True
                    return

        ss.ApplyInputs(ss.TrainEnv)
        ss.AlphaCyc(True)   # train
        ss.TrialStats(True) # accumulate

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
        ss.TrainEnv.Table = etable.NewIdxView(ss.ABPats)
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

        ss.SumSSE = 0
        ss.SumAvgSSE = 0
        ss.SumCosDiff = 0
        ss.SumErr = 0
        ss.FirstZero = -1
        ss.NZero = 0

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
        ss.TrlSSE = out.SSE(0.5) # 0.5 = per-unit tolerance -- right side of .5
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
        SaveWeights saves the network weights -- when called with giv.CallMethod
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
            if ss.ViewOn and ss.TestUpdt.value > leabra.AlphaCycle:
                ss.UpdateView(False)
            if returnOnChg:
                return

        ss.ApplyInputs(ss.TestEnv)
        ss.AlphaCyc(False)
        ss.TrialStats(False)
        ss.LogTstTrl(ss.TstTrlLog)

    def TestItem(ss, idx):
        """
        TestItem tests given item which is at given index in test item list
        """
        cur = ss.TestEnv.Trial.Cur
        ss.TestEnv.Trial.Cur = idx
        ss.TestEnv.SetTrialName()
        ss.ApplyInputs(ss.TestEnv)
        ss.AlphaCyc(False)
        ss.TrialStats(False)
        ss.TestEnv.Trial.Cur = cur

    def TestAll(ss):
        """
        TestAll runs through the full set of testing items
        """
        ss.TestNm = "AB"
        ss.TestEnv.Table = etable.NewIdxView(ss.ABPats)
        ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
        while True:
            ss.TestTrial(True)
            chg = env.CounterChg(ss.TestEnv, env.Epoch)
            if chg or ss.StopNow:
                break
        if not ss.StopNow:
            ss.TestNm = "AC"
            ss.TestEnv.Table = etable.NewIdxView(ss.ACPats)
            ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
            while True:
                ss.TestTrial(True)
                chg = env.CounterChg(ss.TestEnv, env.Epoch)
                if chg or ss.StopNow:
                    break

            ss.LogTstEpc(ss.TstEpcLog)

    def RunTestAll(ss):
        """
        RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
        """
        ss.StopNow = False
        ss.TestAll()
        ss.Stopped()

    def RepsAnalysis(ss):
        """
        RepsAnalysis does a full test and then runs tests of representations
        """
        ss.RunTestAll()

        rels = etable.NewIdxView(ss.TstTrlLog)
        rels.SortCol(ss.TstTrlLog.ColIdx("TrialName"), True)
        ss.HiddenReps.SimMat.TableColStd(rels, "Hidden", "TrialName", True, metric.Correlation)
        ss.HiddenReps.PCA.TableColStd(rels, "Hidden", metric.Covariance)
        ss.HiddenReps.PCA.ProjectColToTable(ss.HiddenReps.PCAPrjn, rels, "Hidden", "TrialName", go.Slice_int([0, 1]))
        ss.ConfigPCAPlot(ss.HiddenReps.PCAPlot, ss.HiddenReps.PCAPrjn, "Hidden")
        ss.ClustPlot(ss.HiddenReps.ClustPlot, rels, "Hidden")

        ss.Stopped()

    def ConfigPCAPlot(ss, plt, dt, nm):
        plt.Params.Title = "AB-AC PCA Plot: " + nm
        plt.Params.XAxisCol = "Prjn0"
        plt.SetTable(dt)
        plt.Params.Lines = False
        plt.Params.Points = True

        plt.SetColParams("TrialName", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Prjn0", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Prjn1", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)

    def ClustPlot(ss, plt, ix, colNm):
        """
        ClustPlot does one cluster plot on given table column
        """
        nm = ix.Table.MetaData["name"]
        smat = simat.SimMat()
        smat.TableColStd(ix, colNm, "TrialName", False, metric.Euclidean)
        pt = etable.Table()
        clust.Plot(pt, clust.GlomStd(smat, clust.Contrast), smat)
        plt.InitName(plt, colNm)
        plt.Params.Title = "Cluster Plot of: " + nm + " " + colNm
        plt.Params.XAxisCol = "X"
        plt.SetTable(pt)

        plt.SetColParams("X", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Y", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Label", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)

    def SetParams(ss, sheet, setMsg):
        """
        SetParams sets the params for "Base" and then current ParamSet.
        If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
        otherwise just the named sheet
        if setMsg = true then we output a message for each param that was set.
        """
        if sheet == "":
            ss.Params.ValidateSheets(go.Slice_string(["Network", "Sim"]))

        spo = ss.Params.SetByName("Base").SheetByName("Network").SelByName("Prjn")
        spo.Params.SetParamByName("Prjn.WtInit.Var", ("%g" % ss.WtInitVar))
        spo.Params.SetParamByName("Prjn.Learn.XCal.LLrn", ("%g" % ss.XCalLLrn))
        spo.Params.SetParamByName("Prjn.Learn.Lrate", ("%g" % ss.Lrate))

        ss.SetParamsSet("Base", sheet, setMsg)
        if ss.ParamSet != "" and ss.ParamSet != "Base":
            sps = ss.ParamSet.split()
            for ps in sps:
                ss.SetParamsSet(ps, sheet, setMsg)
        hid = leabra.Layer(ss.Net.LayerByName("Hidden"))
        hid.Inhib.Layer.Gi = ss.HiddenInhibGi

        fmc = leabra.LeabraPrjn(hid.RcvPrjns.SendName("Context")).AsLeabra()
        fmc.WtScale.Rel = ss.FmContext


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

    def OpenPat(ss, dt, fname, name, desc):
        dt.OpenCSV(fname, etable.Tab)
        dt.SetMetaData("name", name)
        dt.SetMetaData("desc", desc)

    def OpenPats(ss):
        ss.OpenPat(ss.ABPats, "ab_pats.tsv", "AB", "AB Training patterns")
        ss.OpenPat(ss.ACPats, "ac_pats.tsv", "AC", "AC Training patterns")

    def ValsTsr(ss, name):
        """
        ValsTsr gets value tensor of given name, creating if not yet made
        """
        if name in ss.ValsTsrs:
            return ss.ValsTsrs[name]
        tsr = etensor.Float32()
        ss.ValsTsrs[name] = tsr
        return tsr

    def LogTrnEpc(ss, dt):
        """
        LogTrnEpc adds data from current epoch to the TrnEpcLog table.
        computes epoch averages prior to logging.

    # this is triggered by increment so use previous value
        """
        row = dt.Rows
        dt.SetNumRows(row + 1)

        epc = ss.TrainEnv.Epoch.Prv
        nt = float(ss.TrainEnv.Table.Len()) # number of trials in view

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

        dt.SetCellFloat("Run", row, float(ss.TrainEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellFloat("SSE", row, ss.EpcSSE)
        dt.SetCellFloat("AvgSSE", row, ss.EpcAvgSSE)
        dt.SetCellFloat("PctErr", row, ss.EpcPctErr)
        dt.SetCellFloat("PctCor", row, ss.EpcPctCor)
        dt.SetCellFloat("CosDiff", row, ss.EpcCosDiff)

        # note: essential to use Go version of update when called from another goroutine
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
            [etable.Column("Run", etensor.INT64, go.nil, go.nil),
            etable.Column("Epoch", etensor.INT64, go.nil, go.nil),
            etable.Column("SSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("AvgSSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("PctErr", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("PctCor", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("CosDiff", etensor.FLOAT64, go.nil, go.nil)]
        )
        dt.SetFromSchema(sch, 0)

    def ConfigTrnEpcPlot(ss, plt, dt):
        plt.Params.Title = "AB-AC Epoch Plot"
        plt.Params.XAxisCol = "Epoch"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

        return plt

    def LogTstTrl(ss, dt):
        """
        LogTstTrl adds data from current trial to the TstTrlLog table.
    # this is triggered by increment so use previous value
        log always contains number of testing items
        """
        epc = ss.TrainEnv.Epoch.Prv
        trl = ss.TestEnv.Trial.Cur

        row = dt.Rows
        if ss.TestNm == "AB" and trl == 0: # reset at start
            row = 0
            dt.SetNumRows(row)
        if dt.Rows <= row:
            dt.SetNumRows(row + 1)

        dt.SetCellFloat("Run", row, float(ss.TrainEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellString("TestNm", row, ss.TestNm)
        dt.SetCellFloat("Trial", row, float(trl))
        dt.SetCellString("TrialName", row, ss.TestEnv.TrialName.Cur)
        dt.SetCellFloat("Err", row, ss.TrlErr)
        dt.SetCellFloat("SSE", row, ss.TrlSSE)
        dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
        dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

        for lnm in ss.TstRecLays :
            tsr = ss.ValsTsr(lnm)
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            ly.UnitValsTensor(tsr, "ActM") # get minus phase act
            dt.SetCellTensor(lnm, row, tsr)

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
            etable.Column("TestNm", etensor.STRING, go.nil, go.nil),
            etable.Column("Trial", etensor.INT64, go.nil, go.nil),
            etable.Column("TrialName", etensor.STRING, go.nil, go.nil),
            etable.Column("Err", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("SSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("AvgSSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("CosDiff", etensor.FLOAT64, go.nil, go.nil)]
        )
        for lnm in ss.TstRecLays :
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            sch.append( etable.Column(lnm, etensor.FLOAT64, ly.Shp.Shp, go.nil))
        dt.SetFromSchema(sch, 0)

    def ConfigTstTrlPlot(ss, plt, dt):
        plt.Params.Title = "AB-AC Test Trial Plot"
        plt.Params.XAxisCol = "Trial"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("TestNm", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Err", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("SSE", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0) # default plot
        plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

        for lnm in ss.TstRecLays :
            plt.SetColParams(lnm, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
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
        dt.SetCellFloat("SSE", row, agg.Sum(tix, "SSE")[0])
        dt.SetCellFloat("AvgSSE", row, agg.Mean(tix, "AvgSSE")[0])
        dt.SetCellFloat("PctErr", row, agg.Mean(tix, "Err")[0])
        dt.SetCellFloat("PctCor", row, 1-agg.Mean(tix, "Err")[0])
        dt.SetCellFloat("CosDiff", row, agg.Mean(tix, "CosDiff")[0])

        trix = etable.NewIdxView(trl)
        spl = split.GroupBy(trix, go.Slice_string(["TestNm"]))
        for ts in ss.TstStatNms :
            split.Agg(spl, ts, agg.AggMean)
        ss.TstStats = spl.AggsToTable(etable.ColNameOnly)

        for ri in range(ss.TstStats.Rows):
            tst = ss.TstStats.CellString("TestNm", ri)
            for ts in ss.TstStatNms :
                dt.SetCellFloat(tst+" "+ts, row, ss.TstStats.CellFloat(ts, ri))

        # note: essential to use Go version of update when called from another goroutine
        ss.TstEpcPlot.GoUpdate()

    def ConfigTstEpcLog(ss, dt):
        dt.SetMetaData("name", "TstEpcLog")
        dt.SetMetaData("desc", "Summary stats for testing trials")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        sch = etable.Schema(
            [etable.Column("Run", etensor.INT64, go.nil, go.nil),
            etable.Column("Epoch", etensor.INT64, go.nil, go.nil),
            etable.Column("SSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("AvgSSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("PctErr", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("PctCor", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("CosDiff", etensor.FLOAT64, go.nil, go.nil)]
        )
        for tn in ss.TstNms :
            for ts in ss.TstStatNms :
                sch.append( etable.Column(tn + " " + ts, etensor.FLOAT64, go.nil, go.nil))
        dt.SetFromSchema(sch, 0)

    def ConfigTstEpcPlot(ss, plt, dt):
        plt.Params.Title = "AB-AC Testing Epoch Plot"
        plt.Params.XAxisCol = "Epoch"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0) # default plot
        plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("PctErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        for tn in ss.TstNms :
            for ts in ss.TstStatNms :
                plt.SetColParams(tn+" "+ts, eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) # default plot

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
        epcix = etable.NewIdxView(epclog)
        # compute mean over last N epochs for run level
        nlast = 1
        if nlast > epcix.Len()-1:
            nlast = epcix.Len() - 1
        epcix.Idxs = epcix.Idxs[epcix.Len()-nlast:]

        tstlog = ss.TstEpcLog
        tstix = etable.NewIdxView(tstlog)
        # compute mean over last N epochs for run level
        if nlast > tstix.Len()-1:
            nlast = tstix.Len() - 1
        tstix.Idxs = tstix.Idxs[tstix.Len()-nlast:]

        params = "hid_gi: %g, wt_var: %g, fm_ctxt: %g, lrate: %g" % (ss.HiddenInhibGi, ss.WtInitVar, ss.FmContext, ss.Lrate)

        dt.SetCellFloat("Run", row, float(run))
        dt.SetCellString("Params", row, params)
        dt.SetCellFloat("FirstZero", row, float(ss.FirstZero))
        dt.SetCellFloat("SSE", row, agg.Mean(epcix, "SSE")[0])
        dt.SetCellFloat("AvgSSE", row, agg.Mean(epcix, "AvgSSE")[0])
        dt.SetCellFloat("PctErr", row, agg.Mean(epcix, "PctErr")[0])
        dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
        dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])
        dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
        dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])

        for tn in ss.TstNms :
            for ts in ss.TstStatNms :
                nm = tn + " " + ts
                dt.SetCellFloat(nm, row, agg.Mean(tstix, nm)[0])

        runix = etable.NewIdxView(dt)
        spl = split.GroupBy(runix, go.Slice_string(["Params"]))
        for tn in ss.TstNms :
            nm = tn + " " + "Err"
            split.Desc(spl, nm)
        split.Desc(spl, "FirstZero")
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
        for tn in ss.TstNms :
            for ts in ss.TstStatNms :
                sch.append( etable.Column(tn + " " + ts, etensor.FLOAT64, go.nil, go.nil))
        dt.SetFromSchema(sch, 0)

    def ConfigRunPlot(ss, plt, dt):
        plt.Params.Title = "AB-AC Run Plot"
        plt.Params.XAxisCol = "Run"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("FirstZero", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("PctErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        for tn in ss.TstNms :
            for ts in ss.TstStatNms :
                plt.SetColParams(tn+" "+ts, eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) # default plot

        return plt

    def ConfigNetView(ss, nv):
        nv.ViewDefaults()
        nv.Scene().Camera.Pose.Pos.Set(0, .95, 3.2) # more "head on" than default which is more "top down"
        nv.Scene().Camera.LookAt(mat32.Vec3(0, 0, 0), mat32.Vec3(0, 1, 0))

    def ConfigGui(ss):
        """
        ConfigGui configures the GoGi gui interface for this simulation,
        """
        width = 1600
        height = 1200

        gi.SetAppName("abac")
        gi.SetAppAbout('explores the classic paired associates learning task in a cortical-like network, which exhibits catastrophic levels of interference. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch8/abac/README.md">README.md on GitHub</a>.</p>')

        win = gi.NewMainWindow("abac", "AB-AC Interference", width, height)
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
        tv.AddTab(plt, "TstTrlPlot")
        ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

        plt = eplot.Plot2D()
        tv.AddTab(plt, "TstEpcPlot")
        ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.TstEpcLog)

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

        tbar.AddAction(gi.ActOpts(Label="Step Run", Icon="fast-fwd", Tooltip="Advances one full training Run at a time.", UpdateFunc=UpdtFuncNotRunning), recv, StepRunCB)
        
        tbar.AddSeparator("test")
        
        tbar.AddAction(gi.ActOpts(Label="Test Trial", Icon="step-fwd", Tooltip="Runs the next testing trial.", UpdateFunc=UpdtFuncNotRunning), recv, TestTrialCB)
        
        tbar.AddAction(gi.ActOpts(Label="Test All", Icon="fast-fwd", Tooltip="Tests all of the testing trials.", UpdateFunc=UpdtFuncNotRunning), recv, TestAllCB)

        tbar.AddAction(gi.ActOpts(Label= "Reps Analysis", Icon= "fast-fwd", Tooltip= "Does an All Test All and analyzes the resulting Hidden and AgentCode activations.", UpdateFunc=UpdtFuncNotRunning), recv, RepsAnalysisCB)
        
        tbar.AddSeparator("log")
        
        # tbar.AddAction(gi.ActOpts(Label= "Env", Icon= "gear", Tooltip= "select training input patterns: AB or AC."), win.This(),
        #     funcrecv, send, sig, data:
        #         giv.CallMethod(ss, "SetEnv", vp))

        tbar.AddAction(gi.ActOpts(Label="Reset RunLog", Icon="reset", Tooltip="Resets the accumulated log of all Runs, which are tagged with the ParamSet used"), recv, ResetRunLogCB)

        tbar.AddSeparator("misc")
        
        tbar.AddAction(gi.ActOpts(Label= "Build Net", Icon= "new", Tooltip= "Build network -- do this if any sizes have changed"), recv, BuildNetCB)
        
        tbar.AddAction(gi.ActOpts(Label="New Seed", Icon="new", Tooltip="Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."), recv, NewRndSeedCB)

        tbar.AddAction(gi.ActOpts(Label= "Defaults", Icon= "update", Tooltip= "Restore initial default parameters.", UpdateFunc=UpdtFuncNotRunning), recv, DefaultsCB)
        
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


