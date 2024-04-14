#!/usr/local/bin/pyleabra -i

# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# use:
# pyleabra -i ra25.py
# to run in gui interactive mode from the command line (or pyleabra, import ra25)
# see main function at the end for startup args

# sem is trained using Hebbian learning on paragraphs from an early draft
# of the *Computational Explorations..* textbook, allowing it to learn about
# the overall statistics of when different words co-occur with other words,
# and thereby learning a surprisingly capable (though clearly imperfect)
# level of semantic knowlege about the topics covered in the textbook.
# This replicates the key results from the Latent Semantic Analysis
# research by Landauer and Dumais (1997).

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

from sem_env import SemEnv

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


def QuizAllCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.ToolBar.UpdateActions()
        TheSim.RunQuizAll()


def ClustPlotsCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.ClustPlots()


def ResetRunLogCB(recv, send, sig, data):
    TheSim.RunLog.SetNumRows(0)
    TheSim.RunPlot.Update()


def OpenWtsCB(recv, send, sig, data):
    TheSim.OpenWts()


def WtWordsCB(recv, send, sig, data):
    TheSim.WtWords()


def NewRndSeedCB(recv, send, sig, data):
    TheSim.NewRndSeed()


def ReadmeCB(recv, send, sig, data):
    core.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch9/sem/README.md")


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
        self.Words1 = "attention"
        self.SetTags("Words1", 'desc:"space-separated words to test the network with"')
        self.Words2 = "binding"
        self.SetTags("Words2", 'desc:"space-separated words to test the network with"')
        self.ExcitLateralScale = float(0.05)
        self.SetTags(
            "ExcitLateralScale",
            'def:"0.05" desc:"excitatory lateral (recurrent) WtScale.Rel value"',
        )
        self.InhibLateralScale = float(0.05)
        self.SetTags(
            "InhibLateralScale",
            'def:"0.05" desc:"inhibitory lateral (recurrent) WtScale.Abs value"',
        )
        self.ExcitLateralLearn = True
        self.SetTags(
            "ExcitLateralLearn",
            'def:"true" desc:"do excitatory lateral (recurrent) connections learn?"',
        )
        self.WtWordsThr = float(0.75)
        self.SetTags(
            "WtWordsThr",
            'def:"0.75" desc:"threshold for weight strength for including in WtWords"',
        )
        self.Net = leabra.Network()
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
        self.TstQuizLog = etable.Table()
        self.SetTags(
            "TstQuizLog", 'view:"no-inline" desc:"testing quiz epoch-level log data"'
        )
        self.TstTrlLog = etable.Table()
        self.SetTags(
            "TstTrlLog", 'view:"no-inline" desc:"testing trial-level log data"'
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
        self.Tag = str()
        self.SetTags(
            "Tag",
            'desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"',
        )
        self.MaxRuns = int(1)
        self.SetTags("MaxRuns", 'desc:"maximum number of model runs to perform"')
        self.MaxEpcs = int(50)
        self.SetTags("MaxEpcs", 'desc:"maximum number of epochs to run per model run"')
        self.NZeroStop = int(-1)
        self.SetTags(
            "NZeroStop",
            'desc:"if a positive number, training will stop after this many epochs with zero SSE"',
        )
        self.TrainEnv = SemEnv()
        self.SetTags("TrainEnv", 'desc:"Training environment -- training paragraphs"')
        self.TestEnv = SemEnv()
        self.SetTags(
            "TestEnv", 'desc:"Testing environment -- manages iterating over testing"'
        )
        self.QuizEnv = SemEnv()
        self.SetTags(
            "QuizEnv", 'desc:"Quiz environment -- manages iterating over testing"'
        )
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
        self.LayStatNms = go.Slice_string(["Hidden"])
        self.SetTags(
            "LayStatNms",
            'desc:"names of layers to collect more detailed stats on (avg act, etc)"',
        )

        # statistics: note use float64 as that is best for etable.Table
        self.TstWords = str()
        self.SetTags(
            "TstWords", 'inactive:"+" desc:"words that were tested (short form)"'
        )
        self.TstWordsCorrel = float()
        self.SetTags(
            "TstWordsCorrel",
            'inactive:"+" desc:"correlation between hidden pattern for Words1 vs. Words2"',
        )
        self.TstQuizPctCor = float()
        self.SetTags(
            "TstQuizPctCor", 'inactive:"+" desc:"proportion correct for the quiz"'
        )
        self.EpcPerTrlMSec = float()
        self.SetTags(
            "EpcPerTrlMSec",
            'view:"-" desc:"how long did the epoch take per trial in wall-clock milliseconds"',
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
        self.TstQuizPlot = 0
        self.SetTags("TstQuizPlot", 'view:"-" desc:"the testing quiz epoch plot"')
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
        self.InQuiz = False
        self.SetTags("InQuiz", 'view:"-" desc:"true if in quiz"')
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
        ss.Params.OpenJSON("sem.params")
        ss.Defaults()

    def Defaults(ss):
        ss.Words1 = "attention"
        ss.Words2 = "binding"
        ss.ExcitLateralScale = 0.05
        ss.InhibLateralScale = 0.05
        ss.ExcitLateralLearn = True
        ss.WtWordsThr = 0.75

    def Config(ss):
        """
        Config configures all the elements using the standard functions
        """
        ss.InitParams()
        ss.ConfigEnv()
        ss.ConfigNet(ss.Net)
        ss.ConfigTrnEpcLog(ss.TrnEpcLog)
        ss.ConfigTstEpcLog(ss.TstEpcLog)
        ss.ConfigTstQuizLog(ss.TstQuizLog)
        ss.ConfigTstTrlLog(ss.TstTrlLog)
        ss.ConfigRunLog(ss.RunLog)

    def ConfigEnv(ss):
        if ss.MaxRuns == 0:
            ss.MaxRuns = 1
        if ss.MaxEpcs == 0:  # allow user override
            ss.MaxEpcs = 50
            ss.NZeroStop = -1

        ss.TrainEnv.Nm = "TrainEnv"
        ss.TrainEnv.Dsc = "training params and state"
        ss.TrainEnv.Defaults()
        ss.TrainEnv.OpenTexts(["cecn_lg_f5.text"])
        ss.TrainEnv.OpenWords("cecn_lg_f5.words")  # could also compute from words
        ss.TrainEnv.Validate()
        ss.TrainEnv.Run.Max = (
            ss.MaxRuns
        )  # note: we are not setting epoch max -- do that manually

        ss.TestEnv.Nm = "TestEnv"
        ss.TestEnv.Dsc = "testing env: for Words1, 2"
        ss.TestEnv.Sequential = True
        ss.TestEnv.OpenWords("cecn_lg_f5.words")
        ss.TestEnv.SetParas([ss.Words1, ss.Words2])
        ss.TestEnv.Validate()

        ss.QuizEnv.Nm = "QuizEnv"
        ss.QuizEnv.Dsc = "quiz environment"
        ss.QuizEnv.Sequential = True
        ss.QuizEnv.OpenWords("cecn_lg_f5.words")
        ss.QuizEnv.OpenTexts(["quiz.text"])
        ss.QuizEnv.Validate()

        ss.TrainEnv.Init(0)
        ss.TestEnv.Init(0)
        ss.QuizEnv.Init(0)

    def ConfigNet(ss, net):
        net.InitName(net, "Sem")
        inl = net.AddLayer2D("Input", 43, 45, emer.Input)
        hid = net.AddLayer2D("Hidden", 20, 20, emer.Hidden)

        full = prjn.NewFull()
        net.ConnectLayers(inl, hid, full, emer.Forward)

        circ = prjn.NewCircle()
        circ.TopoWts = True
        circ.Radius = 4
        circ.Sigma = 0.75

        rec = net.ConnectLayers(hid, hid, circ, emer.Lateral)
        rec.SetClass("ExciteLateral")

        inh = net.ConnectLayers(hid, hid, full, emer.Inhib)
        inh.SetClass("InhibLateral")

        net.Defaults()
        ss.SetParams("Network", False)  # only set Network params
        net.Build()
        ss.InitWts(net)

    def InitWts(ss, net):
        net.InitTopoScales()  # needed for gaussian topo Circle wts
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
        elif ss.InQuiz:
            return "Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t" % (
                ss.TrainEnv.Run.Cur,
                ss.TrainEnv.Epoch.Cur,
                ss.QuizEnv.Trial.Cur,
                ss.Time.Cycle,
                ss.QuizEnv.String(),
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

        ss.SetInputActAvg(ss.Net)  # needs to track actual external input

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

    def SetInputActAvg(ss, net):
        """
        Sets Input layer Inhib.ActAvg.Init from ext input
        """
        nin = 0
        inp = leabra.Layer(net.LayerByName("Input"))
        nn = len(inp.Neurons)
        for ni in range(nn):
            nrn = inp.Neurons[ni]
            if nrn.Ext > 0:
                nin += 1
        if nin > 0:
            avg = float(nin) / float(inp.Shp.Len())
            inp.Inhib.ActAvg.Init = avg

    def ApplyInputs(ss, en):
        """
        ApplyInputs applies input patterns from given envirbonment.
        It is good practice to have this be a separate method with appropriate
        args so that it can be used for various different contexts
        (training, testing, etc).
        """
        ss.Net.InitExt()

        lays = go.Slice_string(["Input"])
        for lnm in lays:
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            pats = en.State(ly.Nm)
            if pats != 0:
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
            ss.LogTrnEpc(ss.TrnEpcLog)
            if ss.ViewOn and ss.TrainUpdate.value > leabra.AlphaCycle:
                ss.UpdateView(True)
            if epc >= ss.MaxEpcs:
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
        if ss.SaveWts:
            fnm = ss.WeightsFileName()
            print("Saving Weights to: %s\n" % fnm)
            ss.Net.SaveWtsJSON(core.FileName(fnm))

    def NewRun(ss):
        """
        NewRun intializes a new run of the model, using the TrainEnv.Run counter
        for the new run value
        """
        run = ss.TrainEnv.Run.Cur
        ss.TrainEnv.Init(run)
        ss.TestEnv.Init(run)
        ss.QuizEnv.Init(run)
        ss.Time.Reset()
        ss.InitWts(ss.Net)
        ss.InitStats()
        ss.TrnEpcLog.SetNumRows(0)
        ss.TstEpcLog.SetNumRows(0)
        ss.TstQuizLog.SetNumRows(0)
        ss.NeedsNewRun = False

    def InitStats(ss):
        """
        InitStats initializes all the statistics, especially important for the
        cumulative epoch stats -- called at start of new run
        """

    def TrialStats(ss, accum):
        """
        TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
        accum is true.  Note that we're accumulating stats here on the Sim side so the
        core algorithm side remains as simple as possible, and doesn't need to worry about
        different time-scales over which stats could be accumulated etc.
        You can also aggregate directly from log data, as is done for testing stats
        """
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

    def OpenWts(ss):
        """
        OpenWts opens trained weights w/ rec=0.05
        """
        ss.Net.OpenWtsJSON("trained_rec05.wts.gz")

    def ConfigWts(ss, dt):
        dt.SetShape(go.Slice_int([14, 14, 12, 12]), go.nil, go.nil)
        dt.SetMetaData("grid-fill", "1")

    def TestTrial(ss, returnOnChg):
        """
        TestTrial runs one trial of testing -- always sequentially presented inputs
        """
        ss.InQuiz = False
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
        ss.LogTstTrl(ss.TstTrlLog, False)

    def TestAll(ss):
        """
        TestAll runs through the full set of testing items
        """
        err = ss.TestEnv.SetParas(go.Slice_string([ss.Words1, ss.Words2]))
        if err != 0:
            core.PromptDialog(
                go.nil,
                core.DlgOpts(Title="Words errors", Prompt=err.Error()),
                core.AddOk,
                core.NoCancel,
                go.nil,
                go.nil,
            )
            return
        ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
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

    def QuizTrial(ss, returnOnChg):
        """
        QuizTrial runs one trial of testing -- always sequentially presented inputs
        """
        ss.InQuiz = True
        ss.QuizEnv.Step()

        chg = ss.QuizEnv.CounterChg(env.Epoch)
        if chg:
            if ss.ViewOn and ss.TestUpdate.value > leabra.AlphaCycle:
                ss.UpdateView(False)
            ss.LogTstQuiz(ss.TstQuizLog)
            if returnOnChg:
                return

        ss.ApplyInputs(ss.QuizEnv)
        ss.AlphaCyc(False)
        ss.TrialStats(False)
        ss.LogTstTrl(ss.TstTrlLog, True)

    def QuizAll(ss):
        """
        QuizAll runs through the full set of testing items
        """
        ss.QuizEnv.Init(ss.TrainEnv.Run.Cur)
        while True:
            ss.QuizTrial(True)
            chg = ss.QuizEnv.CounterChg(env.Epoch)
            if chg or ss.StopNow:
                break

    def RunQuizAll(ss):
        """
        RunQuizAll runs through the full set of testing items, has stop running = false at end -- for gui
        """
        ss.StopNow = False
        ss.QuizAll()
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
        nt = ss.Net
        hid = leabra.Layer(nt.LayerByName("Hidden"))
        elat = leabra.Prjn(hid.RcvPrjns[1])
        elat.WtScale.Rel = ss.ExcitLateralScale
        elat.Learn.Learn = ss.ExcitLateralLearn
        ilat = leabra.Prjn(hid.RcvPrjns[2])
        ilat.WtScale.Abs = ss.InhibLateralScale

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

    def WtWords(ss):
        if ss.NetView.Data.PrjnLay != "Hidden":
            log.Println("WtWords: must select unit in Hidden layer in NetView")
            return go.nil
        ly = ss.Net.LayerByName(ss.NetView.Data.PrjnLay)
        slay = ss.Net.LayerByName("Input")
        pvals = go.Slice_float32()
        slay.SendPrjnValues(pvals, "Wt", ly, ss.NetView.Data.PrjnUnIndex, "")
        ww = go.Slice_string()
        for i, wrd in enumerate(ss.TrainEnv.Words):
            wv = pvals[i]
            if wv > ss.WtWordsThr:
                ww.append(wrd)
        views.SliceViewDialogNoStyle(
            ss.vp, ww, views.DlgOpts(Title="WtWords Result"), go.nil, go.nil
        )

    def LogTrnEpc(ss, dt):
        """
        LogTrnEpc adds data from current epoch to the TrnEpcLog table.
        computes epoch averages prior to logging.
        """
        row = dt.Rows
        dt.SetNumRows(row + 1)

        epc = ss.TrainEnv.Epoch.Prv
        nt = float(ss.TrainEnv.Trial.Max)

        # if ss.LastEpcTime.IsZero():
        #     ss.EpcPerTrlMSec = 0
        # else:
        #     iv = time.Now().Sub(ss.LastEpcTime)
        #     ss.EpcPerTrlMSec = float(iv) / (nt * float(time.Millisecond))
        # ss.LastEpcTime = time.Now()

        dt.SetCellFloat("Run", row, float(ss.TrainEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellFloat("PerTrlMSec", row, ss.EpcPerTrlMSec)

        for lnm in ss.LayStatNms:
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            dt.SetCellFloat(
                ly.Nm + " ActAvg", row, float(ly.Pools[0].ActAvg.ActPAvgEff)
            )

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
                etable.Column("PerTrlMSec", etensor.FLOAT64, go.nil, go.nil),
            ]
        )
        for lnm in ss.LayStatNms:
            sch.append(etable.Column(lnm + " ActAvg", etensor.FLOAT64, go.nil, go.nil))
        dt.SetFromSchema(sch, 0)

    def ConfigTrnEpcPlot(ss, plt, dt):
        plt.Params.Title = "Semantics Epoch Plot"
        plt.Params.XAxisCol = "Epoch"
        plt.SetTable(dt)

        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("PerTrlMSec", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

        for lnm in ss.LayStatNms:
            plt.SetColParams(
                lnm + " ActAvg", eplot.On, eplot.FixMin, 0, eplot.FixMax, 0.5
            )
        return plt

    def LogTstTrl(ss, dt, quiz):
        """
        LogTstTrl adds data from current trial to the TstTrlLog table.
        log always contains number of testing items
        """
        epc = ss.TrainEnv.Epoch.Prv

        trl = ss.TestEnv.Trial.Cur
        trlnm = ss.TestEnv.String()
        if quiz:
            trl = ss.QuizEnv.Trial.Cur
            trlnm = ss.QuizEnv.String()
        row = trl

        if dt.Rows <= row:
            dt.SetNumRows(row + 1)

        dt.SetCellFloat("Run", row, float(ss.TrainEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellFloat("Trial", row, float(trl))
        dt.SetCellString("TrialName", row, trlnm)

        for lnm in ss.LayStatNms:
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            vt = ss.ValuesTsr(lnm)
            ly.UnitValuesTensor(vt, "ActM")
            dt.SetCellTensor(lnm, row, vt)

        ss.TstTrlPlot.GoUpdate()

    def ConfigTstTrlLog(ss, dt):
        dt.SetMetaData("name", "TstTrlLog")
        dt.SetMetaData("desc", "Record of testing per input pattern")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        sch = etable.Schema(
            [
                etable.Column("Run", etensor.INT64, go.nil, go.nil),
                etable.Column("Epoch", etensor.INT64, go.nil, go.nil),
                etable.Column("Trial", etensor.INT64, go.nil, go.nil),
                etable.Column("TrialName", etensor.STRING, go.nil, go.nil),
            ]
        )
        for lnm in ss.LayStatNms:
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            sch.append(etable.Column(lnm, etensor.FLOAT64, ly.Shp.Shp, go.nil))
        dt.SetFromSchema(sch, 0)

    def ConfigTstTrlPlot(ss, plt, dt):
        plt.Params.Title = "Semantics Test Trial Plot"
        plt.Params.XAxisCol = "Trial"
        plt.SetTable(dt)

        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

        for lnm in ss.LayStatNms:
            plt.SetColParams(lnm, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        return plt

    def WordsShort(ss, wr):
        wf = wr.split()
        mx = min(len(wf), 2)
        ws = ""
        for i in range(mx):
            w = wf[i]
            if len(w) > 4:
                w = w[:4]
            ws += w
            if i < mx - 1:
                ws += "-"
        return ws

    def WordsLabel(ss):
        return ss.WordsShort(ss.Words1) + " v " + ss.WordsShort(ss.Words2)

    def LogTstEpc(ss, dt):
        row = dt.Rows
        dt.SetNumRows(row + 1)

        trl = ss.TstTrlLog
        epc = ss.TrainEnv.Epoch.Prv

        wr1 = etensor.Float64(trl.CellTensor("Hidden", 0))
        wr2 = etensor.Float64(trl.CellTensor("Hidden", 1))

        ss.TstWords = ss.WordsLabel()
        ss.TstWordsCorrel = metric.Correlation64(wr1.Values, wr2.Values)

        dt.SetCellFloat("Run", row, float(ss.TrainEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellString("Words", row, ss.TstWords)
        dt.SetCellFloat("TstWordsCorrel", row, ss.TstWordsCorrel)

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
                etable.Column("Words", etensor.STRING, go.nil, go.nil),
                etable.Column("TstWordsCorrel", etensor.FLOAT64, go.nil, go.nil),
            ]
        )
        dt.SetFromSchema(sch, 0)

    def ConfigTstEpcPlot(ss, plt, dt):
        plt.Params.Title = "Semantics Testing Epoch Plot"
        plt.Params.XAxisCol = "Words"
        plt.Params.Type = eplot.Bar
        plt.SetTable(dt)
        plt.Params.XAxisRot = 45
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Words", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams(
            "TstWordsCorrel", eplot.On, eplot.FixMin, -0.1, eplot.FixMax, 0.75
        )
        return plt

    def LogTstQuiz(ss, dt):
        trl = ss.TstTrlLog
        epc = ss.TrainEnv.Epoch.Prv  # ?

        nper = 4  # number of paras per quiz question: Q, A, B, C
        nt = trl.Rows
        nq = int(nt / nper)
        pctcor = 0.0
        srow = dt.Rows
        dt.SetNumRows(srow + nq + 1)
        for qi in range(nq):
            ri = nper * qi
            qv = etensor.Float64(trl.CellTensor("Hidden", ri))
            mxai = 0
            mxcor = 0.0
            row = srow + qi
            for ai in range(nper - 1):
                av = etensor.Float64(trl.CellTensor("Hidden", ri + ai + 1))
                cor = metric.Correlation64(qv.Values, av.Values)
                if cor > mxcor:
                    mxai = ai
                    mxcor = cor
                dt.SetCellTensorFloat1D("Correls", row, ai, cor)
            ans = go.Slice_string(["A", "B", "C"])[mxai]
            err = 1.0
            if mxai == 0:  # A
                pctcor += 1
                err = 0

            # note: this shows how to use agg methods to compute summary data from another
            # data table, instead of incrementing on the Sim
            dt.SetCellFloat("Run", row, float(ss.TrainEnv.Run.Cur))
            dt.SetCellFloat("Epoch", row, float(epc))
            dt.SetCellFloat("QNo", row, float(qi))
            dt.SetCellString("Resp", row, ans)
            dt.SetCellFloat("Err", row, err)
        pctcor /= float(nq)
        row = dt.Rows - 1
        dt.SetCellFloat("Run", row, float(ss.TrainEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellFloat("QNo", row, -1)
        dt.SetCellString("Resp", row, "Total")
        dt.SetCellFloat("Err", row, pctcor)

        ss.TstQuizPctCor = pctcor

        # note: essential to use Go version of update when called from another goroutine
        ss.TstQuizPlot.GoUpdate()

    def ConfigTstQuizLog(ss, dt):
        dt.SetMetaData("name", "TstQuizLog")
        dt.SetMetaData("desc", "Summary stats for testing trials")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        sch = etable.Schema(
            [
                etable.Column("Run", etensor.INT64, go.nil, go.nil),
                etable.Column("Epoch", etensor.INT64, go.nil, go.nil),
                etable.Column("QNo", etensor.INT64, go.nil, go.nil),
                etable.Column("Resp", etensor.STRING, go.nil, go.nil),
                etable.Column("Err", etensor.FLOAT64, go.nil, go.nil),
                etable.Column("Correls", etensor.FLOAT64, go.Slice_int([3]), go.nil),
            ]
        )
        dt.SetFromSchema(sch, 0)

    def ConfigTstQuizPlot(ss, plt, dt):
        plt.Params.Title = "Semantics Testing Quiz Plot"
        plt.Params.XAxisCol = "QNo"
        plt.Params.Type = eplot.Bar
        plt.SetTable(dt)
        # plt.Params.XAxisRot = 45
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("QNo", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Resp", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Err", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        pl = plt.SetColParams("Correls", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        pl.TensorIndex = -1
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
        nlast = 10
        if nlast > epcix.Len() - 1:
            nlast = epcix.Len() - 1
        epcix.Indexes = epcix.Indexes[epcix.Len() - nlast - 1 :]

        # params := ss.Params.Name
        params = "params"

        dt.SetCellFloat("Run", row, float(run))
        dt.SetCellString("Params", row, params)

        # runix := etable.NewIndexView(dt)
        # spl := split.GroupBy(runix, []string{"Params"})
        # split.Desc(spl, "FirstZero")
        # split.Desc(spl, "PctCor")
        # ss.RunStats = spl.AggsToTable(etable.AddAggName)

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
            ]
        )
        dt.SetFromSchema(sch, 0)

    def ConfigRunPlot(ss, plt, dt):
        plt.Params.Title = "Semantics Run Plot"
        plt.Params.XAxisCol = "Run"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        return plt

    def ConfigNetView(ss, nv):
        nv.ViewDefaults()
        cam = nv.Scene().Camera
        cam.Pose.Pos.Set(0.0, 1.733, 2.3)
        cam.LookAt(math32.Vector3(0, 0, 0), math32.Vector3(0, 1, 0))

    # cam.Pose.Quat.SetFromAxisAngle(math32.Vector3{-1, 0, 0}, 0.4077744)

    def ConfigGui(ss):
        """
        ConfigGui configures the GoGi gui interface for this simulation,
        """
        width = 1600
        height = 1200

        core.SetAppName("sem")
        core.SetAppAbout(
            'sem is trained using Hebbian learning on paragraphs from an early draft of the *Computational Explorations..* textbook, allowing it to learn about the overall statistics of when different words co-occur with other words, and thereby learning a surprisingly capable (though clearly imperfect) level of semantic knowlege about the topics covered in the textbook.  This replicates the key results from the Latent Semantic Analysis research by Landauer and Dumais (1997). See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch9/sem/README.md">README.md on GitHub</a>.</p>'
        )

        win = core.NewMainWindow("sem", "Sem Semantic Hebbian Learning", width, height)
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
        tv.AddTab(plt, "TstQuizPlot")
        ss.TstQuizPlot = ss.ConfigTstQuizPlot(plt, ss.TstQuizLog)

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

        tbar.AddSeparator("spec")

        tbar.AddAction(
            core.ActOpts(
                Label="Open Weights",
                Icon="update",
                Tooltip="Open weights trained on first phase of training (excluding 'novel' objects)",
                UpdateFunc=UpdateFuncNotRunning,
            ),
            recv,
            OpenWtsCB,
        )

        tbar.AddAction(
            core.ActOpts(
                Label="Wt Words",
                Icon="search",
                Tooltip="get words for currently-selected hidden-layer unit in netview.",
                UpdateFunc=UpdateFuncNotRunning,
            ),
            recv,
            WtWordsCB,
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
                Label="Quiz All",
                Icon="fast-fwd",
                Tooltip="all of the quiz testing trials.",
                UpdateFunc=UpdateFuncNotRunning,
            ),
            recv,
            QuizAllCB,
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
