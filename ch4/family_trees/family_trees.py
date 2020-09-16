#!/usr/local/bin/pyleabra -i

# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# use:
# just type file name to run, or:
# pyleabra -i <file>.py 

# family_trees shows how learning can recode inputs that have
# no similarity structure into a hidden layer that captures
# the *functional* similarity structure of the items.

from leabra import go, leabra, emer, relpos, eplot, env, agg, patgen, prjn, etable, efile, split, etensor, params, netview, rand, erand, gi, giv, pygiv, pyparams, mat32, metric, simat, pca, clust

import importlib as il
import io, sys, getopt
from datetime import datetime, timezone
from enum import Enum

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

def GenTestTrialCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.GenTestTrial(False)
        TheSim.IsRunning = False
        TheSim.UpdateClassView()
        TheSim.vp.SetNeedsFullRender()

def AllTestTrialCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.AllTestTrial(False)
        TheSim.IsRunning = False
        TheSim.UpdateClassView()
        TheSim.vp.SetNeedsFullRender()

def GenTestAllCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.ToolBar.UpdateActions()
        TheSim.RunGenTestAll()

def AllTestAllCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.ToolBar.UpdateActions()
        TheSim.RunAllTestAll()

def RepsAnalysisCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.RepsAnalysis()

def ResetRunLogCB(recv, send, sig, data):
    TheSim.RunLog.SetNumRows(0)
    TheSim.RunPlot.Update()

def NewRndSeedCB(recv, send, sig, data):
    TheSim.NewRndSeed()

def ReadmeCB(recv, send, sig, data):
    gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch4/family_trees/README.md")

def UpdtFuncNotRunning(act):
    act.SetActiveStateUpdt(not TheSim.IsRunning)
    
def UpdtFuncRunning(act):
    act.SetActiveStateUpdt(TheSim.IsRunning)

FilterRowMap = {}
    
def FilterTestEnv(et, row):
    return not row in FilterRowMap

    
############################################
# Enums -- note: must start at 0 for GUI

class LearnType(Enum):
    PureHebb = 0
    PureError = 1
    HebbError = 2

    
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
        self.Net = leabra.Network()
        self.SetTags("Net", 'view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"')
        self.Learn = LearnType.HebbError
        self.SetTags("Learn", 'desc:"select which type of learning to use"')
        self.Pats = etable.Table()
        self.SetTags("Pats", 'view:"no-inline" desc:"training patterns"')
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
        self.Params = params.Sets()
        self.SetTags("Params", 'view:"no-inline" desc:"full collection of param sets"')
        self.ParamSet = str()
        self.SetTags("ParamSet", 'desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set"')
        self.Tag = str()
        self.SetTags("Tag", 'desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"')
        self.MaxRuns = int(10)
        self.SetTags("MaxRuns", 'desc:"maximum number of model runs to perform"')
        self.MaxEpcs = int(500)
        self.SetTags("MaxEpcs", 'desc:"maximum number of epochs to run per model run"')
        self.NZeroStop = int(1)
        self.SetTags("NZeroStop", 'desc:"if a positive number, training will stop after this many epochs with zero SSE"')
        self.TrainEnv = env.FixedTable()
        self.SetTags("TrainEnv", 'desc:"Training environment -- contains everything about iterating over input / output patterns over training"')
        self.GenTestEnv = env.FixedTable()
        self.SetTags("GenTestEnv", 'desc:"Generalization Testing environment (4 held-out items not trained -- not enough training data to really drive generalization here) -- manages iterating over testing"')
        self.AllTestEnv = env.FixedTable()
        self.SetTags("AllTestEnv", 'desc:"Test all items -- manages iterating over testing"')
        self.Time = leabra.Time()
        self.SetTags("Time", 'desc:"leabra timing parameters and state"')
        self.ViewOn = True
        self.SetTags("ViewOn", 'desc:"whether to update the network view while running"')
        self.TrainUpdt = leabra.TimeScales.AlphaCycle
        self.SetTags("TrainUpdt", 'desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"')
        self.TestUpdt = leabra.TimeScales.Cycle
        self.SetTags("TestUpdt", 'desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"')
        self.TestInterval = int(5)
        self.SetTags("TestInterval", 'desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"')
        self.TstRecLays = go.Slice_string(["Hidden", "AgentCode"])
        self.SetTags("TstRecLays", 'desc:"names of layers to record activations etc of during testing"')
        self.HiddenRel = Reps()
        self.SetTags("HiddenRel", 'view:"inline" desc:"representational analysis of Hidden layer, sorted by relationship"')
        self.HiddenAgent = Reps()
        self.SetTags("HiddenAgent", 'view:"inline" desc:"representational analysis of Hidden layer, sorted by agent"')
        self.AgentAgent = Reps()
        self.SetTags("AgentAgent", 'view:"inline" desc:"representational analysis of AgentCode layer, sorted by agent"')

        # statistics: note use float64 as that is best for etable.Table
        self.TrlErr = float(0)
        self.SetTags("TrlErr", 'inactive:"+" desc:"1 if trial was error, 0 if correct -- based on SSE = 0 (subject to .5 unit-wise tolerance)"')
        self.TrlSSE = float(0)
        self.SetTags("TrlSSE", 'inactive:"+" desc:"current trial\'s sum squared error"')
        self.TrlAvgSSE = float(0)
        self.SetTags("TrlAvgSSE", 'inactive:"+" desc:"current trial\'s average sum squared error"')
        self.TrlCosDiff = float(0)
        self.SetTags("TrlCosDiff", 'inactive:"+" desc:"current trial\'s cosine difference"')
        self.EpcSSE = float(0)
        self.SetTags("EpcSSE", 'inactive:"+" desc:"last epoch\'s total sum squared error"')
        self.EpcAvgSSE = float(0)
        self.SetTags("EpcAvgSSE", 'inactive:"+" desc:"last epoch\'s average sum squared error (average over trials, and over units within layer)"')
        self.EpcPctErr = float(0)
        self.SetTags("EpcPctErr", 'inactive:"+" desc:"last epoch\'s percent of trials that had SSE > 0 (subject to .5 unit-wise tolerance)"')
        self.EpcPctCor = float(0)
        self.SetTags("EpcPctCor", 'inactive:"+" desc:"last epoch\'s percent of trials that had SSE == 0 (subject to .5 unit-wise tolerance)"')
        self.EpcCosDiff = float(0)
        self.SetTags("EpcCosDiff", 'inactive:"+" desc:"last epoch\'s average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"')
        self.EpcPerTrlMSec = float(0)
        self.SetTags("EpcPerTrlMSec", 'inactive:"+" desc:"how long did the epoch take per trial in wall-clock milliseconds"')
        self.FirstZero = int(-1)
        self.SetTags("FirstZero", 'inactive:"+" desc:"epoch at when SSE first went to zero"')
        self.NZero = int(0)
        self.SetTags("NZero", 'inactive:"+" desc:"number of epochs in a row with zero SSE"')

        # internal state - view:"-"
        self.SumErr = float(0)
        self.SetTags("SumErr", 'view:"-" inactive:"+" desc:"sum to increment as we go through epoch"')
        self.SumSSE = float(0)
        self.SetTags("SumSSE", 'view:"-" inactive:"+" desc:"sum to increment as we go through epoch"')
        self.SumAvgSSE = float(0)
        self.SetTags("SumAvgSSE", 'view:"-" inactive:"+" desc:"sum to increment as we go through epoch"')
        self.SumCosDiff = float(0)
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
        self.RndSeed = int(1)
        self.SetTags("RndSeed", 'view:"-" desc:"the current random seed"')
        self.LastEpcTime = int(0)
        self.SetTags("LastEpcTime", 'view:"-" desc:"timer for last epoch"')
        self.vp  = 0 
        self.SetTags("vp", 'view:"-" desc:"viewport"')
        
    def InitParams(ss):
        """
        Sets the default set of parameters -- Base is always applied, and others can be optionally
        selected to apply on top of that
        """
        ss.Params.OpenJSON("family_trees.params")

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

        # note: code below pulls out these specific testing patterns into test env
        # and removes from training
        tsts = go.Slice_string(["James.Wife.Vicky", "Lucia.Fath.Robert", "Angela.Bro.Marco", "Christi.Daug.Jenn"])

        trix = etable.NewIdxView(ss.Pats)
        tsix = etable.NewIdxView(ss.Pats)
        tsix.Idxs = tsix.Idxs[:0]
        
        FilterRowMap = { } # global dict

        for ts in tsts:
            ix = ss.Pats.RowsByString("Name", ts, etable.Equals, etable.UseCase)
            tsix.Idxs.append(ix[0])
            FilterRowMap[ix[0]] = True
            
        trix.Filter(FilterTestEnv)

        ss.TrainEnv.Nm = "TrainEnv"
        ss.TrainEnv.Dsc = "training params and state"
        ss.TrainEnv.Table = trix
        ss.TrainEnv.Validate()
        ss.TrainEnv.Run.Max = ss.MaxRuns # note: we are not setting epoch max -- do that manually

        ss.GenTestEnv.Nm = "GenTestEnv"
        ss.GenTestEnv.Dsc = "hold-out testing params and state"
        ss.GenTestEnv.Table = tsix
        ss.GenTestEnv.Sequential = True
        ss.GenTestEnv.Validate()

        ss.AllTestEnv.Nm = "AllTestEnv"
        ss.AllTestEnv.Dsc = "test all params and state"
        ss.AllTestEnv.Table = etable.NewIdxView(ss.Pats)
        ss.AllTestEnv.Sequential = True
        ss.AllTestEnv.Validate()

        ss.TrainEnv.Init(0)
        ss.GenTestEnv.Init(0)
        ss.AllTestEnv.Init(0)

    def ConfigNet(ss, net):
        net.InitName(net, "FamTrees")
        ag = net.AddLayer2D("Agent", 4, 6, emer.Input)
        rl = net.AddLayer2D("Relation", 2, 6, emer.Input)
        agcd = net.AddLayer2D("AgentCode", 7, 7, emer.Hidden)
        rlcd = net.AddLayer2D("RelationCode", 7, 7, emer.Hidden)
        hid = net.AddLayer2D("Hidden", 7, 7, emer.Hidden)
        ptcd = net.AddLayer2D("PatientCode", 7, 7, emer.Hidden)
        pt = net.AddLayer2D("Patient", 4, 6, emer.Target)

        agcd.SetClass("Code")
        rlcd.SetClass("Code")
        ptcd.SetClass("Code")
        ag.SetClass("Person")
        pt.SetClass("Person")
        rl.SetClass("Relation")

        # much faster without threads on!
        # agcd.SetThread(1)
        # rlcd.SetThread(1)
        # ptcd.SetThread(1)

        full = prjn.NewFull()
        net.ConnectLayers(ag, agcd, full, emer.Forward)
        net.ConnectLayers(rl, rlcd, full, emer.Forward)
        net.BidirConnectLayersPy(agcd, hid, full)
        net.BidirConnectLayersPy(rlcd, hid, full)
        net.BidirConnectLayersPy(hid, ptcd, full)
        net.BidirConnectLayersPy(ptcd, pt, full)

        rl.SetRelPos(relpos.Rel(Rel= relpos.RightOf, Other= "Agent", YAlign= relpos.Front, Space= 2))
        pt.SetRelPos(relpos.Rel(Rel= relpos.RightOf, Other= "Relation", YAlign= relpos.Front, Space= 2))
        agcd.SetRelPos(relpos.Rel(Rel= relpos.Above, Other= "Agent", YAlign= relpos.Front, XAlign= relpos.Left))
        rlcd.SetRelPos(relpos.Rel(Rel= relpos.RightOf, Other= "AgentCode", YAlign= relpos.Front, Space= 1))
        ptcd.SetRelPos(relpos.Rel(Rel= relpos.RightOf, Other= "RelationCode", YAlign= relpos.Front, Space= 1))
        hid.SetRelPos(relpos.Rel(Rel= relpos.Above, Other= "RelationCode", YAlign= relpos.Front, XAlign= relpos.Middle))

        net.Defaults()
        ss.SetParams("Network", False) # only set Network params
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
        ss.RndSeed = datetime.now(timezone.utc)

    def Counters(ss, train):
        """
        Counters returns a string of the current counter state
        use tabs to achieve a reasonable formatting overall
        and add a few tabs at the end to allow for expansion..
        """
        if train:
            return "Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t" % (ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Trial.Cur, ss.Time.Cycle, ss.TrainEnv.TrialName.Cur)
        else:
            return "Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t" % (ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.GenTestEnv.Trial.Cur, ss.Time.Cycle, ss.GenTestEnv.TrialName.Cur)

    def UpdateView(ss, train):
        if ss.NetView != 0 and ss.NetView.IsVisible():
            ss.NetView.Record(ss.Counters(train))
            ss.NetView.GoUpdate()

    def AlphaCyc(ss, train):
        """
        AlphaCyc runs one alpha-cycle (100 msec, 4 quarters) of processing.
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

        # update prior weight changes at start, so any DWt values remain visible at end
        # you might want to do this less frequently to achieve a mini-batch update
        # in which case, move it out to the TrainTrial method where the relevant
        # counters are being dealt with.
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
            ss.Net.DWt()
        if ss.ViewOn and viewUpdt == leabra.AlphaCycle:
            ss.UpdateView(train)

    def ApplyInputs(ss, en):
        """
        ApplyInputs applies input patterns from given envirbonment.
        It is good practice to have this be a separate method with appropriate
        args so that it can be used for various different contexts
        (training, testing, etc).
        """
        ss.Net.InitExt()

        lays = go.Slice_string(["Agent", "Relation", "Patient"])
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
            if ss.ViewOn and ss.TrainUpdt.value > leabra.AlphaCycle:
                ss.UpdateView(True)
            if ss.TestInterval > 0 and epc%ss.TestInterval == 0: # note: epc is *next* so won't trigger first time
                ss.GenTestAll()
            if epc >= ss.MaxEpcs or (ss.NZeroStop > 0 and ss.NZero >= ss.NZeroStop):
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
        ss.GenTestEnv.Init(run)
        ss.AllTestEnv.Init(run)
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
        out = leabra.Layer(ss.Net.LayerByName("Patient"))
        ss.TrlCosDiff = float(out.CosDiff.Cos)
        ss.TrlSSE = out.SSE(0.5)
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

    def GenTestTrial(ss, returnOnChg):
        """
        GenTestTrial runs one trial of testing -- always sequentially presented inputs
        """
        ss.GenTestEnv.Step()

        chg = env.CounterChg(ss.GenTestEnv, env.Epoch)
        if chg:
            if ss.ViewOn and ss.TestUpdt.value > leabra.AlphaCycle:
                ss.UpdateView(False)
            ss.LogTstEpc(ss.TstEpcLog)
            if returnOnChg:
                return

        ss.ApplyInputs(ss.GenTestEnv)
        ss.AlphaCyc(False)
        ss.TrialStats(False)
        ss.LogTstTrl(ss.TstTrlLog, ss.GenTestEnv.Trial.Cur, ss.GenTestEnv.TrialName.Cur)

    def GenTestAll(ss):
        """
        GenTestAll runs through the full set of testing items
        """
        ss.GenTestEnv.Init(ss.TrainEnv.Run.Cur)
        while True:
            ss.GenTestTrial(True)
            chg = env.CounterChg(ss.GenTestEnv, env.Epoch)
            if chg or ss.StopNow:
                break

    def RunGenTestAll(ss):
        """
        RunGenTestAll runs through the full set of testing items, has stop running = false at end -- for gui
        """
        ss.StopNow = False
        ss.GenTestAll()
        ss.Stopped()

    def AllTestTrial(ss, returnOnChg):
        """
        AllTestTrial runs one trial of testing -- always sequentially presented inputs
        """
        ss.AllTestEnv.Step()

        chg = env.CounterChg(ss.AllTestEnv, env.Epoch)
        if chg:
            if ss.ViewOn and ss.TestUpdt.value > leabra.AlphaCycle:
                ss.UpdateView(False)
            ss.LogTstEpc(ss.TstEpcLog)
            if returnOnChg:
                return

        ss.ApplyInputs(ss.AllTestEnv)
        ss.AlphaCyc(False)
        ss.TrialStats(False)
        ss.LogTstTrl(ss.TstTrlLog, ss.AllTestEnv.Trial.Cur, ss.AllTestEnv.TrialName.Cur)

    def AllTestAll(ss):
        """
        AllTestAll runs through the full set of testing items
        """
        ss.AllTestEnv.Init(ss.TrainEnv.Run.Cur)
        while True:
            ss.AllTestTrial(True)
            chg = env.CounterChg(ss.AllTestEnv, env.Epoch)
            if chg or ss.StopNow:
                break

    def RunAllTestAll(ss):
        """
        RunAllTestAll runs through the full set of testing items, has stop running = false at end -- for gui
        """
        ss.StopNow = False
        ss.AllTestAll()
        ss.Stopped()

    def RepsAnalysis(ss):
        """
        RepsAnalysis does a full test and then runs tests of representations
        """
        ss.RunAllTestAll()

        nmtsr = etensor.String(ss.TstTrlLog.ColByName("TrialName"))
        names = list(nmtsr.Values)
        nv = len(names)

        for i in range(nv):
            nm = names[i]
            nnm = nm[nm.index(".")+1:]
            nnm = nnm[:nnm.index(".")]
            nmtsr.Values[i] = nnm

        rels = etable.NewIdxView(ss.TstTrlLog)
        rels.SortCol(ss.TstTrlLog.ColIdx("TrialName"), True)
        ss.HiddenRel.SimMat.TableColStd(rels, "Hidden", "TrialName", True, metric.Correlation)
        ss.HiddenRel.PCA.TableColStd(rels, "Hidden", metric.Covariance)
        ss.HiddenRel.PCA.ProjectColToTable(ss.HiddenRel.PCAPrjn, rels, "Hidden", "TrialName", go.Slice_int([0, 1]))
        ss.ConfigPCAPlot(ss.HiddenRel.PCAPlot, ss.HiddenRel.PCAPrjn, "Hidden Rel")
        ss.ClustPlot(ss.HiddenRel.ClustPlot, rels, "Hidden")

        for i in range(nv):
            nm = names[i]
            nnm = nm[:nm.index(".")]
            nmtsr.Values[i] = nnm
        ags = etable.NewIdxView(ss.TstTrlLog)
        ags.SortCol(ss.TstTrlLog.ColIdx("TrialName"), True)
        ss.HiddenAgent.SimMat.TableColStd(ags, "Hidden", "TrialName", True, metric.Correlation)
        ss.HiddenAgent.PCA.TableColStd(ags, "Hidden", metric.Covariance)
        ss.HiddenAgent.PCA.ProjectColToTable(ss.HiddenAgent.PCAPrjn, ags, "Hidden", "TrialName", go.Slice_int([2, 3]))
        ss.ConfigPCAPlot(ss.HiddenAgent.PCAPlot, ss.HiddenAgent.PCAPrjn, "Hidden Agent")
        ss.HiddenAgent.PCAPlot.SetColParams("Prjn3", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
        ss.HiddenAgent.PCAPlot.Params.XAxisCol = "Prjn2"
        ss.ClustPlot(ss.HiddenAgent.ClustPlot, ags, "Hidden")

        ss.AgentAgent.SimMat.TableColStd(ags, "AgentCode", "TrialName", True, metric.Correlation)
        ss.AgentAgent.PCA.TableColStd(ags, "AgentCode", metric.Covariance)
        ss.AgentAgent.PCA.ProjectColToTable(ss.AgentAgent.PCAPrjn, ags, "AgentCode", "TrialName", go.Slice_int([0, 1]))
        ss.ConfigPCAPlot(ss.AgentAgent.PCAPlot, ss.AgentAgent.PCAPrjn, "AgentCode")
        ss.ClustPlot(ss.AgentAgent.ClustPlot, ags, "AgentCode")

        nmtsr.Values.copy(names)
        ss.Stopped()

    def ConfigPCAPlot(ss, plt, dt, nm):
        plt.Params.Title = "Family Trees PCA Plot: " + nm
        plt.Params.XAxisCol = "Prjn0"
        plt.SetTable(dt)
        plt.Params.Lines = False
        plt.Params.Points = True
        # order of params: on, fixMin, min, fixMax, max
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
        if ss.Learn == LearnType.PureHebb:
            ss.SetParamsSet("PureHebb", sheet, setMsg)
        elif ss.Learn == LearnType.PureError:
            ss.SetParamsSet("PureError", sheet, setMsg)
        elif ss.Learn == LearnType.HebbError:
            ss.SetParamsSet("HebbError", sheet, setMsg)

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

    def OpenPats(ss):
        ss.Pats.SetMetaData("name", "Family Trees")
        ss.Pats.SetMetaData("desc", "Family Trees Training patterns")
        ss.Pats.OpenCSV("family_trees.tsv", etable.Tab)

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
        return ss.Net.Nm + "_" + ss.RunName() + "_" + lognm + ".csv"

    def LogTrnEpc(ss, dt):
        """
        LogTrnEpc adds data from current epoch to the TrnEpcLog table.
        computes epoch averages prior to logging.
        """
        row = dt.Rows
        dt.SetNumRows(row + 1)

        epc = ss.TrainEnv.Epoch.Prv
        nt = float(ss.TrainEnv.Table.Len())

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
        plt.Params.Title = "Family Trees Epoch Plot"
        plt.Params.XAxisCol = "Epoch"
        plt.SetTable(dt)

        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
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

        row = trl
        if dt.Rows <= row:
            dt.SetNumRows(row + 1)

        dt.SetCellFloat("Run", row, float(ss.TrainEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellFloat("Trial", row, float(trl))
        dt.SetCellString("TrialName", row, trlnm)
        dt.SetCellFloat("Err", row, ss.TrlErr)
        dt.SetCellFloat("SSE", row, ss.TrlSSE)
        dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
        dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

        for lnm in ss.TstRecLays:
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

        nt = ss.GenTestEnv.Table.Len() # number in view
        sch = etable.Schema(
            [etable.Column("Run", etensor.INT64, go.nil, go.nil),
            etable.Column("Epoch", etensor.INT64, go.nil, go.nil),
            etable.Column("Trial", etensor.INT64, go.nil, go.nil),
            etable.Column("TrialName", etensor.STRING, go.nil, go.nil),
            etable.Column("Err", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("SSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("AvgSSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("CosDiff", etensor.FLOAT64, go.nil, go.nil)]
        )
        for lnm in ss.TstRecLays:
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            sch.append(etable.Column(lnm, etensor.FLOAT64, ly.Shp.Shp, go.nil))
        dt.SetFromSchema(sch, nt)

    def ConfigTstTrlPlot(ss, plt, dt):
        plt.Params.Title = "Family Trees Test Trial Plot"
        plt.Params.XAxisCol = "Trial"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Err", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("SSE", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0) # default plot
        plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

        for lnm in ss.TstRecLays:
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

        # note: essential to use Go version of update when called from another goroutine
        if ss.TstEpcPlot != 0:
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
        dt.SetFromSchema(sch, 0)

    def ConfigTstEpcPlot(ss, plt, dt):
        plt.Params.Title = "Family Trees Testing Epoch Plot"
        plt.Params.XAxisCol = "Epoch"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("SSE", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0) # default plot
        plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("PctErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        return plt

    def LogRun(ss, dt):
        """
        LogRun adds data from current run to the RunLog table.
        """
        run = ss.TrainEnv.Run.Cur # this is NOT triggered by increment yet -- use Cur
        row = dt.Rows
        dt.SetNumRows(row + 1)

        epclog = ss.TrnEpcLog
        epcix = etable.NewIdxView(epclog)
        # compute mean over last N epochs for run level
        nlast = 5
        if nlast > epcix.Len()-1:
            nlast = epcix.Len() - 1
        epcix.Idxs = epcix.Idxs[epcix.Len()-nlast:]

        params = ss.Learn.name

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
        if ss.RunPlot != 0:
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
        plt.Params.Title = "Family Trees Run Plot"
        plt.Params.XAxisCol = "Run"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("FirstZero", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0) # default plot
        plt.SetColParams("SSE", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("PctErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        return plt

    def ConfigGui(ss):
        """
        ConfigGui configures the GoGi gui interface for this simulation,
        """
        width = 1600
        height = 1200

        gi.SetAppName("family_trees")
        gi.SetAppAbout('shows how learning can recode inputs that have no similarity structure into a hidden layer that captures the *functional* similarity structure of the items. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch4/family_trees/README.md">README.md on GitHub</a>.</p>')

        win = gi.NewMainWindow("family_trees", "Family Trees", width, height)
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
        
        tbar.AddAction(gi.ActOpts(Label="Gen Test Trial", Icon="step-fwd", Tooltip="Runs the next testing trial.", UpdateFunc=UpdtFuncNotRunning), recv, GenTestTrialCB)
        
        tbar.AddAction(gi.ActOpts(Label="Gen Test All", Icon="fast-fwd", Tooltip="Tests all of the testing trials.", UpdateFunc=UpdtFuncNotRunning), recv, GenTestAllCB)

        tbar.AddAction(gi.ActOpts(Label="All Test Trial", Icon="step-fwd", Tooltip="Runs the next testing trial.", UpdateFunc=UpdtFuncNotRunning), recv, AllTestTrialCB)
        
        tbar.AddAction(gi.ActOpts(Label="All Test All", Icon="fast-fwd", Tooltip="Tests all of the testing trials.", UpdateFunc=UpdtFuncNotRunning), recv, AllTestAllCB)

        tbar.AddAction(gi.ActOpts(Label= "Reps Analysis", Icon= "fast-fwd", Tooltip= "Does an All Test All and analyzes the resulting Hidden and AgentCode activations.", UpdateFunc=UpdtFuncNotRunning), recv, RepsAnalysisCB)
        
        tbar.AddAction(gi.ActOpts(Label="Reset RunLog", Icon="reset", Tooltip="Resets the accumulated log of all Runs, which are tagged with the ParamSet used"), recv, ResetRunLogCB)

        tbar.AddSeparator("log")
        
        tbar.AddAction(gi.ActOpts(Label= "Reset RunLog", Icon= "update", Tooltip= "Reset the accumulated log of all Runs, which are tagged with the ParamSet used"), recv, ResetRunLogCB)

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


