#!/usr/local/bin/pyleabra -i

# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# use:
# just type file name to run, or:
# pyleabra -i <file>.py 

# objrec explores how a hierarchy of areas in the ventral stream of visual
# processing (up to inferotemporal (IT) cortex) can produce robust object
# recognition that is invariant to changes in position, size, etc of retinal
# input images.

from leabra import go, leabra, emer, relpos, eplot, env, agg, patgen, prjn, etable, efile, split, etensor, etview, params, netview, rand, erand, gi, giv, pygiv, pyparams, mat32, actrf

import importlib as il
import io, sys, getopt
from datetime import datetime, timezone

from led_env import LEDEnv

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

def ResetRunLogCB(recv, send, sig, data):
    TheSim.RunLog.SetNumRows(0)
    TheSim.RunPlot.Update()

def NewRndSeedCB(recv, send, sig, data):
    TheSim.NewRndSeed()

def ReadmeCB(recv, send, sig, data):
    gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch6/objrec/README.md")

def UpdtFuncNotRunning(act):
    act.SetActiveStateUpdt(not TheSim.IsRunning)
    
def UpdtFuncRunning(act):
    act.SetActiveStateUpdt(TheSim.IsRunning)

def OpenTrainedWtsCB(recv, send, sig, data):    
    TheSim.OpenTrainedWts()

def TrainNovelCB(recv, send, sig, data):    
    TheSim.TrainNovel()
    
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
        self.TrnEpcLog = etable.Table()
        self.SetTags("TrnEpcLog", 'view:"no-inline" desc:"training epoch-level log data"')
        self.TstEpcLog = etable.Table()
        self.SetTags("TstEpcLog", 'view:"no-inline" desc:"testing epoch-level log data"')
        self.TstTrlLog = etable.Table()
        self.SetTags("TstTrlLog", 'view:"no-inline" desc:"testing trial-level log data"')
        self.ActRFs = actrf.RFs()
        self.SetTags("ActRFs", 'view:"no-inline" desc:"activation-based receptive fields"')
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
        self.V1V4Prjn = prjn.NewPoolTile()
        self.SetTags("V1V4Prjn", 'view:"projection from V1 to V4 which is tiled 4x4 skip 2 with topo scale values"')
        self.V1V4Prjn.Size.Set(4, 4)
        self.V1V4Prjn.Skip.Set(2, 2)
        self.V1V4Prjn.Start.Set(-1, -1)
        self.V1V4Prjn.TopoRange.Min = 0.8
        self.MaxRuns = int(1)
        self.SetTags("MaxRuns", 'desc:"maximum number of model runs to perform"')
        self.MaxEpcs = int(50)
        self.SetTags("MaxEpcs", 'desc:"maximum number of epochs to run per model run"')
        self.MaxTrls = int(100)
        self.SetTags("MaxTrls", 'desc:"maximum number of training trials per epoch"')
        self.NZeroStop = int(-1)
        self.SetTags("NZeroStop", 'desc:"if a positive number, training will stop after this many epochs with zero SSE"')
        self.TrainEnv = LEDEnv()
        self.SetTags("TrainEnv", 'desc:"Training environment -- LED training"')
        self.PNovel = float(0)
        self.SetTags("PNovel", 'desc:"proportion of novel training items to use -- set this to 0.5 after initial training"')
        self.NovelTrainEnv = LEDEnv()
        self.SetTags("NovelTrainEnv", 'desc:"Novel items training environment -- LED training"')
        self.TestEnv = LEDEnv()
        self.SetTags("TestEnv", 'desc:"Testing environment -- LED testing"')
        self.Time = leabra.Time()
        self.SetTags("Time", 'desc:"leabra timing parameters and state"')
        self.ViewOn = True
        self.SetTags("ViewOn", 'desc:"whether to update the network view while running"')
        self.TrainUpdt = leabra.TimeScales.Quarter
        self.SetTags("TrainUpdt", 'desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"')
        self.TestUpdt = leabra.TimeScales.Quarter
        self.SetTags("TestUpdt", 'desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"')
        self.LayStatNms = go.Slice_string(["V1", "Output"])
        self.SetTags("LayStatNms", 'desc:"names of layers to collect more detailed stats on (avg act, etc)"')
        self.ActRFNms = go.Slice_string(["V4:Image", "V4:Output", "IT:Image", "IT:Output"])
        self.SetTags("ActRFNms", 'desc:"names of layers to compute activation rfields on"')

        # statistics: note use float64 as that is best for etable.Table
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
        self.EpcPerTrlMSec = float()
        self.SetTags("EpcPerTrlMSec", 'inactive:"+" desc:"how long did the epoch take per trial in wall-clock milliseconds"')
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
        self.CurImgGrid = 0
        self.SetTags("CurImgGrid", 'view:"-" desc:"the current image grid view"')
        self.ActRFGrids = {}
        self.SetTags("ActRFGrids", 'view:"-" desc:"the act rf grid views"')
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
        self.LastEpcTime = 0
        self.SetTags("LastEpcTime", 'view:"-" desc:"timer for last epoch"')
        self.vp  = 0 
        self.SetTags("vp", 'view:"-" desc:"viewport"')

    def InitParams(ss):
        """
        Sets the default set of parameters -- Base is always applied, and others can be optionally
        selected to apply on top of that
        """
        ss.Params.OpenJSON("objrec.params")

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
        if ss.MaxRuns == 0: # allow user override
            ss.MaxRuns = 1
        if ss.MaxEpcs == 0: # allow user override
            ss.MaxEpcs = 50
            ss.NZeroStop = -1
        if ss.MaxTrls == 0: # allow user override
            ss.MaxTrls = 100

        ss.TrainEnv.Nm = "TrainEnv"
        ss.TrainEnv.Dsc = "training params and state"
        ss.TrainEnv.Defaults()
        ss.TrainEnv.MinLED = 0
        ss.TrainEnv.MaxLED = 17 # exclude last 2 by default
        ss.TrainEnv.Validate()
        ss.TrainEnv.Run.Max = ss.MaxRuns # note: we are not setting epoch max -- do that manually
        ss.TrainEnv.Trial.Max = ss.MaxTrls

        ss.NovelTrainEnv.Nm = "NovelTrainEnv"
        ss.NovelTrainEnv.Dsc = "novel items training params and state"
        ss.NovelTrainEnv.Defaults()
        ss.NovelTrainEnv.MinLED = 18
        ss.NovelTrainEnv.MaxLED = 19 # only last 2 items
        ss.NovelTrainEnv.Validate()
        ss.NovelTrainEnv.Run.Max = ss.MaxRuns # note: we are not setting epoch max -- do that manually
        ss.NovelTrainEnv.Trial.Max = ss.MaxTrls
        ss.NovelTrainEnv.XFormRand.TransX.Set(-0.125, 0.125)
        ss.NovelTrainEnv.XFormRand.TransY.Set(-0.125, 0.125)
        ss.NovelTrainEnv.XFormRand.Scale.Set(0.775, 0.925) # 1/2 around midpoint
        ss.NovelTrainEnv.XFormRand.Rot.Set(-2, 2)

        ss.TestEnv.Nm = "TestEnv"
        ss.TestEnv.Dsc = "testing params and state"
        ss.TestEnv.Defaults()
        ss.TestEnv.MinLED = 0
        ss.TestEnv.MaxLED = 19     # all by default
        ss.TestEnv.Trial.Max = 500 # 1000 is too long!
        ss.TestEnv.Validate()

        ss.TrainEnv.Init(0)
        ss.NovelTrainEnv.Init(0)
        ss.TestEnv.Init(0)

    def ConfigNet(ss, net):
        net.InitName(net, "Objrec")
        v1 = net.AddLayer4D("V1", 10, 10, 5, 4, emer.Input)
        v4 = net.AddLayer4D("V4", 5, 5, 7, 7, emer.Hidden)
        it = net.AddLayer2D("IT", 10, 10, emer.Hidden)
        out = net.AddLayer2D("Output", 4, 5, emer.Target)

        net.ConnectLayers(v1, v4, ss.V1V4Prjn, emer.Forward)
        net.BidirConnectLayersPy(v4, it, prjn.NewFull())
        net.BidirConnectLayersPy(it, out, prjn.NewFull())
        
        v4IT = v4.SendPrjns().RecvName("IT")
        itOut = it.SendPrjns().RecvName("Output")
        outIT = it.RecvPrjns().SendName("Output")

        v4IT.SetClass("NovLearn")
        itOut.SetClass("NovLearn")
        outIT.SetClass("NovLearn")

        it.SetRelPos(relpos.Rel(Rel= relpos.RightOf, Other= "V4", YAlign= relpos.Front, Space= 2))
        out.SetRelPos(relpos.Rel(Rel= relpos.RightOf, Other= "IT", YAlign= relpos.Front, Space= 2))

        # about the same on mac with and without threading
        # v4.SetThread(1)
        # it.SetThread(2)

        net.Defaults()
        ss.SetParams("Network", False) # only set Network params
        net.Build()
        ss.InitWts(net)

    def InitWts(ss, net):
        net.InitTopoScales() # sets all wt scales
        net.InitWts()
        net.LrateMult(1) # restore initial learning rate value

    def Init(ss):
        """
        Init restarts the run, and initializes everything, including network weights
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
            return "Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t" % (ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Trial.Cur, ss.Time.Cycle, ss.TrainEnv.String())
        else:
            return "Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t" % (ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TestEnv.Trial.Cur, ss.Time.Cycle, ss.TestEnv.String())

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
        args so that it can be used for various different contexts
        (training, testing, etc).
        """
        ss.Net.InitExt()

        lays = ["V1", "Output"]
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
        if ss.PNovel > 0:
            ss.NovelTrainEnv.Step() # keep in sync

        # Key to query counters FIRST because current state is in NEXT epoch
        # if epoch counter has changed
        epc = ss.TrainEnv.CounterCur(env.Epoch)
        chg = ss.TrainEnv.CounterChg(env.Epoch)

        if chg:
            ss.LogTrnEpc(ss.TrnEpcLog)
            ss.LrateSched(epc)
            if ss.ViewOn and ss.TrainUpdt.value > leabra.AlphaCycle:
                ss.UpdateView(True)
            if epc >= ss.MaxEpcs or (ss.NZeroStop > 0 and ss.NZero >= ss.NZeroStop):
                # done with training..
                ss.RunEnd()
                if ss.TrainEnv.Run.Incr(): # we are done!
                    ss.StopNow = True
                    return
                else:
                    ss.NeedsNewRun = True
                    return

        # note: type must be in place before apply inputs
        ss.Net.LayerByName("Output").SetType(emer.Target)
        if erand.BoolP(ss.PNovel):
            ss.ApplyInputs(ss.NovelTrainEnv)
        else:
            ss.ApplyInputs(ss.TrainEnv)
        ss.AlphaCyc(True)   # train
        ss.TrialStats(True) # accumulate
        if ss.CurImgGrid != 0:
            ss.CurImgGrid.UpdateSig()

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
        ss.InitWts(ss.Net)
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
        else:
            ss.UpdtActRFs()
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

    def LrateSched(ss, epc):
        """
        LrateSched implements the learning rate schedule
        """
        if epc == 40:
            ss.Net.LrateMult(0.5)
            print("dropped lrate 0.5 at epoch: %d\n" % epc)

    def OpenTrainedWts(ss):
        """
        OpenTrainedWts opens trained weights
        """
        ss.Net.OpenWtsJSON("objrec_train1.wts.gz")

    def TrainNovel(ss):
        """
        TrainNovel prepares network for training novel items: loads saved weights
        changes PNovel -- just do Step Run after this.
        """
        ss.NewRun()
        ss.OpenTrainedWts()
        ss.SetParamsSet("NovelLearn", "Network", True)
        ss.TrainEnv.Epoch.Cur = 40
        ss.LrateSched(40)
        ss.PNovel = 0.5

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

        ss.Net.LayerByName("Output").SetType(emer.Compare)
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
        ss.TestEnv.DoObject(idx)
        ss.ApplyInputs(ss.TestEnv)
        ss.AlphaCyc(False)
        ss.TrialStats(False)
        ss.TestEnv.Trial.Cur = cur

    def TestAll(ss):
        """
        TestAll runs through the full set of testing items
        """
        ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
        ss.ActRFs.Reset()
        while True:
            ss.TestTrial(True)
            chg = ss.TestEnv.CounterChg(env.Epoch)
            if chg or ss.StopNow:
                break
        ss.ActRFs.Avg()
        ss.ActRFs.Norm()
        ss.ViewActRFs()

    def RunTestAll(ss):
        """
        RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
        """
        ss.StopNow = False
        ss.TestAll()
        ss.Stopped()

    def UpdtActRFs(ss):
        """
        UpdtActRFs updates activation rf's -- only called during testing
        """
        oly = ss.Net.LayerByName("Output")
        ovt = ss.ValsTsr("Output")
        oly.UnitValsTensor(ovt, "ActM")
        if not "Image" in ss.ValsTsrs:
            ss.ValsTsrs["Image"] = ss.TestEnv.Vis.ImgTsr
        naf = len(ss.ActRFNms)
        if len(ss.ActRFs.RFs) != naf:
            for anm in ss.ActRFNms:
                sp = anm.split( ":")
                lnm = sp[0]
                ly = ss.Net.LayerByName(lnm)
                if ly == 0:
                    continue
                lvt = ss.ValsTsr(lnm)
                ly.UnitValsTensor(lvt, "ActM")
                tnm = sp[1]
                tvt = ss.ValsTsr(tnm)
                ss.ActRFs.AddRF(anm, lvt, tvt)
        for anm in ss.ActRFNms:
            sp = anm.split( ":")
            lnm = sp[0]
            ly = ss.Net.LayerByName(lnm)
            if ly == 0:
                continue
            lvt = ss.ValsTsr(lnm)
            ly.UnitValsTensor(lvt, "ActM")
            tnm = sp[1]
            tvt = ss.ValsTsr(tnm)
            ss.ActRFs.Add(anm, lvt, tvt, 0.01)

    def ViewActRFs(ss):
        """
        ViewActRFs displays act rfs
        """
        for nm in ss.ActRFNms:
            tg = ss.ActRFGrids[nm]
            if tg.Tensor.Len() == 0:
                rf = ss.ActRFs.RFByName(nm)
                tg.SetTensor(rf.NormRF)
            else:
                tg.UpdateSig()

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
        dt.SetCellFloat("PerTrlMSec", row, ss.EpcPerTrlMSec)

        for lnm in ss.LayStatNms :
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            dt.SetCellFloat(ly.Nm+" ActAvg", row, float(ly.Pools[0].ActAvg.ActPAvgEff))

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
            etable.Column("CosDiff", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("PerTrlMSec", etensor.FLOAT64, go.nil, go.nil)]
        )
        for lnm in ss.LayStatNms:
            sch.append(etable.Column(lnm + " ActAvg", etensor.FLOAT64, go.nil, go.nil))
        dt.SetFromSchema(sch, 0)

    def ConfigTrnEpcPlot(ss, plt, dt):
        plt.Params.Title = "Object Recognition Epoch Plot"
        plt.Params.XAxisCol = "Epoch"
        plt.SetTable(dt)

        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("PerTrlMSec", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

        for lnm in ss.LayStatNms:
            plt.SetColParams(lnm+" ActAvg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 0.5)
        return plt

    def LogTstTrl(ss, dt):
        """
        LogTstTrl adds data from current trial to the TstTrlLog table.
        log always contains number of testing items
        """
        epc = ss.TrainEnv.Epoch.Prv

        trl = ss.TestEnv.Trial.Cur
        row = trl

        if dt.Rows <= row:
            dt.SetNumRows(row + 1)

        dt.SetCellFloat("Run", row, float(ss.TrainEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellFloat("Trial", row, float(trl))
        dt.SetCellFloat("Obj", row, float(ss.TestEnv.CurLED))
        dt.SetCellString("TrialName", row, ss.TestEnv.String())
        dt.SetCellFloat("Err", row, ss.TrlErr)
        dt.SetCellFloat("SSE", row, ss.TrlSSE)
        dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
        dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

        for lnm in ss.LayStatNms:
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            dt.SetCellFloat(ly.Nm+" ActM.Avg", row, float(ly.Pools[0].ActM.Avg))

        # note: essential to use Go version of update when called from another goroutine
        ss.TstTrlPlot.GoUpdate()

    def ConfigTstTrlLog(ss, dt):
        # inp := ss.Net.LayerByName("V1").(leabra.LeabraLayer)
        # out := ss.Net.LayerByName("Output").(leabra.LeabraLayer)

        dt.SetMetaData("name", "TstTrlLog")
        dt.SetMetaData("desc", "Record of testing per input pattern")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        nt = ss.TestEnv.Trial.Max
        sch = etable.Schema(
            [etable.Column("Run", etensor.INT64, go.nil, go.nil),
            etable.Column("Epoch", etensor.INT64, go.nil, go.nil),
            etable.Column("Trial", etensor.INT64, go.nil, go.nil),
            etable.Column("Obj", etensor.INT64, go.nil, go.nil),
            etable.Column("TrialName", etensor.STRING, go.nil, go.nil),
            etable.Column("Err", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("SSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("AvgSSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("CosDiff", etensor.FLOAT64, go.nil, go.nil)]
        )
        for lnm in ss.LayStatNms:
            sch.append( etable.Column(lnm + " ActM.Avg", etensor.FLOAT64, go.nil, go.nil))
        dt.SetFromSchema(sch, nt)

    def ConfigTstTrlPlot(ss, plt, dt):
        plt.Params.Title = "Object Recognition Test Trial Plot"
        plt.Params.XAxisCol = "Trial"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Obj", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Err", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("SSE", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0) # default plot
        plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

        for lnm in ss.LayStatNms :
            plt.SetColParams(lnm+" ActM.Avg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 0.5)
        return plt

    def LogTstEpc(ss, dt):
        trl = ss.TstTrlLog
        tix = etable.NewIdxView(trl)
        # epc := ss.TrainEnv.Epoch.Prv // ?

        spl = split.GroupBy(tix, go.Slice_string(["Obj"]))
        split.AggTry(spl, "Err", agg.AggMean)
        objs = spl.AggsToTable(etable.AddAggName)
        no = objs.Rows
        dt.SetNumRows(no)
        for i in range(no):
            dt.SetCellFloat("Obj", i, float(i))
            dt.SetCellFloat("PctErr", i, objs.Cols[1].FloatVal1D(i))
        ss.TstEpcPlot.GoUpdate()

    def ConfigTstEpcLog(ss, dt):
        dt.SetMetaData("name", "TstEpcLog")
        dt.SetMetaData("desc", "Summary stats for testing trials")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        dt.SetFromSchema(etable.Schema(
            [etable.Column("Obj", etensor.INT64, go.nil, go.nil),
            etable.Column("PctErr", etensor.FLOAT64, go.nil, go.nil)]
        ), 0)

    def ConfigTstEpcPlot(ss, plt, dt):
        plt.Params.Title = "Object Recognition Testing Epoch Plot"
        plt.Params.XAxisCol = "Obj"
        plt.Params.Type = eplot.Bar
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Obj", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
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
        nlast = 5
        if nlast > epcix.Len()-1:
            nlast = epcix.Len() - 1
        epcix.Idxs = epcix.Idxs[epcix.Len()-nlast:]

        # params := ss.Params.Name
        params = "params"

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

        dt.SetFromSchema(etable.Schema(
            [etable.Column("Run", etensor.INT64, go.nil, go.nil),
            etable.Column("Params", etensor.STRING, go.nil, go.nil),
            etable.Column("FirstZero", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("SSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("AvgSSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("PctErr", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("PctCor", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("CosDiff", etensor.FLOAT64, go.nil, go.nil)]
        ), 0)

    def ConfigRunPlot(ss, plt, dt):
        plt.Params.Title = "Object Recognition Run Plot"
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

    def ConfigNetView(ss, nv):
        nv.ViewDefaults()
        cam = (nv.Scene().Camera)
        cam.Pose.Pos.Set(0.0, 1.733, 2.3)
        cam.LookAt(mat32.Vec3(0, 0, 0), mat32.Vec3(0, 1, 0))
        # cam.Pose.Quat.SetFromAxisAngle(mat32.Vec3{-1, 0, 0}, 0.4077744)

    def ConfigGui(ss):
        """
        ConfigGui configures the GoGi gui interface for this simulation,
        """
        width = 1600
        height = 1200

        gi.SetAppName("objrec")
        gi.SetAppAbout('This simulation explores how a hierarchy of areas in the ventral stream of visual processing (up to inferotemporal (IT) cortex) can produce robust object recognition that is invariant to changes in position, size, etc of retinal input images. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch6/objrec/README.md">README.md on GitHub</a>.</p>')

        win = gi.NewMainWindow("objrec", "Object Recognition", width, height)
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

        tg = etview.TensorGrid()
        tv.AddTab(tg, "Image")
        tg.SetStretchMax()
        ss.CurImgGrid = tg
        tg.SetTensor(ss.TrainEnv.Vis.ImgTsr)

        plt = eplot.Plot2D()
        tv.AddTab(plt, "TstTrlPlot")
        ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

        plt = eplot.Plot2D()
        tv.AddTab(plt, "TstEpcPlot")
        ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.TstEpcLog)

        plt = eplot.Plot2D()
        tv.AddTab(plt, "RunPlot")
        ss.RunPlot = ss.ConfigRunPlot(plt, ss.RunLog)

        ss.ActRFGrids = {}
        for nm in ss.ActRFNms:
            ss.ActRFGrids[nm] = etview.TensorGrid()
            tg = ss.ActRFGrids[nm]
            tv.AddTab(tg, nm)
            tg.SetStretchMax()

        split.SetSplitsList(go.Slice_float32([.2, .8]))

        recv = win.This()
        
        tbar.AddAction(gi.ActOpts(Label="Init", Icon="update", Tooltip="Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc=UpdtFuncNotRunning), recv, InitCB)

        tbar.AddAction(gi.ActOpts(Label="Train", Icon="run", Tooltip="Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.", UpdateFunc=UpdtFuncNotRunning), recv, TrainCB)
        
        tbar.AddAction(gi.ActOpts(Label="Stop", Icon="stop", Tooltip="Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc=UpdtFuncRunning), recv, StopCB)
        
        tbar.AddAction(gi.ActOpts(Label="Step Trial", Icon="step-fwd", Tooltip="Advances one training trial at a time.", UpdateFunc=UpdtFuncNotRunning), recv, StepTrialCB)
        
        tbar.AddAction(gi.ActOpts(Label="Step Epoch", Icon="fast-fwd", Tooltip="Advances one epoch (complete set of training patterns) at a time.", UpdateFunc=UpdtFuncNotRunning), recv, StepEpochCB)

        tbar.AddAction(gi.ActOpts(Label="Step Run", Icon="fast-fwd", Tooltip="Advances one full training Run at a time.", UpdateFunc=UpdtFuncNotRunning), recv, StepRunCB)
        
        tbar.AddSeparator("spec")

        tbar.AddAction(gi.ActOpts(Label= "Open Trained Wts", Icon= "update", Tooltip= "Open weights trained on first phase of training (excluding 'novel' objects)", UpdateFunc=UpdtFuncNotRunning), recv, OpenTrainedWtsCB)

        tbar.AddAction(gi.ActOpts(Label= "Train Novel", Icon= "update", Tooltip= "prepares network for training novel items: loads saved weight, changes PNovel -- just do Step Run after this..", UpdateFunc=UpdtFuncNotRunning), recv, TrainNovelCB)

        tbar.AddSeparator("test")
        
        tbar.AddAction(gi.ActOpts(Label="Test Trial", Icon="step-fwd", Tooltip="Runs the next testing trial.", UpdateFunc=UpdtFuncNotRunning), recv, TestTrialCB)
        
        tbar.AddAction(gi.ActOpts(Label="Test Item", Icon="step-fwd", Tooltip="Prompts for a specific input pattern name to run, and runs it in testing mode.", UpdateFunc=UpdtFuncNotRunning), recv, TestItemCB)
        
        tbar.AddAction(gi.ActOpts(Label="Test All", Icon="fast-fwd", Tooltip="Tests all of the testing trials.", UpdateFunc=UpdtFuncNotRunning), recv, TestAllCB)

        tbar.AddSeparator("log")
        
        tbar.AddAction(gi.ActOpts(Label="Reset RunLog", Icon="reset", Tooltip="Resets the accumulated log of all Runs, which are tagged with the ParamSet used"), recv, ResetRunLogCB)

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

