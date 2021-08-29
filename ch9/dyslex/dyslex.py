#!/usr/local/bin/pyleabra -i

# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# use:
# pyleabra -i ra25.py 
# to run in gui interactive mode from the command line (or pyleabra, import ra25)
# see main function at the end for startup args

# dyslex simulates normal and disordered (dyslexic) reading performance in terms
# of a distributed representation of word-level knowledge across 
# Orthography, Semantics, and Phonology. It is based on a model by
# Plaut and Shallice (1993). Note that this form of dyslexia is *aquired*
# (via brain lesions such as stroke) and not the more prevalent developmental variety.

from leabra import go, leabra, emer, relpos, eplot, env, agg, patgen, prjn, etable, efile, split, etensor, params, netview, rand, erand, gi, giv, pygiv, pyparams, mat32, metric, simat, pca, clust

import importlib as il  #il.reload(ra25) -- doesn't seem to work for reasons unknown
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

def ClustPlotsCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.ClustPlots()

def ResetRunLogCB(recv, send, sig, data):
    TheSim.RunLog.SetNumRows(0)
    TheSim.RunPlot.Update()

def OpenTrainedWtsCB(recv, send, sig, data):    
    TheSim.OpenTrainedWts()

def NewRndSeedCB(recv, send, sig, data):
    TheSim.NewRndSeed()

def ReadmeCB(recv, send, sig, data):
    gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch9/dyslex/README.md")

def UpdtFuncNotRunning(act):
    act.SetActiveStateUpdt(not TheSim.IsRunning)
    
def UpdtFuncRunning(act):
    act.SetActiveStateUpdt(TheSim.IsRunning)
    
class LesionTypes(Enum):
    """
    LesionTypes is the type of lesion
    """
    NoLesion = 0
    SemanticsFull = 1
    DirectFull = 2
    OShidden = 3
    SPhidden = 4
    OPhidden = 5
    OShidDirectFull = 6
    SPhidDirectFull = 7
    OPhidSemanticsFull = 8
    AllPartial = 9 # do all above partial with partials .1..1
    LesionTypesN = 10

class LesionParams(pygiv.ClassViewObj):
    def __init__(self):
        super(LesionParams, self).__init__()
        self.Lesion = LesionTypes.NoLesion
        self.Proportion = float(0)

TheLesion = LesionParams()        
        
def LesionCB2(recv, send, sig, data):
    TheSim.LesionNet(TheLesion.Lesion, TheLesion.Proportion)

def LesionDialog(vp, obj, name, tags, opts):
    """
    LesionDialog returns a dialog with ClassView editor for python
    class objects under GoGi.
    opts must be a giv.DlgOpts instance
    """
    dlg = gi.NewStdDialog(opts.ToGiOpts(), True, True)
    frame = dlg.Frame()
    prIdx = dlg.PromptWidgetIdx(frame)

    cv = obj.NewClassView(name)
    cv.Frame = gi.Frame(frame.InsertNewChild(gi.KiT_Frame(), prIdx+1, "cv-frame"))
    cv.Config()
    
    dlg.UpdateEndNoSig(True)
    dlg.DialogSig.Connect(dlg, LesionCB2)
    dlg.Open(0, 0, vp, go.nil)
    return dlg

def LesionCB(recv, send, sig, data):
    LesionDialog(TheSim.vp, TheLesion, "lesion", {}, giv.DlgOpts(Title="LesionNet", Prompt="Lesion the network using given type of lesion, and given proportion of neurons (0 < Proportion < 1)"))

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
        self.Lesion = LesionTypes.NoLesion
        self.SetTags("Lesion", 'inactive:"+" desc:"type of lesion -- use Lesion button to lesion"')
        self.LesionProp = float(0)
        self.SetTags("LesionProp", 'inactive:"+" desc:"proportion of neurons lesioned -- use Lesion button to lesion"')
        self.Net = leabra.Network()
        self.SetTags("Net", 'view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"')
        self.TrainPats = etable.Table()
        self.SetTags("TrainPats", 'view:"no-inline" desc:"training patterns"')
        self.Semantics = etable.Table()
        self.SetTags("Semantics", 'view:"no-inline" desc:"properties of semantic features "')
        self.CloseOrthos = etable.Table()
        self.SetTags("CloseOrthos", 'view:"no-inline" desc:"list of items that are close in orthography (for visual error scoring)"')
        self.CloseSems = etable.Table()
        self.SetTags("CloseSems", 'view:"no-inline" desc:"list of items that are close in semantics (for semantic error scoring)"')
        self.TrnEpcLog = etable.Table()
        self.SetTags("TrnEpcLog", 'view:"no-inline" desc:"training epoch-level log data"')
        self.TstEpcLog = etable.Table()
        self.SetTags("TstEpcLog", 'view:"no-inline" desc:"testing epoch-level log data"')
        self.TstTrlLog = etable.Table()
        self.SetTags("TstTrlLog", 'view:"no-inline" desc:"testing trial-level log data"')
        self.TstStats = etable.Table()
        self.SetTags("TstStats", 'view:"no-inline" desc:"aggregate testing stats"')
        self.RunLog = etable.Table()
        self.SetTags("RunLog", 'view:"no-inline" desc:"summary log of each run"')
        self.RunStats = etable.Table()
        self.SetTags("RunStats", 'view:"no-inline" desc:"aggregate stats on all runs"')
        self.SemClustPlot = eplot.Plot2D()
        self.SetTags("SemClustPlot", 'view:"no-inline" desc:"semantics cluster plot"')
        self.Params = params.Sets()
        self.SetTags("Params", 'view:"no-inline" desc:"full collection of param sets"')
        self.ParamSet = str()
        self.SetTags("ParamSet", 'view:"-" desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set -- can use multiple names separated by spaces (don\'t put spaces in ParamSet names!)"')
        self.MaxRuns = int(1)
        self.SetTags("MaxRuns", 'desc:"maximum number of model runs to perform"')
        self.MaxEpcs = int(250)
        self.SetTags("MaxEpcs", 'desc:"maximum number of epochs to run per model run"')
        self.NZeroStop = int(-1)
        self.SetTags("NZeroStop", 'desc:"if a positive number, training will stop after this many epochs with zero SSE"')
        self.TrainEnv = env.FixedTable()
        self.SetTags("TrainEnv", 'desc:"Training environment -- contains everything about iterating over input / output patterns over training"')
        self.TestEnv = env.FixedTable()
        self.SetTags("TestEnv", 'desc:"Testing environment -- manages iterating over testing"')
        self.Time = leabra.Time()
        self.SetTags("Time", 'desc:"leabra timing parameters and state"')
        self.ViewOn = True
        self.SetTags("ViewOn", 'desc:"whether to update the network view while running"')
        self.TrainUpdt = leabra.TimeScales.Quarter
        self.SetTags("TrainUpdt", 'desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"')
        self.TestUpdt = leabra.TimeScales.Cycle
        self.SetTags("TestUpdt", 'desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"')
        self.TestInterval = int(10)
        self.SetTags("TestInterval", 'desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"')
        self.LayStatNms = go.Slice_string(["OShidden"])
        self.SetTags("LayStatNms", 'desc:"names of layers to collect more detailed stats on (avg act, etc)"')

        # statistics: note use float64 as that is best for etable.Table
        self.TrlName = str()
        self.SetTags("TrlName", 'inactive:"+" desc:"name of current input pattern"')
        self.TrlPhon = str()
        self.SetTags("TrlPhon", 'inactive:"+" desc:"name of closest phonology pattern"')
        self.TrlPhonSSE = float()
        self.SetTags("TrlPhonSSE", 'inactive:"+" desc:"SSE for closest phonology pattern -- > 3 = blend"')
        self.TrlConAbs = float()
        self.SetTags("TrlConAbs", 'inactive:"+" desc:"0 = concrete, 1 = abstract"')
        self.TrlVisErr = float()
        self.SetTags("TrlVisErr", 'inactive:"+" desc:"visual error -- close to similar other"')
        self.TrlSemErr = float()
        self.SetTags("TrlSemErr", 'inactive:"+" desc:"semantic error -- close to similar other"')
        self.TrlVisSemErr = float()
        self.SetTags("TrlVisSemErr", 'inactive:"+" desc:"visual + semantic error -- close to similar other"')
        self.TrlBlendErr = float()
        self.SetTags("TrlBlendErr", 'inactive:"+" desc:"blend error"')
        self.TrlOtherErr = float()
        self.SetTags("TrlOtherErr", 'inactive:"+" desc:"some other error"')
        self.TrlErr = float()
        self.SetTags("TrlErr", 'inactive:"+" desc:"1 if trial was error, 0 if correct -- based on *closest pattern* stat, not SSE"')
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
        self.ValsTsrs = {}
        self.SetTags("ValsTsrs", 'view:"-" desc:"for holding layer values"')
        self.IsRunning = False
        self.SetTags("IsRunning", 'view:"-" desc:"true if sim is running"')
        self.StopNow = False
        self.SetTags("StopNow", 'view:"-" desc:"flag to stop running"')
        self.NeedsNewRun = False
        self.SetTags("NeedsNewRun", 'view:"-" desc:"flag to initialize NewRun if last one finished"')
        self.RndSeed = int(10)
        self.SetTags("RndSeed", 'inactive:"+" desc:"the current random seed"')
        self.vp  = 0
        self.SetTags("vp", 'view:"-" desc:"viewport"')

    def InitParams(ss):
        """
        Sets the default set of parameters -- Base is always applied, and others can be optionally
        selected to apply on top of that
        """
        ss.Params.OpenJSON("dyslex.params")
        ss.Defaults()

    def Defaults(ss):
        ss.Lesion = LesionTypes.NoLesion
        ss.LesionProp = 0

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
        if ss.MaxRuns == 0: # allow user override
            ss.MaxRuns = 1
        if ss.MaxEpcs == 0: # allow user override
            ss.MaxEpcs = 250
            ss.NZeroStop = -1

        ss.TrainEnv.Nm = "TrainEnv"
        ss.TrainEnv.Dsc = "training params and state"
        ss.TrainEnv.Table = etable.NewIdxView(ss.TrainPats)
        ss.TrainEnv.Validate()
        ss.TrainEnv.Run.Max = ss.MaxRuns # note: we are not setting epoch max -- do that manually

        ss.TestEnv.Nm = "TestEnv"
        ss.TestEnv.Dsc = "testing params and state"
        ss.TestEnv.Table = etable.NewIdxView(ss.TrainPats)
        ss.TestEnv.Sequential = True
        ss.TestEnv.Validate()

        ss.TrainEnv.Init(0)
        ss.TestEnv.Init(0)

    def ConfigNet(ss, net):
        net.InitName(net, "Dyslex")
        ort = net.AddLayer2D("Orthography", 6, 8, emer.Input)
        oph = net.AddLayer2D("OPhidden", 7, 7, emer.Hidden)
        phn = net.AddLayer4D("Phonology", 1, 7, 7, 2, emer.Target)
        osh = net.AddLayer2D("OShidden", 10, 7, emer.Hidden)
        sph = net.AddLayer2D("SPhidden", 10, 7, emer.Hidden)
        sem = net.AddLayer2D("Semantics", 10, 12, emer.Target)

        full = prjn.NewFull()
        net.BidirConnectLayersPy(ort, osh, full)
        net.BidirConnectLayersPy(osh, sem, full)
        net.BidirConnectLayersPy(sem, sph, full)
        net.BidirConnectLayersPy(sph, phn, full)
        net.BidirConnectLayersPy(ort, oph, full)
        net.BidirConnectLayersPy(oph, phn, full)

        # lateral cons
        net.LateralConnectLayer(ort, full)
        net.LateralConnectLayer(sem, full)
        net.LateralConnectLayer(phn, full)

        oph.SetRelPos(relpos.Rel(Rel= relpos.RightOf, Other= "Orthography", YAlign= relpos.Front, Space= 1))
        phn.SetRelPos(relpos.Rel(Rel= relpos.RightOf, Other= "OPhidden", YAlign= relpos.Front, Space= 1))
        osh.SetRelPos(relpos.Rel(Rel= relpos.Above, Other= "Orthography", YAlign= relpos.Front, XAlign= relpos.Left, XOffset= 4))
        sph.SetRelPos(relpos.Rel(Rel= relpos.Above, Other= "Phonology", YAlign= relpos.Front, XAlign= relpos.Left, XOffset= 2))
        sem.SetRelPos(relpos.Rel(Rel= relpos.Above, Other= "OShidden", YAlign= relpos.Front, XAlign= relpos.Left, XOffset= 4))

        net.Defaults()
        ss.SetParams("Network", False) # only set Network params
        net.Build()
        net.InitWts()

    def LesionNet(ss, les, prop):
        """
        LesionNet does lesion of network with given proportion of neurons damaged
        0 < prop < 1.
        """
        net = ss.Net
        lesStep = float(0.1)
        
        if les == LesionTypes.AllPartial:
            for ls in range(LesionTypes.OShidden.value, LesionTypes.AllPartial.value):
                for prp in np.arange(lesStep, 1.0, lesStep):
                    ss.UnLesionNet(net)
                    ss.LesionNetImpl(net, LesionTypes(ls), prp)
                    ss.TestAll()
        else:
            ss.UnLesionNet(net)
            ss.LesionNetImpl(net, les, prop)

    def UnLesionNet(ss, net):
        net.LayersSetOff(False)
        net.UnLesionNeurons()
        net.InitActs()

    def LesionNetImpl(ss, net, les, prop):
        ss.Lesion = LesionTypes(les)
        ss.LesionProp = float(prop)
        if les == LesionTypes.NoLesion:
            pass
        if les == LesionTypes.SemanticsFull:
            net.LayerByName("OShidden").SetOff(True)
            net.LayerByName("Semantics").SetOff(True)
            net.LayerByName("SPhidden").SetOff(True)
        if les == LesionTypes.DirectFull:
            net.LayerByName("OPhidden").SetOff(True)
        if les == LesionTypes.OShidden:
            leabra.Layer(net.LayerByName("OShidden")).LesionNeurons(prop)
        if les == LesionTypes.SPhidden:
            leabra.Layer(net.LayerByName("SPhidden")).LesionNeurons(prop)
        if les == LesionTypes.OPhidden:
            leabra.Layer(net.LayerByName("OPhidden")).LesionNeurons(prop)
        if les == LesionTypes.OShidDirectFull:
            net.LayerByName("OPhidden").SetOff(True)
            leabra.Layer(net.LayerByName("OShidden")).LesionNeurons(prop)
        if les == LesionTypes.SPhidDirectFull:
            net.LayerByName("OPhidden").SetOff(True)
            leabra.Layer(net.LayerByName("SPhidden")).LesionNeurons(prop)
        if les == LesionTypes.OPhidSemanticsFull:
            net.LayerByName("OShidden").SetOff(True)
            net.LayerByName("Semantics").SetOff(True)
            net.LayerByName("SPhidden").SetOff(True)
            leabra.Layer(net.LayerByName("OPhidden")).LesionNeurons(prop)

    def Init(ss):
        """
        Init restarts the run, and initializes everything, including network weights
        and resets the epoch log table
        """
        rand.Seed(ss.RndSeed)
        ss.ConfigEnv()
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

        lays = go.Slice_string(["Orthography", "Semantics", "Phonology"])
        for lnm in lays :
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            pats = en.State(ly.Nm)
            if pats != 0:
                ly.ApplyExt(pats)

    def SetInputLayer(ss, layno):
        """
        SetInputLayer determines which layer is the input -- others are targets
        0 = Ortho, 1 = Sem, 2 = Phon, 3 = Ortho + compare for others
        """
        lays = go.Slice_string(["Orthography", "Semantics", "Phonology"])
        test = False
        if layno > 2:
            layno = 0
            test = True
        for i, lnm in enumerate(lays):
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            if i == layno:
                ly.SetType(emer.Input)
            else:
                if test:
                    ly.SetType(emer.Compare)
                else:
                    ly.SetType(emer.Target)

    def SetRndInputLayer(ss):
        """
        SetRndInputLayer sets one of 3 visible layers as input at random
        """
        ss.SetInputLayer(rand.Intn(3))

    def TrainTrial(ss):
        """
        TrainTrial runs one trial of training using TrainEnv
        """
        if ss.NeedsNewRun:
            ss.NewRun()

        ss.TrainEnv.Step()

        epc = env.CounterCur(ss.TrainEnv, env.Epoch)
        chg = env.CounterChg(ss.TrainEnv, env.Epoch)
        if chg:
            ss.LogTrnEpc(ss.TrnEpcLog)
            ss.LrateSched(epc)
            if ss.ViewOn and ss.TrainUpdt.value > leabra.AlphaCycle:
                ss.UpdateView(True)
            if ss.TestInterval > 0 and epc%ss.TestInterval == 0:
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

        # note: type must be in place before apply inputs
        if ss.TrainEnv.Epoch.Cur < ss.MaxEpcs-10:
            ss.SetRndInputLayer()
        else:
            ss.SetInputLayer(0) # final training on Ortho reading

        ss.ApplyInputs(ss.TrainEnv)
        ss.AlphaCyc(True)                              # train
        ss.TrialStats(True, ss.TrainEnv.TrialName.Cur) # accumulate

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
        ss.Time.Reset()
        ss.Net.InitWts()
        ss.Net.LrateMult(1)
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

        ss.TrlPhonSSE = 0
        ss.TrlVisErr = 0
        ss.TrlSemErr = 0
        ss.TrlVisSemErr = 0
        ss.TrlBlendErr = 0
        ss.TrlOtherErr = 0
        ss.TrlErr = 0
        ss.TrlSSE = 0
        ss.TrlAvgSSE = 0
        ss.TrlCosDiff = 0
        ss.EpcSSE = 0
        ss.EpcAvgSSE = 0
        ss.EpcPctErr = 0
        ss.EpcCosDiff = 0

    def TrialStats(ss, accum, trlnm):
        """
        TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
        accum is true.  Note that we're accumulating stats here on the Sim side so the
        core algorithm side remains as simple as possible, and doesn't need to worry about
        different time-scales over which stats could be accumulated etc.
        You can also aggregate directly from log data, as is done for testing stats
        """
        ss.TrlCosDiff = 0
        ss.TrlSSE, ss.TrlAvgSSE = 0, 0
        ntrg = 0
        for lyi in ss.Net.Layers:
            ly = leabra.Layer(handle=lyi)
            if ly.Typ != emer.Target:
                continue
            ss.TrlCosDiff += float(ly.CosDiff.Cos)
            sse = ly.SSE(0.5)
            ss.TrlSSE += sse
            ss.TrlAvgSSE += sse / len(ly.Neurons)
            ntrg += 1
        if ntrg > 0:
            ss.TrlCosDiff /= float(ntrg)
            ss.TrlSSE /= float(ntrg)
            ss.TrlAvgSSE /= float(ntrg)

        if ss.TrlSSE == 0:
            ss.TrlErr = 0
        else:
            ss.TrlErr = 1
        if accum:
            ss.SumErr += ss.TrlErr
            ss.SumSSE += ss.TrlSSE
            ss.SumAvgSSE += ss.TrlAvgSSE
            ss.SumCosDiff += ss.TrlCosDiff

        ss.TrlName = trlnm
        pidx = ss.TrainPats.RowsByString("Name", trlnm, etable.Equals, etable.UseCase)[0]
        if pidx < 20:
            ss.TrlConAbs = 0
        else:
            ss.TrlConAbs = 1
        if not accum: # test
            ss.DyslexStats(ss.Net)

        return

    def DyslexStats(ss, net):
        """
        DyslexStats computes dyslexia pronunciation, semantics stats
        """
        rcn = ss.ClosestStat(net, "Phonology", "ActM", ss.TrainPats, "Phonology", "Name")
        sse = rcn[1]
        cnm = rcn[2]
        ss.TrlPhon = cnm
        ss.TrlPhonSSE = float(sse)
        if sse > 3:
            ss.TrlBlendErr = 1
        else:
            ss.TrlBlendErr = 0
            ss.TrlVisErr = 0
            ss.TrlSemErr = 0
            ss.TrlVisSemErr = 0
            ss.TrlOtherErr = 0
            if ss.TrlName != ss.TrlPhon:
                ss.TrlVisErr = ss.ClosePat(ss.TrlName, ss.TrlPhon, ss.CloseOrthos)
                ss.TrlSemErr = ss.ClosePat(ss.TrlName, ss.TrlPhon, ss.CloseSems)
                if ss.TrlVisErr > 0 and ss.TrlSemErr > 0:
                    ss.TrlVisSemErr = 1
                if ss.TrlVisErr == 0 and ss.TrlSemErr == 0:
                    ss.TrlOtherErr = 1

    def ClosestStat(ss, net, lnm, varnm, dt, colnm, namecol):
        """
        ClosestStat finds the closest pattern in given column of given table to
        given layer activation pattern using given variable.  Returns the row number,
        sse value, and value of a column named namecol for that row if non-empty.
        Column must be etensor.Float32
        """
        vt = ss.ValsTsr(lnm)
        ly = leabra.Layer(net.LayerByName(lnm))
        ly.UnitValsTensor(vt, varnm)
        col = dt.ColByName(colnm)
        
        rc = metric.ClosestRow32Py(vt, etensor.Float32(col), metric.SumSquaresBinTol)
        row = int(rc[0])
        sse = float(rc[1])
        nm = ""
        if namecol != "":
            nm = dt.CellString(namecol, row)
        return (row, sse, nm)

    def ClosePat(ss, trlnm, phon, clsdt):
        """
        ClosePat looks up phon pattern name in given table of close names -- if found returns 1, else 0
        """
        rws = clsdt.RowsByString(trlnm, phon, etable.Equals, etable.UseCase)
        return float(len(rws))

    def TrainEpoch(ss):
        """
        TrainEpoch runs training trials for remainder of this epoch
        """
        ss.SetParams("Network", False)
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
        """
        if epc == 100:
            ss.Net.LrateMult(1)

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

    def OpenTrainedWts(ss):
        """
        OpenTrainedWts opens trained weights
        """
        ss.Net.OpenWtsJSON("trained.wts")

    def TestTrial(ss, returnOnChg):
        """
        TestTrial runs one trial of testing -- always sequentially presented inputs
        """
        ss.TestEnv.Step()

        chg = env.CounterChg(ss.TestEnv, env.Epoch)
        if chg:
            if ss.ViewOn and ss.TestUpdt.value > leabra.AlphaCycle:
                ss.UpdateView(False)
            ss.LogTstEpc(ss.TstEpcLog)
            if returnOnChg:
                return

        ss.SetInputLayer(3)
        ss.ApplyInputs(ss.TestEnv)
        ss.AlphaCyc(False)
        ss.TrialStats(False, ss.TestEnv.TrialName.Cur)
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
        ss.TrialStats(False, ss.TestEnv.TrialName.Cur) # !accumulate
        ss.TestEnv.Trial.Cur = cur

    def TestAll(ss):
        """
        TestAll runs through the full set of testing items
        """
        ss.SetParams("Network", False)
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

    def OpenPat(ss, dt, fname, name, desc):
        dt.OpenCSV(fname, etable.Tab)
        dt.SetMetaData("name", name)
        dt.SetMetaData("desc", desc)

    def OpenPats(ss):
        ss.OpenPat(ss.TrainPats, "train_pats.tsv", "TrainPats", "Training patterns")
        ss.OpenPat(ss.Semantics, "semantics.tsv", "Semantics", "Semantics features and properties")
        ss.OpenPat(ss.CloseOrthos, "close_orthos.tsv", "CloseOrthos", "Close Orthography items")
        ss.OpenPat(ss.CloseSems, "close_sems.tsv", "CloseSems", "Close Semantic items")

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

        for lnm in ss.LayStatNms :
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            dt.SetCellFloat(ly.Nm+" ActAvg", row, float(ly.Pools[0].ActAvg.ActPAvgEff))

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
        for lnm in ss.LayStatNms :
            sch.append( etable.Column(lnm + " ActAvg", etensor.FLOAT64, go.nil, go.nil))
        dt.SetFromSchema(sch, 0)

    def ConfigTrnEpcPlot(ss, plt, dt):
        plt.Params.Title = "Dyslex Epoch Plot"
        plt.Params.XAxisCol = "Epoch"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("AvgSSE", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) # default plot
        plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

        for lnm in ss.LayStatNms :
            plt.SetColParams(lnm+" ActAvg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 0.5)
        return plt

    def LogTstTrl(ss, dt):
        """
        LogTstTrl adds data from current trial to the TstTrlLog table.
    # this is triggered by increment so use previous value
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
        dt.SetCellString("TrialName", row, ss.TrlName.split( "_")[0])
        dt.SetCellString("Phon", row, ss.TrlPhon)
        dt.SetCellFloat("PhonSSE", row, ss.TrlPhonSSE)
        dt.SetCellFloat("ConAbs", row, ss.TrlConAbs)
        dt.SetCellFloat("Vis", row, ss.TrlVisErr)
        dt.SetCellFloat("Sem", row, ss.TrlSemErr)
        dt.SetCellFloat("VisSem", row, ss.TrlVisSemErr)
        dt.SetCellFloat("Blend", row, ss.TrlBlendErr)
        dt.SetCellFloat("Other", row, ss.TrlOtherErr)
        dt.SetCellFloat("Err", row, ss.TrlErr)
        dt.SetCellFloat("SSE", row, ss.TrlSSE)
        dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
        dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

        for lnm in ss.LayStatNms :
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            dt.SetCellFloat(ly.Nm+" ActM.Avg", row, float(ly.Pools[0].ActM.Avg))

        # note: essential to use Go version of update when called from another goroutine
        ss.TstTrlPlot.GoUpdate()

    def ConfigTstTrlLog(ss, dt):
        dt.SetMetaData("name", "TstTrlLog")
        dt.SetMetaData("desc", "Record of testing per input pattern")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        nt = ss.TestEnv.Table.Len() # number in view
        sch = etable.Schema(
            [etable.Column("Run", etensor.INT64, go.nil, go.nil),
            etable.Column("Epoch", etensor.INT64, go.nil, go.nil),
            etable.Column("Trial", etensor.INT64, go.nil, go.nil),
            etable.Column("TrialName", etensor.STRING, go.nil, go.nil),
            etable.Column("Phon", etensor.STRING, go.nil, go.nil),
            etable.Column("PhonSSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("ConAbs", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("Vis", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("Sem", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("VisSem", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("Blend", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("Other", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("Err", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("SSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("AvgSSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("CosDiff", etensor.FLOAT64, go.nil, go.nil)]
        )
        for lnm in ss.LayStatNms :
            sch.append( etable.Column(lnm + " ActM.Avg", etensor.FLOAT64, go.nil, go.nil))
        dt.SetFromSchema(sch, nt)

    def ConfigTstTrlPlot(ss, plt, dt):
        plt.Params.Title = "Dyslex Test Trial Plot"
        plt.Params.XAxisCol = "TrialName"
        plt.Params.Type = eplot.Bar
        plt.SetTable(dt) # this sets defaults so set params after
        plt.Params.XAxisRot = 45
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Phon", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("PhonSSE", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("ConAbs", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("Vis", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("Sem", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("VisSem", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("Blend", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("Other", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("Err", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

        for lnm in ss.LayStatNms :
            plt.SetColParams(lnm+" ActM.Avg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 0.5)
        return plt

    def LesionStr(ss):
        if ss.Lesion.value <= LesionTypes.DirectFull.value:
            return str(ss.Lesion)
        return "%s %g" % (ss.Lesion, ss.LesionProp)

    def LogTstEpc(ss, dt):
        row = dt.Rows
        dt.SetNumRows(row + 1)

        trl = ss.TstTrlLog
        tix = etable.NewIdxView(trl)
        epc = ss.TrainEnv.Epoch.Prv # ?

        cols = go.Slice_string(["Vis", "Sem", "VisSem", "Blend", "Other"])

        spl = split.GroupBy(tix, go.Slice_string(["ConAbs"]))

        for cl in cols :
            split.Agg(spl, cl, agg.AggSum)
        tst = spl.AggsToTable(etable.ColNameOnly)
        ss.TstStats = tst

        dt.SetCellFloat("Run", row, float(ss.TrainEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellString("Lesion", row, ss.LesionStr())
        dt.SetCellFloat("LesionProp", row, float(ss.LesionProp))

        for cl in cols :
            dt.SetCellFloat("Con"+cl, row, tst.CellFloat(cl, 0))
            dt.SetCellFloat("Abs"+cl, row, tst.CellFloat(cl, 1))

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
            etable.Column("Lesion", etensor.STRING, go.nil, go.nil),
            etable.Column("LesionProp", etensor.FLOAT64, go.nil, go.nil)]
        )
        cols = go.Slice_string(["Vis", "Sem", "VisSem", "Blend", "Other"])
        for ty in go.Slice_string(["Con", "Abs"]) :
            for cl in cols :
                sch.append( etable.Column(ty + cl, etensor.FLOAT64, go.nil, go.nil))
        dt.SetFromSchema(sch, 0)

    def ConfigTstEpcPlot(ss, plt, dt):
        plt.Params.Title = "Dyslex Testing Epoch Plot"
        plt.Params.XAxisCol = "Lesion"
        plt.Params.Type = eplot.Bar
        plt.SetTable(dt) # this sets defaults so set params after
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Lesion", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("LesionProp", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

        cols = go.Slice_string(["Vis", "Sem", "VisSem", "Blend", "Other"])
        for ty in go.Slice_string(["Con", "Abs"]) :
            for cl in cols :
                plt.SetColParams(ty+cl, eplot.On, eplot.FixMin, 0, eplot.FixMax, 10)
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

        params = ""

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
        plt.Params.Title = "Dyslex Run Plot"
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

    def ClustPlots(ss):
        """
        ClustPlots does all cluster plots
        """
        if ss.SemClustPlot == 0:
            ss.SemClustPlot = eplot.Plot2D()

        tpcp = ss.TrainPats.Clone()
        nm = tpcp.ColByName("Name")
        for r in range(tpcp.Rows):
            n = nm.StringVal1D(r)
            n = n.split( "_")[0]
            nm.SetString1D(r, n)
        ss.ClustPlot(ss.SemClustPlot, etable.NewIdxView(tpcp), "Semantics")

    def ClustPlot(ss, plt, ix, colNm):
        """
        ClustPlot does one cluster plot on given table column
        """
        nm = ix.Table.MetaData["name"]
        smat = simat.SimMat()
        smat.TableColStd(ix, colNm, "Name", False, metric.InvCosine)
        pt = etable.Table()
        clust.Plot(pt, clust.GlomStd(smat, clust.Contrast), smat)
        plt.InitName(plt, colNm)
        plt.Params.Title = "Cluster Plot of: " + nm + " " + colNm
        plt.Params.XAxisCol = "X"
        plt.SetTable(pt)

        plt.SetColParams("X", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Y", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Label", True, False, 0, False, 0)

    def ConfigNetView(ss, nv):
        nv.ViewDefaults()
        nv.Scene().Camera.Pose.Pos.Set(0, 1.05, 2.75)
        nv.Scene().Camera.LookAt(mat32.Vec3(0, 0, 0), mat32.Vec3(0, 1, 0))

    def ConfigGui(ss):
        """
        ConfigGui configures the GoGi gui interface for this simulation,
        """
        width = 1600
        height = 1200

        gi.SetAppName("dyslex")
        gi.SetAppAbout('Simulates normal and disordered (dyslexic) reading performance in terms of a distributed representation of word-level knowledge across Orthography, Semantics, and Phonology. It is based on a model by Plaut and Shallice (1993). Note that this form of dyslexia is *aquired* (via brain lesions such as stroke) and not the more prevalent developmental variety.  See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch9/dyslex/README.md">README.md on GitHub</a>.</p>')

        win = gi.NewMainWindow("dyslex", "Dyslex", width, height)
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
        
        tbar.AddAction(gi.ActOpts(Label="Test Item", Icon="step-fwd", Tooltip="Prompts for a specific input pattern name to run, and runs it in testing mode.", UpdateFunc=UpdtFuncNotRunning), recv, TestItemCB)
        
        tbar.AddAction(gi.ActOpts(Label="Test All", Icon="fast-fwd", Tooltip="Tests all of the testing trials.", UpdateFunc=UpdtFuncNotRunning), recv, TestAllCB)

        tbar.AddSeparator("log")
        
        tbar.AddAction(gi.ActOpts(Label= "Open Trained Wts", Icon= "update", Tooltip= "Open weights trained on first phase of training (excluding 'novel' objects)", UpdateFunc=UpdtFuncNotRunning), recv, OpenTrainedWtsCB)

        tbar.AddAction(gi.ActOpts(Label= "Lesion", Icon= "cut", Tooltip= "Lesion network"), recv, LesionCB)
            
        tbar.AddAction(gi.ActOpts(Label="Reset RunLog", Icon="reset", Tooltip="Resets the accumulated log of all Runs, which are tagged with the ParamSet used"), recv, ResetRunLogCB)

        tbar.AddSeparator("anal")

        tbar.AddAction(gi.ActOpts(Label= "Cluster Plot", Icon= "file-image", Tooltip= "run cluster plot on input patterns."), recv, ClustPlotsCB)

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

