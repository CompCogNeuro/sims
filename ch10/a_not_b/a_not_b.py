#!/usr/local/bin/pyleabra -i

# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# use:
# pyleabra -i ra25.py
# to run in gui interactive mode from the command line (or pyleabra, import ra25)
# see main function at the end for startup args

# a_not_b explores how the development of PFC active maintenance abilities can help
# to make behavior more flexible, in the sense that it can rapidly shift with changes
# in the environment. The development of flexibility has been extensively explored
# in the context of Piaget's famous A-not-B task, where a toy is first hidden several
# times in one hiding location (A), and then hidden in a new location (B). Depending
# on various task parameters, young kids reliably reach back at A instead of updating
# to B.

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
    mat32,
    metric,
    simat,
    pca,
    clust,
    etview,
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


def DefaultsCB(recv, send, sig, data):
    TheSim.Defaults()
    TheSim.Init()
    TheSim.UpdateClassView()
    TheSim.vp.SetNeedsFullRender()


def NewRndSeedCB(recv, send, sig, data):
    TheSim.NewRndSeed()


def ReadmeCB(recv, send, sig, data):
    gi.OpenURL(
        "https://github.com/CompCogNeuro/sims/blob/master/ch10/a_not_b/README.md"
    )


def UpdateFuncNotRunning(act):
    act.SetActiveStateUpdate(not TheSim.IsRunning)


def UpdateFuncRunning(act):
    act.SetActiveStateUpdate(TheSim.IsRunning)


class Delays(Enum):
    """
    Delays is delay case to use
    """

    Delay3 = 0
    Delay5 = 1
    Delay1 = 2
    DelaysN = 3


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
        self.Delay = Delays.Delay3
        self.SetTags("Delay", 'desc:"which delay to use -- pres Init when changing"')
        self.RecurrentWt = float(0.4)
        self.SetTags(
            "RecurrentWt",
            'def:"0.4" step:"0.01" desc:"strength of recurrent weight in Hidden layer from each unit back to self"',
        )
        self.Net = leabra.Network()
        self.SetTags(
            "Net",
            'view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"',
        )
        self.Delay3Pats = etable.Table()
        self.SetTags("Delay3Pats", 'view:"no-inline" desc:"delay 3 patterns"')
        self.Delay5Pats = etable.Table()
        self.SetTags("Delay5Pats", 'view:"no-inline" desc:"delay 5 patterns"')
        self.Delay1Pats = etable.Table()
        self.SetTags("Delay1Pats", 'view:"no-inline" desc:"delay 1 patterns"')
        self.TrnTrlLog = etable.Table()
        self.SetTags(
            "TrnTrlLog", 'view:"no-inline" desc:"testing trial-level log data"'
        )
        self.Params = params.Sets()
        self.SetTags("Params", 'view:"no-inline" desc:"full collection of param sets"')
        self.ParamSet = str()
        self.SetTags(
            "ParamSet",
            'view:"-" desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set -- can use multiple names separated by spaces (don\'t put spaces in ParamSet names!)"',
        )
        self.MaxRuns = int(1)
        self.SetTags("MaxRuns", 'desc:"maximum number of model runs to perform"')
        self.MaxEpcs = int(1)
        self.SetTags("MaxEpcs", 'desc:"maximum number of epochs to run per model run"')
        self.TrainEnv = env.FixedTable()
        self.SetTags(
            "TrainEnv",
            'desc:"Training environment -- contains everything about iterating over input / output patterns over training"',
        )
        self.Time = leabra.Time()
        self.Time.CycPerQtr = 4  # key!
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
        self.TstRecLays = go.Slice_string(
            ["Location", "Cover", "Toy", "Hidden", "GazeExpect", "Reach"]
        )
        self.SetTags(
            "TstRecLays",
            'desc:"names of layers to record activations etc of during testing"',
        )

        # statistics: note use float64 as that is best for etable.Table
        self.PrvGpName = str()
        self.SetTags("PrvGpName", 'view:"-" desc:"previous group name"')
        self.Win = 0
        self.SetTags("Win", 'view:"-" desc:"main GUI window"')
        self.NetView = 0
        self.SetTags("NetView", 'view:"-" desc:"the network viewer"')
        self.ToolBar = 0
        self.SetTags("ToolBar", 'view:"-" desc:"the master toolbar"')
        self.TrnTrlTable = 0
        self.SetTags("TrnTrlTable", 'view:"-" desc:"the train trial table view"')
        self.TrnTrlPlot = 0
        self.SetTags("TrnTrlPlot", 'view:"-" desc:"the train trial plot"')
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
        ss.Params.OpenJSON("a_not_b.params")
        ss.Defaults()

    def Defaults(ss):
        ss.RecurrentWt = 0.4
        ss.Time.CycPerQtr = 4  # key!

    def Config(ss):
        """
        Config configures all the elements using the standard functions
        """
        ss.InitParams()
        ss.OpenPats()
        ss.ConfigEnv()
        ss.ConfigNet(ss.Net)
        ss.ConfigTrnTrlLog(ss.TrnTrlLog)

    def ConfigEnv(ss):
        if ss.MaxRuns == 0:  # allow user override
            ss.MaxRuns = 1
        if ss.MaxEpcs == 0:  # allow user override
            ss.MaxEpcs = 1

        ss.TrainEnv.Nm = "TrainEnv"
        ss.TrainEnv.Dsc = "training params and state"
        if ss.Delay == Delays.Delay3:
            ss.TrainEnv.Table = etable.NewIndexView(ss.Delay3Pats)
        if ss.Delay == Delays.Delay5:
            ss.TrainEnv.Table = etable.NewIndexView(ss.Delay5Pats)
        if ss.Delay == Delays.Delay1:
            ss.TrainEnv.Table = etable.NewIndexView(ss.Delay1Pats)
        ss.TrainEnv.Sequential = True
        ss.TrainEnv.Validate()
        ss.TrainEnv.Run.Max = (
            ss.MaxRuns
        )  # note: we are not setting epoch max -- do that manually

        ss.TrainEnv.Init(0)

    def ConfigNet(ss, net):
        net.InitName(net, "AnotB")
        loc = net.AddLayer2D("Location", 1, 3, emer.Input)
        cvr = net.AddLayer2D("Cover", 1, 2, emer.Input)
        toy = net.AddLayer2D("Toy", 1, 2, emer.Input)
        hid = net.AddLayer2D("Hidden", 1, 3, emer.Hidden)
        gze = net.AddLayer2D("GazeExpect", 1, 3, emer.Compare)
        rch = net.AddLayer2D("Reach", 1, 3, emer.Compare)

        full = prjn.NewFull()
        self = prjn.NewOneToOne()
        net.ConnectLayers(loc, hid, full, emer.Forward)
        net.ConnectLayers(cvr, hid, full, emer.Forward)
        net.ConnectLayers(toy, hid, full, emer.Forward)
        net.ConnectLayers(hid, hid, self, emer.Lateral)
        net.ConnectLayers(hid, gze, full, emer.Forward)
        net.ConnectLayers(hid, rch, full, emer.Forward)
        net.ConnectLayers(gze, gze, self, emer.Lateral)

        cvr.SetRelPos(
            relpos.Rel(
                Rel=relpos.RightOf, Other="Location", YAlign=relpos.Front, Space=1
            )
        )
        toy.SetRelPos(
            relpos.Rel(Rel=relpos.RightOf, Other="Cover", YAlign=relpos.Front, Space=1)
        )
        hid.SetRelPos(
            relpos.Rel(
                Rel=relpos.Above,
                Other="Cover",
                YAlign=relpos.Front,
                XAlign=relpos.Left,
                YOffset=1,
                XOffset=-1,
            )
        )
        gze.SetRelPos(
            relpos.Rel(
                Rel=relpos.Above,
                Other="Hidden",
                YAlign=relpos.Front,
                XAlign=relpos.Left,
                XOffset=-4,
            )
        )
        rch.SetRelPos(
            relpos.Rel(
                Rel=relpos.RightOf, Other="GazeExpect", YAlign=relpos.Front, Space=4
            )
        )

        net.Defaults()
        ss.SetParams("Network", False)  # only set Network params
        net.Build()
        ss.InitWts(ss.Net)

    def InitWts(ss, net):
        net.InitWts()
        hid = leabra.Layer(ss.Net.LayerByName("Hidden"))
        fmloc = leabra.LeabraPrjn(hid.RcvPrjns.SendName("Location"))
        gze = leabra.Layer(ss.Net.LayerByName("GazeExpect"))
        hidgze = leabra.LeabraPrjn(gze.RcvPrjns.SendName("Hidden"))
        rch = leabra.Layer(ss.Net.LayerByName("Reach"))
        hidrch = leabra.LeabraPrjn(rch.RcvPrjns.SendName("Hidden"))
        for i in range(3):
            fmloc.SetSynValue("Wt", i, i, 0.7)
            hidgze.SetSynValue("Wt", i, i, 0.7)
            hidrch.SetSynValue("Wt", i, i, 0.7)

    def Init(ss):
        """
            Init restarts the run, and initializes everything, including network weights

        # all sheets
            and resets the epoch log table
        """
        rand.Seed(ss.RndSeed)
        ss.StopNow = False
        ss.SetParams("", False)
        ss.ConfigEnv()
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
        return "Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t" % (
            ss.TrainEnv.Run.Cur,
            ss.TrainEnv.Epoch.Cur,
            ss.TrainEnv.Trial.Cur,
            ss.Time.Cycle,
            ss.TrainEnv.TrialName.Cur,
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

    def ApplyInputs(ss, en):
        """
            ApplyInputs applies input patterns from given envirbonment.
            It is good practice to have this be a separate method with appropriate

        # going to the same layers, but good practice and cheap anyway
            args so that it can be used for various different contexts
            (training, testing, etc).
        """
        ss.Net.InitExt()

        lays = go.Slice_string(["Location", "Cover", "Toy", "Reach"])
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

        rch = leabra.Layer(ss.Net.LayerByName("Reach"))
        if "choice" in ss.TrainEnv.TrialName.Cur:
            rch.SetType(emer.Compare)
        else:
            rch.SetType(emer.Input)

        if ss.TrainEnv.GroupName.Cur != ss.PrvGpName:  # init at start of new group
            ss.Net.InitActs()
            ss.PrvGpName = ss.TrainEnv.GroupName.Cur
        train = True
        if "delay" in ss.TrainEnv.TrialName.Cur:
            train = False  # don't learn on delay trials

        ss.ApplyInputs(ss.TrainEnv)
        ss.AlphaCyc(train)
        ss.TrialStats(True)  # accumulate
        ss.LogTrnTrl(ss.TrnTrlLog, ss.TrainEnv.Trial.Cur, ss.TrainEnv.TrialName.Cur)

    def RunEnd(ss):
        """
        RunEnd is called at the end of a run -- save weights, record final log, etc here
        """

    def NewRun(ss):
        """
        NewRun intializes a new run of the model, using the TrainEnv.Run counter
        for the new run value
        """
        run = ss.TrainEnv.Run.Cur
        ss.TrainEnv.Init(run)
        ss.Time.Reset()
        ss.InitWts(ss.Net)
        ss.InitStats()
        ss.TrnTrlLog.SetNumRows(len(ss.TrainEnv.Order))
        ss.PrvGpName = ""
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
        SaveWeights saves the network weights -- when called with giv.CallMethod
        it will auto-prompt for filename
        """
        ss.Net.SaveWtsJSON(filename)

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
            hid = leabra.Layer(ss.Net.LayerByName("Hidden"))
            fmhid = leabra.Prjn(hid.RcvPrjns.SendName("Hidden"))
            fmhid.WtInit.Mean = ss.RecurrentWt
        if sheet == "" or sheet == "Sim":
            if "Sim" in pset.Sheets:
                simp = pset.SheetByNameTry("Sim")
                pyparams.ApplyParams(ss, simp, setMsg)
                simp.Apply(ss, setMsg)

    def OpenPat(ss, dt, fname, name, desc):
        dt.OpenCSV(fname, etable.Tab)
        dt.SetMetaData("name", name)
        dt.SetMetaData("desc", desc)

    def OpenPats(ss):
        ss.OpenPat(
            ss.Delay3Pats, "a_not_b_delay3.tsv", "AnotB Delay=3", "AnotB input patterns"
        )
        ss.OpenPat(
            ss.Delay5Pats, "a_not_b_delay5.tsv", "AnotB Delay=5", "AnotB input patterns"
        )
        ss.OpenPat(
            ss.Delay1Pats, "a_not_b_delay1.tsv", "AnotB Delay=1", "AnotB input patterns"
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

    def LogTrnTrl(ss, dt, trl, trlnm):
        """
        LogTrnTrl adds data from current trial to the TrnTrlLog table.
        log always contains number of testing items
        """
        row = trl
        if dt.Rows <= row:
            dt.SetNumRows(row + 1)

        dt.SetCellFloat("Trial", row, float(trl))
        dt.SetCellString("Group", row, ss.TrainEnv.GroupName.Cur)
        dt.SetCellString("TrialName", row, trlnm)

        for lnm in ss.TstRecLays:
            tsr = ss.ValuesTsr(lnm)
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            ly.UnitValuesTensor(tsr, "ActM")
            dt.SetCellTensor(lnm, row, tsr)

        ss.TrnTrlPlot.GoUpdate()
        ss.TrnTrlTable.UpdateSig()

    def ConfigTrnTrlLog(ss, dt):
        dt.SetMetaData("name", "TrnTrlLog")
        dt.SetMetaData("desc", "Record of testing per input pattern")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        nt = ss.TrainEnv.Table.Len()
        sch = etable.Schema(
            [
                etable.Column("Trial", etensor.INT64, go.nil, go.nil),
                etable.Column("Group", etensor.STRING, go.nil, go.nil),
                etable.Column("TrialName", etensor.STRING, go.nil, go.nil),
            ]
        )
        for lnm in ss.TstRecLays:
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            sch.append(etable.Column(lnm, etensor.FLOAT64, ly.Shp.Shp, go.nil))
        dt.SetFromSchema(sch, nt)

    def ConfigTrnTrlPlot(ss, plt, dt):
        plt.Params.Title = "A not B Train Trial Plot"
        plt.Params.XAxisCol = "TrialName"
        plt.Params.Type = eplot.Bar
        plt.SetTable(dt)
        plt.Params.XAxisRot = 90

        plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Group", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

        for lnm in ss.TstRecLays:
            if lnm == "Reach":
                cp = plt.SetColParams(lnm, eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
                cp.TensorIndex = -1
            else:
                cp = plt.SetColParams(lnm, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
                cp.TensorIndex = -1
        return plt

    def ConfigNetView(ss, nv):
        nv.ViewDefaults()
        nv.Scene().Camera.Pose.Pos.Set(0.1, 1.8, 3.5)
        nv.Scene().Camera.LookAt(mat32.Vec3(0.1, 0.15, 0), mat32.Vec3(0, 1, 0))

    def ConfigGui(ss):
        """
        ConfigGui configures the GoGi gui interface for this simulation,
        """
        width = 1600
        height = 1200

        gi.SetAppName("a_not_b")
        gi.SetAppAbout(
            'explores how the development of PFC active maintenance abilities can help to make behavior more flexible, in the sense that it can rapidly shift with changes in the environment. The development of flexibility has been extensively explored in the context of Piagets famous A-not-B task, where a toy is first hidden several times in one hiding location (A), and then hidden in a new location (B). Depending on various task parameters, young kids reliably reach back at A instead of updating to B. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch10/a_not_b/README.md">README.md on GitHub</a>.</p>'
        )

        win = gi.NewMainWindow("a_not_b", "A not B", width, height)
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

        tabv = etview.TableView()
        tv.AddTab(tabv, "TrnTrlTable")
        tabv.SetTable(ss.TrnTrlLog, go.nil)
        ss.TrnTrlTable = tabv

        plt = eplot.Plot2D()
        tv.AddTab(plt, "TrnTrlPlot")
        ss.TrnTrlPlot = ss.ConfigTrnTrlPlot(plt, ss.TrnTrlLog)

        split.SetSplitsList(go.Slice_float32([0.2, 0.8]))
        recv = win.This()

        tbar.AddAction(
            gi.ActOpts(
                Label="Init",
                Icon="update",
                Tooltip="Initialize everything including network weights, and start over.  Also applies current params.",
                UpdateFunc=UpdateFuncNotRunning,
            ),
            recv,
            InitCB,
        )

        tbar.AddAction(
            gi.ActOpts(
                Label="Train",
                Icon="run",
                Tooltip="Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.",
                UpdateFunc=UpdateFuncNotRunning,
            ),
            recv,
            TrainCB,
        )

        tbar.AddAction(
            gi.ActOpts(
                Label="Stop",
                Icon="stop",
                Tooltip="Interrupts running.  Hitting Train again will pick back up where it left off.",
                UpdateFunc=UpdateFuncRunning,
            ),
            recv,
            StopCB,
        )

        tbar.AddAction(
            gi.ActOpts(
                Label="Step Trial",
                Icon="step-fwd",
                Tooltip="Advances one training trial at a time.",
                UpdateFunc=UpdateFuncNotRunning,
            ),
            recv,
            StepTrialCB,
        )

        tbar.AddAction(
            gi.ActOpts(
                Label="Step Epoch",
                Icon="fast-fwd",
                Tooltip="Advances one epoch (complete set of training patterns) at a time.",
                UpdateFunc=UpdateFuncNotRunning,
            ),
            recv,
            StepEpochCB,
        )

        tbar.AddAction(
            gi.ActOpts(
                Label="Step Run",
                Icon="fast-fwd",
                Tooltip="Advances one full training Run at a time.",
                UpdateFunc=UpdateFuncNotRunning,
            ),
            recv,
            StepRunCB,
        )

        tbar.AddSeparator("misc")

        tbar.AddAction(
            gi.ActOpts(
                Label="New Seed",
                Icon="new",
                Tooltip="Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
            ),
            recv,
            NewRndSeedCB,
        )

        tbar.AddAction(
            gi.ActOpts(
                Label="Defaults",
                Icon="update",
                Tooltip="Restore initial default parameters.",
                UpdateFunc=UpdateFuncNotRunning,
            ),
            recv,
            DefaultsCB,
        )

        tbar.AddAction(
            gi.ActOpts(
                Label="README",
                Icon="file-markdown",
                Tooltip="Opens your browser on the README file that contains instructions for how to run this model.",
            ),
            recv,
            ReadmeCB,
        )

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
