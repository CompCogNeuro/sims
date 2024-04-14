#!/usr/local/bin/pyleabra -i

# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# use:
# just type file name to run, or:
# pyleabra -i <file>.py

# detector: This simulation shows how an individual neuron can act like a
# detector, picking out specific patterns from its inputs and responding
# with varying degrees of selectivity to the match between its synaptic
# weights and the input activity pattern.

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
)

import importlib as il
import io, sys, getopt
from datetime import datetime, timezone

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


def StopCB(recv, send, sig, data):
    TheSim.Stop()


def TestTrialCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.TestTrial()
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
    idxs = TheSim.TestEnv.Table.RowsByString(
        "Name", val, True, True
    )  # contains, ignoreCase
    if len(idxs) == 0:
        gi.PromptDialog(
            vp,
            gi.DlgOpts(
                Title="Name Not Found", Prompt="No patterns found containing: " + val
            ),
            True,
            False,
            go.nil,
            go.nil,
        )
    else:
        if not TheSim.IsRunning:
            TheSim.IsRunning = True
            print("testing index: %s" % idxs[0])
            TheSim.TestItem(idxs[0])
            TheSim.IsRunning = False
            vp.SetNeedsFullRender()


def TestItemCB(recv, send, sig, data):
    win = gi.Window(handle=recv)
    gi.StringPromptDialog(
        win.WinViewport2D(),
        "",
        "Test Item",
        gi.DlgOpts(
            Title="Test Item",
            Prompt="Enter the Name of a given input pattern to test (case insensitive, contains given string.",
        ),
        win,
        TestItemCB2,
    )


def TestAllCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.ToolBar.UpdateActions()
        TheSim.RunTestAll()


def DefaultsCB(recv, send, sig, data):
    TheSim.Defaults()
    TheSim.Init()
    TheSim.UpdateClassView()
    TheSim.vp.SetNeedsFullRender()


def ReadmeCB(recv, send, sig, data):
    gi.OpenURL(
        "https://github.com/CompCogNeuro/sims/blob/master/ch2/detector/README.md"
    )


def UpdateFuncNotRunning(act):
    act.SetActiveStateUpdate(not TheSim.IsRunning)


def UpdateFuncRunning(act):
    act.SetActiveStateUpdate(TheSim.IsRunning)


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
        self.GbarL = float(2.0)
        self.SetTags(
            "GbarL",
            'def:"2" min:"0" max:"4" step:"0.05" desc:"the leak conductance, which pulls against the excitatory input conductance to determine how hard it is to activate the receiving unit"',
        )
        self.Net = leabra.Network()
        self.SetTags(
            "Net",
            'view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"',
        )
        self.Pats = etable.Table()
        self.SetTags(
            "Pats",
            'view:"no-inline" desc:"click to see the testing input patterns to use (digits)"',
        )
        self.TstTrlLog = etable.Table()
        self.SetTags(
            "TstTrlLog",
            'view:"no-inline" desc:"testing trial-level log data -- click to see record of network\'s response to each input"',
        )
        self.Params = params.Sets()
        self.ParamSet = str()
        self.SetTags(
            "ParamSet",
            'view:"-" desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set -- can use multiple names separated by spaces (don\'t put spaces in ParamSet names!)"',
        )
        self.SetTags(
            "Params",
            'view:"no-inline" desc:"full collection of param sets -- not really interesting for this model"',
        )
        self.TestEnv = env.FixedTable()
        self.SetTags(
            "TestEnv", 'desc:"Testing environment -- manages iterating over testing"'
        )
        self.Time = leabra.Time()
        self.SetTags("Time", 'desc:"leabra timing parameters and state"')
        self.ViewUpdate = leabra.TimeScales.Cycle
        self.SetTags(
            "ViewUpdate",
            'desc:"at what time scale to update the display during testing?  Change to AlphaCyc to make display updating go faster"',
        )

        # internal state - view:"-"
        self.Win = 0
        self.SetTags("Win", 'view:"-" desc:"main GUI window"')
        self.NetView = 0
        self.SetTags("NetView", 'view:"-" desc:"the network viewer"')
        self.ToolBar = 0
        self.SetTags("ToolBar", 'view:"-" desc:"the master toolbar"')
        self.TstTrlPlot = 0
        self.SetTags("TstTrlPlot", 'view:"-" desc:"the test-trial plot"')
        self.ValuesTsrs = {}
        self.SetTags("ValuesTsrs", 'view:"-" desc:"for holding layer values"')
        self.IsRunning = False
        self.SetTags("IsRunning", 'view:"-" desc:"true if sim is running"')
        self.StopNow = False
        self.SetTags("StopNow", 'view:"-" desc:"flag to stop running"')
        self.vp = 0
        self.SetTags("vp", 'view:"-" desc:"viewport"')

    def InitParams(ss):
        """
        Sets the default set of parameters -- Base is always applied, and others can be optionally
        selected to apply on top of that
        """
        ss.Params.OpenJSON("detector.params")

    def Defaults(ss):
        """
        Defaults sets default params
        """
        ss.GbarL = float(2.0)

    def Config(ss):
        """
        Config configures all the elements using the standard functions
        """
        ss.Defaults()
        ss.InitParams()
        ss.OpenPats()
        ss.ConfigEnv()
        ss.ConfigNet(ss.Net)
        ss.ConfigTstTrlLog(ss.TstTrlLog)

    def ConfigEnv(ss):
        ss.TestEnv.Nm = "TestEnv"
        ss.TestEnv.Dsc = "testing params and state"
        ss.TestEnv.Table = etable.NewIndexView(ss.Pats)
        ss.TestEnv.Sequential = True
        ss.TestEnv.Validate()
        ss.TestEnv.Init(0)

    def ConfigNet(ss, net):
        net.InitName(net, "Detector")
        inp = net.AddLayer2D("Input", 7, 5, emer.Input)
        recv = net.AddLayer2D("RecvNeuron", 1, 1, emer.Hidden)

        net.ConnectLayers(inp, recv, prjn.NewFull(), emer.Forward)

        net.Defaults()
        ss.SetParams("Network", False)
        net.Build()
        ss.InitWts(net)

    def InitWts(ss, net):
        """
        InitWts initializes weights to digit 8
        """
        net.InitWts()
        digit = 8
        pats = ss.Pats
        dpat = pats.CellTensor("Input", digit)
        recv = net.LayerByName("RecvNeuron")
        prj = recv.RecvPrjns().SendName("Input")
        for i in range(dpat.Len()):
            prj.SetSynValue("Wt", i, 0, float(dpat.FloatValue1D(i)))

    def Init(ss):
        """
        Init restarts the run, and initializes everything, including network weights
        and resets the epoch log table
        """
        ss.ConfigEnv()

        ss.Time.Reset()
        ss.Time.CycPerQtr = 5  # don't need much time
        ss.InitWts(ss.Net)
        ss.StopNow = False
        ss.SetParams("", False)  # all sheets
        ss.UpdateView()

    def Counters(ss):
        """
        Counters returns a string of the current counter state
        use tabs to achieve a reasonable formatting overall
        and add a few tabs at the end to allow for expansion..
        """
        return "Trial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t" % (
            ss.TestEnv.Trial.Cur,
            ss.Time.Cycle,
            ss.TestEnv.TrialName.Cur,
        )

    def UpdateView(ss):
        if ss.NetView != 0 and ss.NetView.IsVisible():
            ss.NetView.Record(ss.Counters())

            ss.NetView.GoUpdate()  # note: using counters is significantly slower..

    def AlphaCyc(ss):
        """
        AlphaCyc runs one alpha-cycle (100 msec, 4 quarters) of processing.
        External inputs must have already been applied prior to calling,
        using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
        If train is true, then learning DWt or WtFmDWt calls are made.
        Handles netview updating within scope of AlphaCycle
        """

        if ss.Win != 0:
            ss.Win.PollEvents()  # this is essential for GUI responsiveness while running
        viewUpdate = ss.ViewUpdate.value

        ss.Net.AlphaCycInit(False)
        ss.Time.AlphaCycStart()
        for qtr in range(4):
            for cyc in range(ss.Time.CycPerQtr):
                ss.Net.Cycle(ss.Time)
                ss.Time.CycleInc()
                if viewUpdate == leabra.Cycle:
                    if cyc != ss.Time.CycPerQtr - 1:  # will be updated by quarter
                        ss.UpdateView()
                if viewUpdate == leabra.FastSpike:
                    if (cyc + 1) % 10 == 0:
                        ss.UpdateView()
            ss.Net.QuarterFinal(ss.Time)
            ss.Time.QuarterInc()
            if viewUpdate <= leabra.Quarter:
                ss.UpdateView()
            if viewUpdate == leabra.Phase:
                if qtr >= 2:
                    ss.UpdateView()

        if viewUpdate == leabra.AlphaCycle:
            ss.UpdateView()

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
                ly.ApplyExt(pats)

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

    def TestTrial(ss):
        """
        TestTrial runs one trial of testing -- always sequentially presented inputs
        """
        ss.TestEnv.Step()

        chg = env.CounterChg(ss.TestEnv, env.Epoch)
        if chg:
            if ss.ViewUpdate.value > leabra.AlphaCycle:
                ss.UpdateView()
            return

        ss.ApplyInputs(ss.TestEnv)
        ss.AlphaCyc()
        ss.LogTstTrl(ss.TstTrlLog)

    def TestItem(ss, idx):
        """
        TestItem tests given item which is at given index in test item list
        """
        cur = ss.TestEnv.Trial.Cur
        ss.TestEnv.Trial.Cur = idx
        ss.TestEnv.SetTrialName()
        ss.ApplyInputs(ss.TestEnv)
        ss.AlphaCyc()
        ss.TestEnv.Trial.Cur = cur

    def TestAll(ss):
        """
        TestAll runs through the full set of testing items
        """
        ss.TestEnv.Init(0)
        while True:
            ss.TestTrial()
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
        recv = leabra.Layer(ss.Net.LayerByName("RecvNeuron"))
        recv.Act.Gbar.L = ss.GbarL

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

    def OpenPats(ss):
        ss.Pats.SetMetaData("name", "DigitPats")
        ss.Pats.SetMetaData("desc", "Testing Digit patterns")
        ss.Pats.OpenCSV("digits.tsv", etable.Tab)

    def LogTstTrl(ss, dt):
        """
        LogTstTrl adds data from current trial to the TstTrlLog table.
        log always contains number of testing items
        """
        inp = leabra.Layer(ss.Net.LayerByName("Input"))
        recv = leabra.Layer(ss.Net.LayerByName("RecvNeuron"))

        trl = ss.TestEnv.Trial.Cur
        row = trl

        if dt.Rows <= row:
            dt.SetNumRows(row + 1)
        dt.SetCellFloat("Trial", row, float(trl))
        dt.SetCellString("TrialName", row, ss.TestEnv.TrialName.Cur)

        ivt = ss.ValuesTsr("Input")
        ovt = ss.ValuesTsr("Output")
        inp.UnitValuesTensor(ivt, "Act")
        dt.SetCellTensor("Input", row, ivt)
        recv.UnitValuesTensor(ovt, "Ge")
        dt.SetCellTensor("Ge", row, ovt)
        recv.UnitValuesTensor(ovt, "Act")
        dt.SetCellTensor("Act", row, ovt)

        ss.TstTrlPlot.GoUpdate()

    def ConfigTstTrlLog(ss, dt):
        inp = leabra.Layer(ss.Net.LayerByName("Input"))
        recv = leabra.Layer(ss.Net.LayerByName("RecvNeuron"))

        dt.SetMetaData("name", "TstTrlLog")
        dt.SetMetaData("desc", "Record of testing per input pattern")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        nt = ss.TestEnv.Table.Len()
        sch = etable.Schema(
            [
                etable.Column("Trial", etensor.INT64, go.nil, go.nil),
                etable.Column("TrialName", etensor.STRING, go.nil, go.nil),
                etable.Column("Input", etensor.FLOAT64, inp.Shp.Shp, go.nil),
                etable.Column("Ge", etensor.FLOAT64, recv.Shp.Shp, go.nil),
                etable.Column("Act", etensor.FLOAT64, recv.Shp.Shp, go.nil),
            ]
        )
        dt.SetFromSchema(sch, nt)

    def ConfigTstTrlPlot(ss, plt, dt):
        plt.Params.Title = "Detector Test Trial Plot"
        plt.Params.XAxisCol = "Trial"
        plt.SetTable(dt)

        plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

        cp = plt.SetColParams("Input", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        cp.TensorIndex = -1  # plot all
        plt.SetColParams("Ge", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("Act", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        return plt

    def ConfigGui(ss):
        """
        ConfigGui configures the GoGi gui interface for this simulation,
        """
        width = 1600
        height = 1200

        gi.SetAppName("detector")
        gi.SetAppAbout(
            'This simulation shows how an individual neuron can act like a detector, picking out specific patterns from its inputs and responding with varying degrees of selectivity to the match between its synaptic weights and the input activity pattern. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch2/detector/README.md">README.md on GitHub</a>.</p>'
        )

        win = gi.NewMainWindow("detector", "Neuron as Detector", width, height)
        ss.Win = win

        vp = win.WinViewport2D()
        ss.vp = vp
        updt = vp.UpdateStart()

        mfr = win.SetMainFrame()

        tbar = gi.AddNewToolBar(mfr, "tbar")
        tbar.SetStretchMaxWidth()
        ss.ToolBar = tbar

        split = gi.AddNewSplitView(mfr, "split")
        split.Dim = math32.X
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

        nv.ViewDefaults()

        plt = eplot.Plot2D()
        tv.AddTab(plt, "TstTrlPlot")
        ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

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
                Label="Test Trial",
                Icon="step-fwd",
                Tooltip="Runs the next testing trial.",
                UpdateFunc=UpdateFuncNotRunning,
            ),
            recv,
            TestTrialCB,
        )

        tbar.AddAction(
            gi.ActOpts(
                Label="Test Item",
                Icon="step-fwd",
                Tooltip="Prompts for a specific input pattern name to run, and runs it in testing mode.",
                UpdateFunc=UpdateFuncNotRunning,
            ),
            recv,
            TestItemCB,
        )

        tbar.AddAction(
            gi.ActOpts(
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
