#!/usr/local/bin/pyleabra -i

# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# use:
# just type file name to run, or:
# pyleabra -i <file>.py 

# face_categ: This project explores how sensory inputs
# (in this case simple cartoon faces) can be categorized
# in multiple different ways, to extract the relevant information
# and collapse across the irrelevant. It allows you to explore both 
# bottom-up processing from face image to categories, and top-down 
# processing from category values to face images (imagery), 
# including the ability to dynamically iterate both bottom-up and
# top-down to cleanup partial inputs (partially occluded face images).

from leabra import go, leabra, emer, relpos, eplot, env, agg, patgen, prjn, etable, efile, split, etensor, params, netview, rand, erand, gi, giv, pygiv, pyparams, mat32, simat, metric, clust

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

def SetInputCB2(recv, send, sig, data):
    if sig == 0:
        TheSim.SetInput(False)
    else:
        TheSim.SetInput(True)

def SetInputCB(recv, send, sig, data):
    win = gi.Window(handle=recv)
    gi.ChoiceDialog(win.WinViewport2D(), gi.DlgOpts(Title="Set Input", Prompt="Set whether the input to the network comes in bottom-up (Input layer) or top-down (Higher-level category layers)"), go.Slice_string(["Bottom-up", "Top-down"]), win, SetInputCB2)

def SetPatsCB2(recv, send, sig, data):
    if sig == 0:
        TheSim.SetPats(False)
    else:
        TheSim.SetPats(True)

def SetPatsCB(recv, send, sig, data):
    win = gi.Window(handle=recv)
    gi.ChoiceDialog(win.WinViewport2D(), gi.DlgOpts(Title="Set Pats", Prompt="Set which set of patterns to present -- full or partial faces"), go.Slice_string(["Full Faces", "Partial Faces"]), win, SetPatsCB2)

def ClusterPlotCB(recv, send, sig, data):
    TheSim.ClusterPlots()
    TheSim.vp.SetNeedsFullRender()
    
def DefaultsCB(recv, send, sig, data):
    TheSim.Defaults()
    TheSim.Init()
    TheSim.UpdateClassView()
    TheSim.vp.SetNeedsFullRender()

def ReadmeCB(recv, send, sig, data):
    gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch3/face_categ/README.md")

def UpdtFuncNotRunning(act):
    act.SetActiveStateUpdt(not TheSim.IsRunning)
    
def UpdtFuncRunning(act):
    act.SetActiveStateUpdt(TheSim.IsRunning)

    
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
        self.Pats = etable.Table()
        self.SetTags("Pats", 'view:"no-inline" desc:"click to see the full face testing input patterns to use"')
        self.PartPats = etable.Table()
        self.SetTags("PartPats", 'view:"no-inline" desc:"click to see the partial face testing input patterns to use"')
        self.TstTrlLog = etable.Table()
        self.SetTags("TstTrlLog", 'view:"no-inline" desc:"testing trial-level log data -- click to see record of network\'s response to each input"')
        self.PrjnTable = etable.Table()
        self.SetTags("PrjnTable", 'view:"no-inline" desc:"projection of testing data"')
        self.Params = params.Sets()
        self.SetTags("Params", 'view:"no-inline" desc:"full collection of param sets -- not really interesting for this model"')
        self.ParamSet = str()
        self.SetTags("ParamSet", 'view:"-" desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set -- can use multiple names separated by spaces (don\'t put spaces in ParamSet names!)"')
        self.TestEnv = env.FixedTable()
        self.SetTags("TestEnv", 'desc:"Testing environment -- manages iterating over testing"')
        self.Time = leabra.Time()
        self.SetTags("Time", 'desc:"leabra timing parameters and state"')
        self.ViewUpdt = leabra.TimeScales.Cycle
        self.SetTags("ViewUpdt", 'desc:"at what time scale to update the display during testing?  Change to AlphaCyc to make display updating go faster"')
        self.TstRecLays = go.Slice_string(["Input", "Emotion", "Gender", "Identity"])
        self.SetTags("TstRecLays", 'desc:"names of layers to record activations etc of during testing"')
        self.ClustFaces = eplot.Plot2D()
        self.SetTags("ClustFaces", 'view:"no-inline" desc:"cluster plot of faces"')
        self.ClustEmote = eplot.Plot2D()
        self.SetTags("ClustEmote", 'view:"no-inline" desc:"cluster plot of emotions"')
        self.ClustGend = eplot.Plot2D()
        self.SetTags("ClustGend", 'view:"no-inline" desc:"cluster plot of genders"')
        self.ClustIdent = eplot.Plot2D()
        self.SetTags("ClustIdent", 'view:"no-inline" desc:"cluster plot of identity"')
        self.PrjnRandom = eplot.Plot2D()
        self.SetTags("PrjnRandom", 'view:"no-inline" desc:"random projection plot"')
        self.PrjnEmoteGend = eplot.Plot2D()
        self.SetTags("PrjnEmoteGend", 'view:"no-inline" desc:"projection plot of emotions & gender"')

        # internal state - view:"-"
        self.Win = 0
        self.SetTags("Win", 'view:"-" desc:"main GUI window"')
        self.NetView = 0
        self.SetTags("NetView", 'view:"-" desc:"the network viewer"')
        self.ToolBar = 0
        self.SetTags("ToolBar", 'view:"-" desc:"the master toolbar"')
        self.TstTrlPlot = 0
        self.SetTags("TstTrlPlot", 'view:"-" desc:"the test-trial plot"')
        self.ValsTsrs = {}
        self.SetTags("ValsTsrs", 'view:"-" desc:"for holding layer values"')
        self.IsRunning = False
        self.SetTags("IsRunning", 'view:"-" desc:"true if sim is running"')
        self.StopNow = False
        self.SetTags("StopNow", 'view:"-" desc:"flag to stop running"')
        self.vp  = 0 
        self.SetTags("vp", 'view:"-" desc:"viewport"')
       
    def InitParams(ss):
        """
        Sets the default set of parameters -- Base is always applied, and others can be optionally
        selected to apply on top of that
        """
        ss.Params.OpenJSON("face_categ.params")

    def Defaults(ss):
        """
        Defaults sets default params
        """

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
        ss.TestEnv.Table = etable.NewIdxView(ss.Pats)
        ss.TestEnv.Sequential = True
        ss.TestEnv.Validate()
        ss.TestEnv.Init(0)

    def ConfigNet(ss, net):
        net.InitName(net, "FaceCateg")
        inp = net.AddLayer2D("Input", 16, 16, emer.Input)
        emo = net.AddLayer2D("Emotion", 1, 2, emer.Compare)
        gend = net.AddLayer2D("Gender", 1, 2, emer.Compare)
        iden = net.AddLayer2D("Identity", 1, 10, emer.Compare)

        full = prjn.NewFull()
    
        net.BidirConnectLayersPy(inp, emo, full)
        net.BidirConnectLayersPy(inp, gend, full)
        net.BidirConnectLayersPy(inp, iden, full)

        emo.SetRelPos(relpos.Rel(Rel= relpos.Above, Other= "Input", YAlign= relpos.Front, XAlign= relpos.Left, Space= 2))
        gend.SetRelPos(relpos.Rel(Rel= relpos.Above, Other= "Input", YAlign= relpos.Front, XAlign= relpos.Right, Space= 2))
        iden.SetRelPos(relpos.Rel(Rel= relpos.Above, Other= "Input", YAlign= relpos.Center, XAlign= relpos.Left, Space= 2))

        net.Defaults()
        ss.SetParams("Network", False)
        net.Build()
        ss.InitWts(net)

    def InitWts(ss, net):
        """
        InitWts loads the saved weights
        """
        net.InitWts()
        net.OpenWtsJSON("faces.wts")

    def Init(ss):
        """
        Init restarts the run, and initializes everything, including network weights
        and resets the epoch log table
        """

        ss.TestEnv.Init(0)
        ss.Time.Reset()
        ss.Time.CycPerQtr = 10 # don't need much time
        ss.InitWts(ss.Net)
        ss.StopNow = False
        ss.SetParams("", False) # all sheets
        ss.UpdateView()

    def Counters(ss):
        """
        Counters returns a string of the current counter state
        use tabs to achieve a reasonable formatting overall
        and add a few tabs at the end to allow for expansion..
        """
        return "Trial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t" % (ss.TestEnv.Trial.Cur, ss.Time.Cycle, ss.TestEnv.TrialName.Cur)

    def UpdateView(ss):
        if ss.NetView != 0 and ss.NetView.IsVisible():
            ss.NetView.Record(ss.Counters())

            ss.NetView.GoUpdate() # note: using counters is significantly slower..


    def AlphaCyc(ss):
        """
        AlphaCyc runs one alpha-cycle (100 msec, 4 quarters) of processing.
        External inputs must have already been applied prior to calling,
        using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
        If train is true, then learning DWt or WtFmDWt calls are made.
        Handles netview updating within scope of AlphaCycle
        """

        if ss.Win != 0:
            ss.Win.PollEvents() # this is essential for GUI responsiveness while running
        viewUpdt = ss.ViewUpdt.value

        ss.Net.AlphaCycInit(False)
        ss.Time.AlphaCycStart()
        for qtr in range(4):
            for cyc in range(ss.Time.CycPerQtr):
                ss.Net.Cycle(ss.Time)
                ss.Time.CycleInc()
                if viewUpdt == leabra.Cycle:
                    if cyc != ss.Time.CycPerQtr-1: # will be updated by quarter
                        ss.UpdateView()
                if viewUpdt == leabra.FastSpike:
                    if (cyc+1)%10 == 0:
                        ss.UpdateView()
            ss.Net.QuarterFinal(ss.Time)
            ss.Time.QuarterInc()
            if viewUpdt <= leabra.Quarter:
                ss.UpdateView()
            if viewUpdt == leabra.Phase:
                if qtr >= 2:
                    ss.UpdateView()

        if viewUpdt == leabra.AlphaCycle:
            ss.UpdateView()

    def ApplyInputs(ss, en):
        """
        ApplyInputs applies input patterns from given envirbonment.
        It is good practice to have this be a separate method with appropriate

        # going to the same layers, but good practice and cheap anyway
        args so that it can be used for various different contexts
        (training, testing, etc).
        """
        ss.Net.InitExt()

        lays = go.Slice_string(["Input", "Emotion", "Gender", "Identity"])
        for lnm in lays :
            ly = leabra.LeabraLayer(ss.Net.LayerByName(lnm)).AsLeabra()
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
            if ss.ViewUpdt.value > leabra.AlphaCycle:
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

    def SetInput(ss, topDown):
        """
        SetInput sets whether the input to the network comes in bottom-up
        (Input layer) or top-down (Higher-level category layers)
        """
        inp =  leabra.Layer(ss.Net.LayerByName("Input"))
        emo =  leabra.Layer(ss.Net.LayerByName("Emotion"))
        gend = leabra.Layer(ss.Net.LayerByName("Gender"))
        iden = leabra.Layer(ss.Net.LayerByName("Identity"))
        if topDown:
            inp.SetType(emer.Compare)
            emo.SetType(emer.Input)
            gend.SetType(emer.Input)
            iden.SetType(emer.Input)
        else:
            inp.SetType(emer.Input)
            emo.SetType(emer.Compare)
            gend.SetType(emer.Compare)
            iden.SetType(emer.Compare)

    def SetPats(ss, partial):
        """
        SetPats selects which patterns to present: full or partial faces
        """
        if partial:
            ss.TestEnv.Table = etable.NewIdxView(ss.PartPats)
            ss.TestEnv.Validate()
            ss.TestEnv.Init(0)
        else:
            ss.TestEnv.Table = etable.NewIdxView(ss.Pats)
            ss.TestEnv.Validate()
            ss.TestEnv.Init(0)

    def OpenPats(ss):
        ss.Pats.SetMetaData("name", "FacePats")
        ss.Pats.SetMetaData("desc", "Testing Face patterns: full faces")
        ss.Pats.OpenCSV("faces.tsv", etable.Tab)

        ss.Pats.SetMetaData("name", "PartFacePats")
        ss.Pats.SetMetaData("desc", "Testing Face patterns: partial faces")
        ss.PartPats.OpenCSV("partial_faces.tsv", etable.Tab)

    def ClusterPlots(ss):
        """
        ClusterPlots computes all the cluster plots from the faces input data
        """
        ss.ClustPlot(ss.ClustFaces, ss.Pats, "Input")
        ss.ClustPlot(ss.ClustEmote, ss.Pats, "Emotion")
        ss.ClustPlot(ss.ClustGend, ss.Pats, "Gender")
        ss.ClustPlot(ss.ClustIdent, ss.Pats, "Identity")

        ss.PrjnPlot()

    def ClustPlot(ss, plt, dt, colNm):
        """
        ClustPlot does one cluster plot on given table column
        """
        ix = etable.NewIdxView(dt)
        smat = simat.SimMat()
        smat.TableColStd(ix, colNm, "Name", False, metric.Euclidean)
        pt = etable.Table()
        clust.Plot(pt, clust.GlomStd(smat, clust.Min), smat)
        plt.InitName(plt, colNm)
        plt.Params.Title = "Cluster Plot of Faces " + colNm
        plt.Params.XAxisCol = "X"
        plt.SetTable(pt)

        plt.SetColParams("X", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Y", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Label", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)

    def PrjnPlot(ss):
        ss.TestAll()

        rvec0 = ss.ValsTsr("rvec0")
        rvec1 = ss.ValsTsr("rvec1")
        rvec0.SetShape(go.Slice_int([256]), go.nil, go.nil)
        rvec1.SetShape(go.Slice_int([256]), go.nil, go.nil)
        for i in range(256):
            rvec0.Values[i] = .15 * (2*rand.Float32() - 1)
            rvec1.Values[i] = .15 * (2*rand.Float32() - 1)

        tst = ss.TstTrlLog
        nr = tst.Rows
        dt = ss.PrjnTable
        ss.ConfigPrjnTable(dt)

        for r in range(nr):
            emote = 0.5*tst.CellTensorFloat1D("Emotion", r, 0) + -0.5*tst.CellTensorFloat1D("Emotion", r, 1)
            emote += .1 * (2*rand.Float64() - 1)

            gend = 0.5*tst.CellTensorFloat1D("Gender", r, 0) + -0.5*tst.CellTensorFloat1D("Gender", r, 1)
            gend += .1 * (2*rand.Float64() - 1) # some jitter so labels are readable
            input = etensor.Float32(tst.CellTensor("Input", r))
            rprjn0 = metric.InnerProduct32(rvec0.Values, input.Values)
            rprjn1 = metric.InnerProduct32(rvec1.Values, input.Values)
            dt.SetCellFloat("Trial", r, tst.CellFloat("Trial", r))
            dt.SetCellString("TrialName", r, tst.CellString("TrialName", r))
            dt.SetCellFloat("GendPrjn", r, gend)
            dt.SetCellFloat("EmotePrjn", r, emote)
            dt.SetCellFloat("RndPrjn0", r, float(rprjn0))
            dt.SetCellFloat("RndPrjn1", r, float(rprjn1))

        plt = ss.PrjnRandom
        plt.InitName(plt, "PrjnRandom")
        plt.Params.Title = "Face Random Prjn Plot"
        plt.Params.XAxisCol = "RndPrjn0"
        plt.SetTable(dt)
        plt.Params.Lines = False
        plt.Params.Points = True
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("TrialName", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("RndPrjn0", eplot.Off, eplot.FixMin, -1, eplot.FixMax, 1)
        plt.SetColParams("RndPrjn1", eplot.On, eplot.FixMin, -1, eplot.FixMax, 1)

        plt = ss.PrjnEmoteGend
        plt.InitName(plt, "PrjnEmoteGend")
        plt.Params.Title = "Face Emotion / Gender Prjn Plot"
        plt.Params.XAxisCol = "GendPrjn"
        plt.SetTable(dt)
        plt.Params.Lines = False
        plt.Params.Points = True
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("TrialName", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("GendPrjn", eplot.Off, eplot.FixMin, -1, eplot.FixMax, 1)
        plt.SetColParams("EmotePrjn", eplot.On, eplot.FixMin, -1, eplot.FixMax, 1)

    def ConfigPrjnTable(ss, dt):
        dt.SetMetaData("name", "PrjnTable")
        dt.SetMetaData("desc", "projection of data onto dimension")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        nt = ss.TestEnv.Table.Len() # number in view
        sch = etable.Schema(
            [etable.Column("Trial", etensor.INT64, go.nil, go.nil),
            etable.Column("TrialName", etensor.STRING, go.nil, go.nil),
            etable.Column("GendPrjn", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("EmotePrjn", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("RndPrjn0", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("RndPrjn1", etensor.FLOAT64, go.nil, go.nil)]
        )
        dt.SetFromSchema(sch, nt)

    def LogTstTrl(ss, dt):
        """
        LogTstTrl adds data from current trial to the TstTrlLog table.
        log always contains number of testing items
        """
        trl = ss.TestEnv.Trial.Cur
        row = trl

        if dt.Rows <= row:
            dt.SetNumRows(row + 1)
        dt.SetCellFloat("Trial", row, float(trl))
        dt.SetCellString("TrialName", row, ss.TestEnv.TrialName.Cur)

        for lnm in ss.TstRecLays:
            tsr = ss.ValsTsr(lnm)
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            ly.UnitValsTensor(tsr, "Act")
            dt.SetCellTensor(lnm, row, tsr)

        ss.TstTrlPlot.GoUpdate()

    def ConfigTstTrlLog(ss, dt):
        dt.SetMetaData("name", "TstTrlLog")
        dt.SetMetaData("desc", "Record of testing per input pattern")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        nt = ss.TestEnv.Table.Len() # number in view
        sch = etable.Schema(
            [etable.Column("Trial", etensor.INT64, go.nil, go.nil),
            etable.Column("TrialName", etensor.STRING, go.nil, go.nil)]
        )
        for lnm in ss.TstRecLays:
            ly = leabra.LeabraLayer(ss.Net.LayerByName(lnm)).AsLeabra()
            sch.append(etable.Column(lnm, etensor.FLOAT32, ly.Shp.Shp, go.nil))
        dt.SetFromSchema(sch, nt)

    def ConfigTstTrlPlot(ss, plt, dt):
        plt.Params.Title = "FaceCateg Test Trial Plot"
        plt.Params.XAxisCol = "Trial"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("TrialName", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)

        for lnm in ss.TstRecLays:
            cp = plt.SetColParams(lnm, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
            cp.TensorIdx = -1 # plot all

        plt.SetColParams("Gender", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) # display
        return plt

    def ConfigNetView(ss, nv):
        labs = go.Slice_string(["happy sad", "female  male", "Albt Bett Lisa Mrk Wnd Zane"])
        nv.ConfigLabels(labs)
        emot = nv.LayerByName("Emotion")
        hs = nv.LabelByName(labs[0])
        hs.Pose = emot.Pose
        hs.Pose.Pos.Y += .1
        hs.Pose.Scale.SetMulScalar(0.5)

        gend = nv.LayerByName("Gender")
        fm = nv.LabelByName(labs[1])
        fm.Pose = gend.Pose
        fm.Pose.Pos.X -= .05
        fm.Pose.Pos.Y += .1
        fm.Pose.Scale.SetMulScalar(0.5)

        id = nv.LayerByName("Identity")
        nms = nv.LabelByName(labs[2])
        nms.Pose = id.Pose
        nms.Pose.Pos.Y += .1
        nms.Pose.Scale.SetMulScalar(0.5)

    def ConfigGui(ss):
        """
        ConfigGui configures the GoGi gui interface for this simulation,
        """
        width = 1600
        height = 1200

        gi.SetAppName("face_categ")
        gi.SetAppAbout('face_categ: This project explores how sensory inputs (in this case simple cartoon faces) can be categorized in multiple different ways, to extract the relevant information and collapse across the irrelevant. It allows you to explore both bottom-up processing from face image to categories, and top-down processing from category values to face images (imagery), including the ability to dynamically iterate both bottom-up and top-down to cleanup partial inputs (partially occluded face images).  See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch3/face_categ/README.md">README.md on GitHub</a>.</p>')

        win = gi.NewMainWindow("face_categ", "Face Categorization", width, height)
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
        nv.ViewDefaults()
        ss.ConfigNetView(nv) # add labels etc

        plt = eplot.Plot2D()
        tv.AddTab(plt, "TstTrlPlot")
        ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

        split.SetSplitsList(go.Slice_float32([.2, .8]))

        recv = win.This()
        
        tbar.AddAction(gi.ActOpts(Label="Init", Icon="update", Tooltip="Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc=UpdtFuncNotRunning), recv, InitCB)

        tbar.AddAction(gi.ActOpts(Label="Stop", Icon="stop", Tooltip="Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc=UpdtFuncRunning), recv, StopCB)
        
        tbar.AddAction(gi.ActOpts(Label="Test Trial", Icon="step-fwd", Tooltip="Runs the next testing trial.", UpdateFunc=UpdtFuncNotRunning), recv, TestTrialCB)
        
        tbar.AddAction(gi.ActOpts(Label="Test Item", Icon="step-fwd", Tooltip="Prompts for a specific input pattern name to run, and runs it in testing mode.", UpdateFunc=UpdtFuncNotRunning), recv, TestItemCB)
        
        tbar.AddAction(gi.ActOpts(Label="Test All", Icon="fast-fwd", Tooltip="Tests all of the testing trials.", UpdateFunc=UpdtFuncNotRunning), recv, TestAllCB)

        tbar.AddSeparator("log")
        
        tbar.AddAction(gi.ActOpts(Label= "SetInput", Icon= "gear", Tooltip= "set whether the input to the network comes in bottom-up (Input layer) or top-down (Higher-level category layers)", UpdateFunc= UpdtFuncNotRunning), recv, SetInputCB)
                
        tbar.AddAction(gi.ActOpts(Label= "SetPats", Icon= "gear", Tooltip= "set which set of patterns to present -- full or partial faces", UpdateFunc= UpdtFuncNotRunning), recv, SetPatsCB)
                
        tbar.AddAction(gi.ActOpts(Label= "Cluster Plots", Icon= "image", Tooltip= "generate cluster plots of the different layer patterns", UpdateFunc= UpdtFuncNotRunning), recv, ClusterPlotCB)
                
        tbar.AddSeparator("misc")
        
        tbar.AddAction(gi.ActOpts(Label= "Defaults", Icon= "update", Tooltip= "Restore initial default parameters.", UpdateFunc= UpdtFuncNotRunning), recv, DefaultsCB)

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


