#!/usr/local/bin/pyleabra -i

# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# use:
# just type file name to run, or:
# pyleabra -i <file>.py 

# cats_dogs: This project explores a simple **semantic network** intended
# to represent a (very small) set of relationships among different features
# used to represent a set of entities in the world.  In our case, we represent
# some features of cats and dogs: their color, size, favorite food, and favorite toy.

from leabra import go, leabra, emer, relpos, eplot, env, agg, patgen, prjn, etable, efile, split, etensor, params, netview, rand, erand, gi, giv, epygiv, mat32

import importlib as il
import io, sys, getopt
from datetime import datetime, timezone

# this will become Sim later.. 
TheSim = 1

# use this for e.g., etable.Column construction args where nil would be passed
nilInts = go.Slice_int()

# use this for e.g., etable.Column construction args where nil would be passed
nilStrs = go.Slice_string()

# LogPrec is precision for saving float values in logs
LogPrec = 4

# note: we cannot use methods for callbacks from Go -- must be separate functions
# so below are all the callbacks from the GUI toolbar actions

def InitCB(recv, send, sig, data):
    TheSim.Init()
    TheSim.ClassView.Update()
    TheSim.vp.SetNeedsFullRender()

def StopCB(recv, send, sig, data):
    TheSim.Stop()

def TestTrialCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.TestTrial()
        TheSim.IsRunning = False
        TheSim.ClassView.Update()
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

def DefaultsCB(recv, send, sig, data):
    TheSim.Defaults()
    TheSim.Init()
    TheSim.vp.SetNeedsFullRender()

def ReadmeCB(recv, send, sig, data):
    gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch3/cats_dogs/README.md")

def UpdtFuncNotRunning(act):
    act.SetActiveStateUpdt(not TheSim.IsRunning)
    
def UpdtFuncRunning(act):
    act.SetActiveStateUpdt(TheSim.IsRunning)

    
#####################################################    
#     Sim

class Sim(object):
    """
    Sim encapsulates the entire simulation model, and we define all the
    functionality as methods on this struct.  This structure keeps all relevant
    state information organized and available without having to pass everything around
    as arguments to methods, and provides the core GUI interface (note the view tags
    for the fields which provide hints to how things should be displayed).
    """
    def __init__(ss):
        ss.Net = leabra.Network()
        ss.Pats     = etable.Table()
        ss.TstCycLog   = etable.Table()
        ss.Params     = params.Sets()
        ss.ParamSet = ""
        ss.TestEnv  = env.FixedTable()
        ss.Time     = leabra.Time()
        ss.ViewUpdt = leabra.Cycle
        ss.TstRecLays = go.Slice_string(["Name", "Identity", "Color", "FavoriteFood", "Size", "Species", "FavoriteToy"])
        
        ss.Win        = 0
        ss.vp         = 0
        ss.ToolBar    = 0
        ss.NetView    = 0
        ss.TstCycPlot = 0
        ss.IsRunning    = False
        ss.StopNow    = False
        ss.ValsTsrs   = {}
       
        # ClassView tags for controlling display of fields
        ss.Tags = {
            'ParamSet': 'view:"-"',
            'Win': 'view:"-"',
            'vp': 'view:"-"',
            'ToolBar': 'view:"-"',
            'NetView': 'view:"-"',
            'TstCycPlot': 'view:"-"',
            'IsRunning': 'view:"-"',
            'StopNow': 'view:"-"',
            'ValsTsrs': 'view:"-"',
            'ClassView': 'view:"-"',
            'Tags': 'view:"-"',
        }
    
    def InitParams(ss):
        """
        Sets the default set of parameters -- Base is always applied, and others can be optionally
        selected to apply on top of that
        """
        ss.Params.OpenJSON("cats_dogs.params")

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
        ss.ConfigTstCycLog(ss.TstCycLog)

    def ConfigEnv(ss):
        ss.TestEnv.Nm = "TestEnv"
        ss.TestEnv.Dsc = "testing params and state"
        ss.TestEnv.Table = etable.NewIdxView(ss.Pats)
        ss.TestEnv.Sequential = True
        ss.TestEnv.Validate()
        ss.TestEnv.Init(0)

    def ConfigNet(ss, net):
        net.InitName(net, "CatsAndDogs")
        name = net.AddLayer2D("Name", 1, 10, emer.Input)
        iden = net.AddLayer2D("Identity", 1, 10, emer.Input)
        color = net.AddLayer2D("Color", 1, 4, emer.Input)
        food = net.AddLayer2D("FavoriteFood", 1, 4, emer.Input)
        size = net.AddLayer2D("Size", 1, 3, emer.Input)
        spec = net.AddLayer2D("Species", 1, 2, emer.Input)
        toy = net.AddLayer2D("FavoriteToy", 1, 4, emer.Input)

        name.SetClass("Id")
        iden.SetClass("Id")

        one2one = prjn.NewOneToOne()
        full = prjn.NewFull()

        net.BidirConnectLayersPy(name, iden, one2one)
        net.BidirConnectLayersPy(color, iden, full)
        net.BidirConnectLayersPy(food, iden, full)
        net.BidirConnectLayersPy(size, iden, full)
        net.BidirConnectLayersPy(spec, iden, full)
        net.BidirConnectLayersPy(toy, iden, full)

        iden.SetRelPos(relpos.Rel(Rel= relpos.Above, Other= "Name", YAlign= relpos.Front, XAlign= relpos.Left, YOffset= 1))
        color.SetRelPos(relpos.Rel(Rel= relpos.Above, Other= "Identity", YAlign= relpos.Front, XAlign= relpos.Left, YOffset= 1))
        food.SetRelPos(relpos.Rel(Rel= relpos.Above, Other= "Identity", YAlign= relpos.Front, XAlign= relpos.Right, YOffset= 1))
        size.SetRelPos(relpos.Rel(Rel= relpos.Above, Other= "Color", YAlign= relpos.Front, XAlign= relpos.Left))
        spec.SetRelPos(relpos.Rel(Rel= relpos.Above, Other= "Color", YAlign= relpos.Front, XAlign= relpos.Right, XOffset= 2))
        toy.SetRelPos(relpos.Rel(Rel= relpos.Above, Other= "FavoriteFood", YAlign= relpos.Front, XAlign= relpos.Right, XOffset= 1))

        net.Defaults()
        ss.SetParams("Network", False)
        net.Build()
        ss.InitWts(net)

    def InitWts(ss, net):
        """
        InitWts loads the saved weights
        """
        net.InitWts()
        net.OpenWtsJSON("cats_dogs.wts")

    def Init(ss):
        """
        Init restarts the run, and initializes everything, including network weights
        # ss.ConfigEnv() // re-config env just in case a different set of patterns was
        # selected or patterns have been modified etc
        and resets the epoch log table
        """

        ss.TestEnv.Init(0)
        ss.Time.Reset()
        # ss.Time.CycPerQtr = 25 // use full 100 cycles, default
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
        viewUpdt = ss.ViewUpdt

        ss.Net.AlphaCycInit()
        ss.Time.AlphaCycStart()
        for qtr in range(4):
            for cyc in range(ss.Time.CycPerQtr):
                ss.Net.Cycle(ss.Time)
                ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
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

        lays = go.Slice_string(["Name", "Identity", "Color", "FavoriteFood", "Size", "Species", "FavoriteToy"])
        for lnm in lays :
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
            ss.ClassView.Update()

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
            if ss.ViewUpdt > leabra.AlphaCycle:
                ss.UpdateView()
            return

        ss.ApplyInputs(ss.TestEnv)
        ss.AlphaCyc()

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
                epygiv.ApplyParams(ss, simp, setMsg)
				
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

    def OpenPats(ss):
        ss.Pats.SetMetaData("name", "CatAndDogPats")
        ss.Pats.SetMetaData("desc", "Testing patterns"))
        ss.Pats.OpenCSV("cats_dogs_pats.tsv", etable.Tab)

    def Harmony(ss, nt):
        """
        Harmony computes the harmony (excitatory net input Ge * Act)
        """
        harm = float(0)
        nu = 0
        for lyi in nt.Layers:
            ly = leabra.Layer(handle=lyi)
            if ly.IsOff():
                continue
            for nrni in ly.Neurons:
                nrn = leabra.Neuron(handle=nrni)
                # todo: this call is failling b/c nrn is not a ptr
                # harm += nrn.Ge * nrn.Act
                nu += 1
        if nu > 0:
            harm /= float(nu)
        return harm

    def LogTstCyc(ss, dt, cyc):
        """
        LogTstCyc adds data from current cycle to the TstCycLog table.
        log always contains number of testing items
        """
        if dt.Rows <= cyc:
            dt.SetNumRows(cyc + 1)
        row = cyc

        harm = ss.Harmony(ss.Net)
        dt.SetCellFloat("Cycle", row, float(cyc))
        dt.SetCellString("TrialName", row, ss.TestEnv.TrialName.Cur)
        dt.SetCellFloat("Harmony", row, float(harm))

        for lnm in ss.TstRecLays:
            tsr = ss.ValsTsr(lnm)
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            ly.UnitValsTensor(tsr, "Act")
            dt.SetCellTensor(lnm, row, tsr)

        ss.TstCycPlot.GoUpdate()

    def ConfigTstCycLog(ss, dt):
        dt.SetMetaData("name", "TstCycLog")
        dt.SetMetaData("desc", "Record of testing per cycle")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        nt = 100
        sch = etable.Schema()
        sch.append(etable.Column("Cycle", etensor.INT64, nilInts, nilStrs))
        sch.append(etable.Column("TrialName", etensor.STRING, nilInts, nilStrs))
        sch.append(etable.Column("Harmony", etensor.FLOAT64, nilInts, nilStrs))
        
        for lnm in ss.TstRecLays:
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            sch.append(etable.Column(lnm, etensor.FLOAT64, ly.Shp.Shp, nilStrs))
            print(lnm)
        dt.SetFromSchema(sch, nt)
        print(sch)

    def ConfigTstCycPlot(ss, plt, dt):
        plt.Params.Title = "CatsAndDogs Test Cycle Plot"
        plt.Params.XAxisCol = "Cycle"
        plt.SetTable(dt)

        plt.SetColParams("Cycle", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Harmony", eplot.On, eplot.FixMin, 0, eplot.FixMax, .25)

        for lnm in ss.TstRecLays:
            cp = plt.SetColParams(lnm, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
            # cp.TensorIdx = -1
        return plt

    def ConfigNetView(ss, nv):
        nv.ViewDefaults()
        nv.Scene().Camera.Pose.Pos.Set(0, 1.5, 3.0) # more "head on" than default which is more "top down"
        nv.Scene().Camera.LookAt(mat32.Vec3(0, 0, 0), mat32.Vec3(0, 1, 0))

        labs = go.Slice_string([" Morr Socks Sylv Garf Fuzz Rex Fido Spot Snoop Butch",
            " black white brown orange", "bugs grass scraps shoe", "small  med  large", "cat     dog", "string feath bone shoe"])
        nv.ConfigLabels(labs)

        lays = go.Slice_string(["Name", "Color", "FavoriteFood", "Size", "Species", "FavoriteToy"])

        li = 0
        for lnm in lays :
            ly = nv.LayerByName(lnm)
            lbl = nv.LabelByName(labs[li])
            lbl.Pose = ly.Pose
            lbl.Pose.Pos.Y += .2
            lbl.Pose.Pos.Z += .02
            lbl.Pose.Scale.SetMul(mat32.Vec3(0.4, 0.08, 0.5))
            li += 1

    def ConfigGui(ss):
        """
        ConfigGui configures the GoGi gui interface for this simulation,
        """
        width = 1600
        height = 1200

        gi.SetAppName("cat_dogs")
        gi.SetAppAbout('cats_dogs: This project explores a simple **semantic network** intended to represent a (very small) set of relationships among different features used to represent a set of entities in the world.  In our case, we represent some features of cats and dogs: their color, size, favorite food, and favorite toy. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch3/cats_dogs/README.md">README.md on GitHub</a>.</p>')

        win = gi.NewMainWindow("cats_dogs", "Cats and Dogs", width, height)
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

        ss.ClassView = epygiv.ClassView("ra25sv", ss.Tags)
        ss.ClassView.AddFrame(split)
        ss.ClassView.SetClass(ss)

        tv = gi.AddNewTabView(split, "tv")

        nv = netview.NetView()
        tv.AddTab(nv, "NetView")
        nv.Var = "Act"
        nv.SetNet(ss.Net)
        ss.NetView = nv
        ss.ConfigNetView(nv)

        plt = eplot.Plot2D()
        tv.AddTab(plt, "TstCycPlot")
        ss.TstCycPlot = ss.ConfigTstCycPlot(plt, ss.TstCycLog)

        split.SetSplitsList(go.Slice_float32([.3, .7]))

        recv = win.This()
        
        tbar.AddAction(gi.ActOpts(Label="Init", Icon="update", Tooltip="Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc=UpdtFuncNotRunning), recv, InitCB)

        tbar.AddAction(gi.ActOpts(Label="Stop", Icon="stop", Tooltip="Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc=UpdtFuncRunning), recv, StopCB)
        
        tbar.AddAction(gi.ActOpts(Label="Test Trial", Icon="step-fwd", Tooltip="Runs the next testing trial.", UpdateFunc=UpdtFuncNotRunning), recv, TestTrialCB)
        
        tbar.AddAction(gi.ActOpts(Label="Test Item", Icon="step-fwd", Tooltip="Prompts for a specific input pattern name to run, and runs it in testing mode.", UpdateFunc=UpdtFuncNotRunning), recv, TestItemCB)
        
        tbar.AddAction(gi.ActOpts(Label="Test All", Icon="fast-fwd", Tooltip="Tests all of the testing trials.", UpdateFunc=UpdtFuncNotRunning), recv, TestAllCB)

        tbar.AddSeparator("log")
        
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
    
main(sys.argv[1:])

