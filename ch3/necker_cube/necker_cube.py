#!/usr/local/bin/pyleabra -i

# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# use:
# just type file name to run, or:
# pyleabra -i <file>.py 

# necker_cube: This simulation explores the use of constraint
# satisfaction in processing ambiguous stimuli. The example we 
# will use is the *Necker cube*, which and can be viewed as a
# cube in one of two orientations, where people flip back and forth.

from leabra import go, leabra, emer, relpos, eplot, env, agg, patgen, prjn, etable, efile, split, etensor, params, netview, rand, erand, gi, giv, epygiv, mat32

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
    TheSim.ClassView.Update()
    TheSim.vp.SetNeedsFullRender()

def StopCB(recv, send, sig, data):
    TheSim.Stop()

def TestTrialCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        # if TheSim.CycPerQtr == 25:
        #     TheSim.TestTrial() # show every update
        # else:
        #     go TheSim.TestTrial() # fast..
        TheSim.TestTrial()
        TheSim.IsRunning = False
        TheSim.ClassView.Update()
        TheSim.vp.SetNeedsFullRender()

def DefaultsCB(recv, send, sig, data):
    TheSim.Defaults()
    TheSim.Init()
    TheSim.ClassView.Update()
    TheSim.vp.SetNeedsFullRender()

def ReadmeCB(recv, send, sig, data):
    gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch3/necker_cube/README.md")

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
        ss.Noise = 0.01
        ss.KNaAdapt = False
        ss.CycPerQtr = 25
        
        ss.Net = leabra.Network()
        ss.TstCycLog   = etable.Table()
        ss.Params     = params.Sets()
        ss.ParamSet = ""
        ss.TestEnv  = env.FixedTable()
        ss.Time     = leabra.Time()
        ss.ViewUpdt = leabra.Cycle
        ss.TstRecLays = go.Slice_string(["NeckerCube"])

        ss.Win        = 0
        ss.vp         = 0
        ss.ToolBar    = 0
        ss.NetViewFF  = 0
        ss.NetViewBidir  = 0
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
        ss.Params.OpenJSON("necker_cube.params")

    def Defaults(ss):
        """
        Defaults sets default params
        """
        ss.Noise = 0.01
        ss.KNaAdapt = False
        ss.CycPerQtr = 25

    def Config(ss):
        """
        Config configures all the elements using the standard functions
        """
        ss.Defaults()
        ss.InitParams()
        ss.ConfigNet(ss.Net)
        ss.ConfigTstCycLog(ss.TstCycLog)

    def ConfigNet(ss, net):
        net.InitName(net, "NeckerCube")
        nc = net.AddLayer4D("NeckerCube", 1, 2, 4, 2, emer.Input)

        net.ConnectLayers(nc, nc, prjn.NewFull(), emer.Lateral)

        net.Defaults()
        ss.SetParams("Network", False)
        net.Build()
        ss.InitWts(net)

    def InitWts(ss, net):
        """
        InitWts loads the saved weights
        """
        net.InitWts()
        net.OpenWtsJSON("necker_cube.wts")

    def Init(ss):
        """
        Init restarts the run, and initializes everything,
        including network weights and resets the epoch log table
        """
        ss.Time.Reset()

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
        return "Cycle:\t%d\t\t\t" % (ss.Time.Cycle)

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

    def ApplyInputs(ss):
        """
        ApplyInputs applies input patterns from given envirbonment.
        It is good practice to have this be a separate method with appropriate
        args so that it can be used for various different contexts
        (training, testing, etc).
        """
        ss.Net.InitExt()

        ly = leabra.Layer(ss.Net.LayerByName("NeckerCube"))
        tsr = ss.ValsTsr("Inputs")
        tsr.SetShape(go.Slice_int([16]), go.nil, go.nil)
        if tsr.FloatVal1D(0) != 1.0:
            for i in range(16):
                tsr.SetFloat1D(i, 1)
        ly.ApplyExt(tsr)

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
        ss.Net.InitActs()
        ss.SetParams("", False)
        ss.ApplyInputs()
        ss.AlphaCyc()

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
        ly = leabra.Layer(ss.Net.LayerByName("NeckerCube"))
        ly.Act.Noise.Var = float(ss.Noise)
        ly.Act.KNa.On = ss.KNaAdapt
        ly.Act.Update()
        ss.Time.CycPerQtr = int(ss.CycPerQtr)

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
                harm += nrn.Ge * nrn.Act
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
        ly = leabra.LeabraLayer(ss.Net.LayerByName("NeckerCube")).AsLeabra()
        dt.SetCellFloat("Cycle", row, float(cyc))
        dt.SetCellFloat("Harmony", row, float(harm))
        # dt.SetCellFloat("GknaFast", row, float(ly.Neurons[0].GknaFast))
        # dt.SetCellFloat("GknaMed", row, float(ly.Neurons[0].GknaMed))
        # dt.SetCellFloat("GknaSlow", row, float(ly.Neurons[0].GknaSlow))

        for lnm in ss.TstRecLays:
            tsr = ss.ValsTsr(lnm)
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            ly.UnitValsTensor(tsr, "Act")
            dt.SetCellTensor(lnm, row, tsr)

        if cyc%10 == 0:
            ss.TstCycPlot.GoUpdate()

    def ConfigTstCycLog(ss, dt):
        dt.SetMetaData("name", "TstCycLog")
        dt.SetMetaData("desc", "Record of testing per cycle")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        nt = 100
        sch = etable.Schema()
        sch.append(etable.Column("Cycle", etensor.INT64, go.nil, go.nil))
        sch.append(etable.Column("TrialName", etensor.STRING, go.nil, go.nil))
        sch.append(etable.Column("Harmony", etensor.FLOAT64, go.nil, go.nil))
        sch.append(etable.Column("GknaFast", etensor.FLOAT64, go.nil, go.nil))
        sch.append(etable.Column("GknaMed", etensor.FLOAT64, go.nil, go.nil))
        sch.append(etable.Column("GknaSlow", etensor.FLOAT64, go.nil, go.nil))

        for lnm in ss.TstRecLays:
            ly = leabra.LeabraLayer(ss.Net.LayerByName(lnm)).AsLeabra()
            sch.append(etable.Column(lnm, etensor.FLOAT64, ly.Shp.Shp, go.nil))
        dt.SetFromSchema(sch, nt)

    def ConfigTstCycPlot(ss, plt, dt):
        plt.Params.Title = "Necker Cube Test Cycle Plot"
        plt.Params.XAxisCol = "Cycle"
        plt.SetTable(dt)

        plt.SetColParams("Cycle", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Harmony", eplot.On, eplot.FixMin, 0, eplot.FixMax, 0.25)
        plt.SetColParams("GknaFast", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 0.25)
        plt.SetColParams("GknaMed", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 0.25)
        plt.SetColParams("GknaSlow", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 0.25)

        for lnm in ss.TstRecLays:
            plt.SetColParams(lnm, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        return plt

    def ConfigNetView(ss, nv):
        nv.ViewDefaults()

    def ConfigGui(ss):
        """
        ConfigGui configures the GoGi gui interface for this simulation,
        """
        width = 1600
        height = 1200

        gi.SetAppName("necker_cube")
        gi.SetAppAbout('This simulation explores the use of constraint satisfaction in processing ambiguous stimuli. The example we will use is the *Necker cube*, which and can be viewed as a cube in one of two orientations, where people flip back and forth.  See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch3/necker_cube/README.md">README.md on GitHub</a>.</p>')

        win = gi.NewMainWindow("necker_cube", "Necker Cube", width, height)
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

        ss.ClassView = epygiv.ClassView("sv", ss.Tags)
        ss.ClassView.AddFrame(split)
        ss.ClassView.SetClass(ss)

        tv = gi.AddNewTabView(split, "tv")

        nv = netview.NetView()
        tv.AddTab(nv, "NetView")
        nv.Var = "Act"
        nv.Params.MaxRecs = 1000
        nv.SetNet(ss.Net)
        ss.NetView = nv
        ss.ConfigNetView(nv)

        plt = eplot.Plot2D()
        tv.AddTab(plt, "TstCycPlot")
        ss.TstCycPlot = ss.ConfigTstCycPlot(plt, ss.TstCycLog)

        split.SetSplitsList(go.Slice_float32([.2, .8]))

        recv = win.This()
        
        tbar.AddAction(gi.ActOpts(Label="Init", Icon="update", Tooltip="Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc=UpdtFuncNotRunning), recv, InitCB)

        tbar.AddAction(gi.ActOpts(Label="Stop", Icon="stop", Tooltip="Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc=UpdtFuncRunning), recv, StopCB)
        
        tbar.AddAction(gi.ActOpts(Label="Test Trial", Icon="step-fwd", Tooltip="Runs the next testing trial.", UpdateFunc=UpdtFuncNotRunning), recv, TestTrialCB)
        
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
    TheSim.Init()
    
main(sys.argv[1:])

