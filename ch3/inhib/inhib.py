#!/usr/local/bin/pyleabra -i

# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# use:
# just type file name to run, or:
# pyleabra -i <file>.py 

# inhib: This simulation explores how inhibitory interneurons can dynamically
# control overall activity levels within the network, by providing both
# feedforward and feedback inhibition to excitatory pyramidal neurons.

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
        TheSim.TestTrial()
        TheSim.IsRunning = False
        TheSim.ClassView.Update()
        TheSim.vp.SetNeedsFullRender()

def ConfigPatsCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.ConfigPats()
        TheSim.vp.SetNeedsFullRender()

def DefaultsCB(recv, send, sig, data):
    TheSim.Defaults()
    TheSim.Init()
    TheSim.ClassView.Update()
    TheSim.vp.SetNeedsFullRender()

def ReadmeCB(recv, send, sig, data):
    gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch3/inhib/README.md")

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
        ss.BidirNet = False
        ss.TrainedWts = False
        ss.InputPct = 20.0
        ss.FFFBinhib = False
        ss.HiddenGbarI = 0.4
        ss.InhibGbarI = 0.75
        ss.FFinhibWtScale = 1.0
        ss.FBinhibWtScale = 1.0
        ss.HiddenGTau = 40.0
        ss.InhibGTau = 20.0
        ss.FmInhibWtScaleAbs = 1.0
        
        ss.NetFF = leabra.Network()
        ss.NetBidir = leabra.Network()
        ss.Pats     = etable.Table()
        ss.TstCycLog   = etable.Table()
        ss.Params     = params.Sets()
        ss.ParamSet = ""
        ss.Time     = leabra.Time()
        ss.ViewUpdt = leabra.Cycle
        ss.TstRecLays = go.Slice_string(["Hidden", "Inhib"])

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
        ss.Params.OpenJSON("inhib.params")


    def Defaults(ss):
        """
        Defaults sets default params
        """
        ss.TrainedWts = False
        ss.InputPct = 20
        ss.FFFBInhib = False
        ss.HiddenGbarI = 0.4
        ss.InhibGbarI = 0.75
        ss.FFinhibWtScale = 1
        ss.FBinhibWtScale = 1
        ss.HiddenGTau = 40
        ss.InhibGTau = 20
        ss.FmInhibWtScaleAbs = 1
        ss.Time.CycPerQtr = 50

    def Config(ss):
        """
        Config configures all the elements using the standard functions
        """
        ss.Defaults()
        ss.InitParams()
        ss.ConfigPats()
        ss.ConfigNetFF(ss.NetFF)
        ss.ConfigNetBidir(ss.NetBidir)
        ss.ConfigTstCycLog(ss.TstCycLog)

    def ConfigNetFF(ss, net):
        net.InitName(net, "InhibFF")
        inp = net.AddLayer2D("Input", 10, 10, emer.Input)
        hid = net.AddLayer2D("Hidden", 10, 10, emer.Hidden)
        inh = net.AddLayer2D("Inhib", 10, 2, emer.Hidden)
        inh.SetClass("InhibLay")

        full = prjn.NewFull()

        pj = net.ConnectLayers(inp, hid, full, emer.Forward)
        pj.SetClass("Excite")
        net.ConnectLayers(hid, inh, full, emer.Back)
        net.ConnectLayers(inp, inh, full, emer.Forward)
        net.ConnectLayers(inh, hid, full, emer.Inhib)
        net.ConnectLayers(inh, inh, full, emer.Inhib)

        inh.SetRelPos(relpos.Rel(Rel= relpos.RightOf, Other= "Hidden", YAlign= relpos.Front, Space= 1))

        net.Defaults()
        ss.SetParams("Network", False)
        net.Build()
        ss.InitWts(net)

    def ConfigNetBidir(ss, net):
        net.InitName(net, "InhibBidir")
        inp = net.AddLayer2D("Input", 10, 10, emer.Input)
        hid = net.AddLayer2D("Hidden", 10, 10, emer.Hidden)
        inh = net.AddLayer2D("Inhib", 10, 2, emer.Hidden)
        inh.SetClass("InhibLay")
        hid2 = net.AddLayer2D("Hidden2", 10, 10, emer.Hidden)
        inh2 = net.AddLayer2D("Inhib2", 10, 2, emer.Hidden)
        inh2.SetClass("InhibLay")

        full = prjn.NewFull()

        pj = net.ConnectLayers(inp, hid, full, emer.Forward)
        pj.SetClass("Excite")
        net.ConnectLayers(inp, inh, full, emer.Forward)
        net.ConnectLayers(hid2, inh, full, emer.Forward)
        net.ConnectLayers(hid, inh, full, emer.Back)
        net.ConnectLayers(inh, hid, full, emer.Inhib)
        net.ConnectLayers(inh, inh, full, emer.Inhib)

        pj = net.ConnectLayers(hid, hid2, full, emer.Forward)
        pj.SetClass("Excite")
        pj = net.ConnectLayers(hid2, hid, full, emer.Back)
        pj.SetClass("Excite")
        net.ConnectLayers(hid, inh2, full, emer.Forward)
        net.ConnectLayers(hid2, inh2, full, emer.Back)
        net.ConnectLayers(inh2, hid2, full, emer.Inhib)
        net.ConnectLayers(inh2, inh2, full, emer.Inhib)

        inh.SetRelPos(relpos.Rel(Rel= relpos.RightOf, Other= "Hidden", YAlign= relpos.Front, Space= 1))
        hid2.SetRelPos(relpos.Rel(Rel= relpos.Above, Other= "Hidden", YAlign= relpos.Front, XAlign= relpos.Middle))
        inh2.SetRelPos(relpos.Rel(Rel= relpos.RightOf, Other= "Hidden2", YAlign= relpos.Front, Space= 1))

        net.Defaults()
        ss.SetParams("Network", False)
        net.Build()
        ss.InitWts(net)

    def InitWts(ss, net):
        """
        InitWts loads the saved weights
        """
        net.InitWts()

    def ConfigPats(ss):
        dt = ss.Pats
        dt.SetMetaData("name", "TrainPats")
        dt.SetMetaData("desc", "Training patterns")
        sch = etable.Schema(
            [etable.Column("Name", etensor.STRING, go.nil, go.nil),
            etable.Column("Input", etensor.FLOAT32, go.Slice_int([10, 10]), go.Slice_string(["Y", "X"]))]
        )
        dt.SetFromSchema(sch, 1)
        patgen.PermutedBinaryRows(dt.Cols[1], int(ss.InputPct), 1, 0)

    def Init(ss):
        """
        Init restarts the run, and initializes everything, including network weights
        and resets the epoch log table
        """
        ss.Time.Reset()
        ss.StopNow = False
        ss.SetParams("", False)
        ss.InitWts(ss.NetFF)
        ss.InitWts(ss.NetBidir)
        ss.UpdateView()

    def Counters(ss):
        """
        Counters returns a string of the current counter state
        use tabs to achieve a reasonable formatting overall
        and add a few tabs at the end to allow for expansion..
        """
        return "Cycle:\t%d\t\t\t" % (ss.Time.Cycle)

    def UpdateView(ss):
        nv = ss.NetViewFF
        if ss.BidirNet:
            nv = ss.NetViewBidir
        if nv != 0 and nv.IsVisible():
            nv.Record(ss.Counters())
            nv.GoUpdate() # note: using counters is significantly slower..

    def Net(ss):
        """
        Net returns the current active network
        """
        if ss.BidirNet:
            return ss.NetBidir
        else:
            return ss.NetFF

    def AlphaCyc(ss):
        """
        AlphaCyc runs one alpha-cycle (100 msec, 4 quarters) of processing.
        External inputs must have already been applied prior to calling,
        using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
        Handles netview updating within scope of AlphaCycle
        """

        if ss.Win != 0:
            ss.Win.PollEvents() # this is essential for GUI responsiveness while running
        viewUpdt = ss.ViewUpdt

        nt = ss.Net()

        nt.AlphaCycInit()
        ss.Time.AlphaCycStart()
        for qtr in range(4):
            for cyc in range(ss.Time.CycPerQtr):
                nt.Cycle(ss.Time)
                ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
                ss.Time.CycleInc()
                if viewUpdt == leabra.Cycle:
                    if cyc != ss.Time.CycPerQtr-1: # will be updated by quarter
                        ss.UpdateView()
                if viewUpdt == leabra.FastSpike:
                    if (cyc+1)%10 == 0:
                        ss.UpdateView()
            nt.QuarterFinal(ss.Time)
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
        nt = ss.Net()
        nt.InitExt()

        ly = leabra.Layer(nt.LayerByName("Input"))
        pat = ss.Pats.CellTensor("Input", 0)
        ly.ApplyExt(pat)

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

    def TestTrial(ss):
        """
        TestTrial runs one trial of testing -- always sequentially presented inputs
        """
        nt = ss.Net()
        nt.InitActs()
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
        nt = ss.Net()
        err = ss.SetParamsSet("Base", sheet, setMsg)
        if ss.TrainedWts:
            ss.SetParamsSet("Trained", sheet, setMsg)
        else:
            ss.SetParamsSet("Untrained", sheet, setMsg)
        ffinhsc = ss.FFinhibWtScale
        if nt == ss.NetBidir:
            ffinhsc *= 0.5 # 2 inhib prjns so .5 ea

        hid = leabra.LeabraLayer(nt.LayerByName("Hidden")).AsLeabra()
        hid.Act.Gbar.I = ss.HiddenGbarI
        hid.Act.Dt.GTau = ss.HiddenGTau
        hid.Act.Update()
        inh = leabra.LeabraLayer(nt.LayerByName("Inhib")).AsLeabra()
        inh.Act.Gbar.I = ss.InhibGbarI
        inh.Act.Dt.GTau = ss.InhibGTau
        inh.Act.Update()
        ff = leabra.LeabraPrjn(inh.RcvPrjns.SendName("Input")).AsLeabra()
        ff.WtScale.Rel = ffinhsc
        fb = leabra.LeabraPrjn(inh.RcvPrjns.SendName("Hidden")).AsLeabra()
        fb.WtScale.Rel = ss.FBinhibWtScale
        hid.Inhib.Layer.On = ss.FFFBInhib
        inh.Inhib.Layer.On = ss.FFFBInhib
        fi = leabra.LeabraPrjn(hid.RcvPrjns.SendName("Inhib")).AsLeabra()
        fi.WtScale.Abs = ss.FmInhibWtScaleAbs
        fi = leabra.LeabraPrjn(inh.RcvPrjns.SendName("Inhib")).AsLeabra()
        fi.WtScale.Abs = ss.FmInhibWtScaleAbs
        if nt == ss.NetBidir:
            hid = leabra.LeabraLayer(nt.LayerByName("Hidden2")).AsLeabra()
            hid.Act.Gbar.I = ss.HiddenGbarI
            hid.Act.Dt.GTau = ss.HiddenGTau
            hid.Act.Update()
            inh = leabra.LeabraLayer(nt.LayerByName("Inhib2")).AsLeabra()
            inh.Act.Gbar.I = ss.InhibGbarI
            inh.Act.Dt.GTau = ss.InhibGTau
            inh.Act.Update()
            hid.Inhib.Layer.On = ss.FFFBInhib
            inh.Inhib.Layer.On = ss.FFFBInhib
            fi = leabra.LeabraPrjn(hid.RcvPrjns.SendName("Inhib2")).AsLeabra()
            fi.WtScale.Abs = ss.FmInhibWtScaleAbs
            fi = leabra.LeabraPrjn(inh.RcvPrjns.SendName("Inhib2")).AsLeabra()
            fi.WtScale.Abs = ss.FmInhibWtScaleAbs
            ff = leabra.LeabraPrjn(inh.RcvPrjns.SendName("Hidden")).AsLeabra()
            ff.WtScale.Rel = ffinhsc
            fb = leabra.LeabraPrjn(inh.RcvPrjns.SendName("Hidden2")).AsLeabra()
            fb.WtScale.Rel = ss.FBinhibWtScale
            inh = leabra.LeabraLayer(nt.LayerByName("Inhib")).AsLeabra()
            ff = leabra.LeabraPrjn(inh.RcvPrjns.SendName("Hidden2")).AsLeabra()
            ff.WtScale.Rel = ffinhsc
        return err

    def SetParamsSet(ss, setNm, sheet, setMsg):
        """
        SetParamsSet sets the params for given params.Set name.
        If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
        otherwise just the named sheet
        if setMsg = true then we output a message for each param that was set.
        """
        nt = ss.Net()
        pset = ss.Params.SetByNameTry(setNm)
        if sheet == "" or sheet == "Network":
            netp = pset.SheetByNameTry("Network")
            nt.ApplyParams(netp, setMsg)

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

    def LogTstCyc(ss, dt, cyc):
        """
        LogTstCyc adds data from current cycle to the TstCycLog table.
        log always contains number of testing items
        """
        nt = ss.Net()
        if dt.Rows <= cyc:
            dt.SetNumRows(cyc + 1)
        row = cyc

        dt.SetCellFloat("Cycle", row, float(cyc))

        for lnm in ss.TstRecLays:
            ly = leabra.Layer(nt.LayerByName(lnm))
            dt.SetCellFloat(lnm+"ActAvg", row, float(ly.Pool(0).Inhib.Act.Avg))

        # note: essential to use Go version of update when called from another goroutine
        if cyc%10 == 0:
            ss.TstCycPlot.GoUpdate()

    def ConfigTstCycLog(ss, dt):
        dt.SetMetaData("name", "TstCycLog")
        dt.SetMetaData("desc", "Record of testing per cycle")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        ncy = 200 # max cycles
        sch = etable.Schema(
            [etable.Column("Cycle", etensor.INT64, go.nil, go.nil)]
        )
        for lnm in ss.TstRecLays:
            sch.append(etable.Column(lnm + "ActAvg", etensor.FLOAT64, go.nil, go.nil))
        dt.SetFromSchema(sch, ncy)

    def ConfigTstCycPlot(ss, plt, dt):
        plt.Params.Title = "Inhib Test Cycle Plot"
        plt.Params.XAxisCol = "Cycle"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Cycle", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

        for lnm in ss.TstRecLays:
            plt.SetColParams(lnm+"ActAvg", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        return plt

    def ConfigGui(ss):
        """
        ConfigGui configures the GoGi gui interface for this simulation,
        """
        width = 1600
        height = 1200

        gi.SetAppName("inhib")
        gi.SetAppAbout('This simulation explores how inhibitory interneurons can dynamically control overall activity levels within the network, by providing both feedforward and feedback inhibition to excitatory pyramidal neurons. See <a href="https://github.com/CompCogNeuro/sims/ch3/inhib/README.md">README.md on GitHub</a>.</p>')

        win = gi.NewMainWindow("inhib", "Inhibition", width, height)
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
        tv.AddTab(nv, "FF Net")
        nv.Var = "Act"
        nv.Params.MaxRecs = 200
        nv.SetNet(ss.NetFF)
        ss.NetViewFF = nv
        nv.ViewDefaults()

        nv = netview.NetView()
        tv.AddTab(nv, "Bidir Net")
        nv.Var = "Act"
        nv.Params.MaxRecs = 200
        nv.SetNet(ss.NetBidir)
        ss.NetViewBidir = nv

        plt = eplot.Plot2D()
        tv.AddTab(plt, "TstCycPlot")
        ss.TstCycPlot = ss.ConfigTstCycPlot(plt, ss.TstCycLog)

        split.SetSplitsList(go.Slice_float32([.2, .8]))

        recv = win.This()
        
        tbar.AddAction(gi.ActOpts(Label="Init", Icon="update", Tooltip="Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc=UpdtFuncNotRunning), recv, InitCB)

        tbar.AddAction(gi.ActOpts(Label="Stop", Icon="stop", Tooltip="Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc=UpdtFuncRunning), recv, StopCB)
        
        tbar.AddAction(gi.ActOpts(Label="Test Trial", Icon="step-fwd", Tooltip="Runs the next testing trial.", UpdateFunc=UpdtFuncNotRunning), recv, TestTrialCB)
        
        tbar.AddSeparator("log")
        
        tbar.AddAction(gi.ActOpts(Label= "Config Pats", Icon= "update", Tooltip= "Generates a new input pattern based on current InputPct amount.", UpdateFunc=UpdtFuncNotRunning), recv, ConfigPatsCB)

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

