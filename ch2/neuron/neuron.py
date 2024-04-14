#!/usr/local/bin/pyleabra -i

# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# use:
# just type file name to run, or:
# pyleabra -i <file>.py 

# neuron: This simulation illustrates the basic properties of neural spiking and
# rate-code activation, reflecting a balance of excitatory and inhibitory
# influences (including leak and synaptic inhibition).

from leabra import go, leabra, emer, relpos, eplot, env, agg, patgen, prjn, etable, efile, split, etensor, params, netview, rand, erand, gi, giv, pygiv, pyparams, math32, spike

import importlib as il
import io, sys, getopt
from datetime import datetime, timezone

import numpy

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

def RunCyclesCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.RunCycles()
        TheSim.IsRunning = False
        TheSim.UpdateClassView()
        TheSim.vp.SetNeedsFullRender()

def ResetPlotCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.ResetTstCycPlot()

def SpikeVsRateCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.SpikeVsRate()
        TheSim.IsRunning = False
        TheSim.UpdateClassView()
        TheSim.vp.SetNeedsFullRender()

def DefaultsCB(recv, send, sig, data):
    TheSim.Defaults()
    TheSim.Init()
    TheSim.UpdateClassView()
    TheSim.vp.SetNeedsFullRender()

def ReadmeCB(recv, send, sig, data):
    gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch2/neuron/README.md")

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
        self.Spike = True
        self.SetTags("Spike", 'desc:"use discrete spiking equations -- otherwise use Noisy X-over-X-plus-1 rate code activation function"')
        self.GbarE = float(0.3)
        self.SetTags("GbarE", 'min:"0" step:"0.01" def:"0.3" desc:"excitatory conductance multiplier -- determines overall value of Ge which drives neuron to be more excited -- pushes up over threshold to fire if strong enough"')
        self.GbarL = float(0.3)
        self.SetTags("GbarL", 'min:"0" step:"0.01" def:"0.3" desc:"leak conductance -- determines overall value of Gl which drives neuron to be less excited (inhibited) -- pushes back to resting membrane potential"')
        self.ErevE = float(1)
        self.SetTags("ErevE", 'min:"0" max:"1" step:"0.01" def:"1" desc:"excitatory reversal (driving) potential -- determines where excitation pushes Vm up to"')
        self.ErevL = float(0.3)
        self.SetTags("ErevL", 'min:"0" max:"1" step:"0.01" def:"0.3" desc:"leak reversal (driving) potential -- determines where excitation pulls Vm down to"')
        self.Noise = float(0.0)
        self.SetTags("Noise", 'min:"0" step:"0.01" desc:"the variance parameter for Gaussian noise added to unit activations on every cycle"')
        self.KNaAdapt = True
        self.SetTags("KNaAdapt", 'desc:"apply sodium-gated potassium adaptation mechanisms that cause the neuron to reduce spiking over time"')
        self.NCycles = int(200)
        self.SetTags("NCycles", 'min:"10" def:"200" desc:"total number of cycles to run"')
        self.OnCycle = int(10)
        self.SetTags("OnCycle", 'min:"0" def:"10" desc:"when does excitatory input into neuron come on?"')
        self.OffCycle = int(160)
        self.SetTags("OffCycle", 'min:"0" def:"160" desc:"when does excitatory input into neuron go off?"')
        self.UpdateInterval = int(10)
        self.SetTags("UpdateInterval", 'min:"1" def:"10"  desc:"how often to update display (in cycles)"')
        self.Net = leabra.Network()
        self.SetTags("Net", 'view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"')
        self.SpikeParams = spike.ActParams()
        self.SetTags("SpikeParams", 'view:"no-inline" desc:"parameters for spiking funcion"')
        self.TstCycLog = etable.Table()
        self.SetTags("TstCycLog", 'view:"no-inline" desc:"testing trial-level log data -- click to see record of network\'s response to each input"')
        self.SpikeVsRateLog = etable.Table()
        self.SetTags("SpikeVsRateLog", 'view:"no-inline" desc:"plot of measured spike rate vs. noisy X/X+1 rate function"')
        self.Params = params.Sets()
        self.SetTags("Params", 'view:"no-inline" desc:"full collection of param sets -- not really interesting for this model"')
        self.ParamSet = str()
        self.SetTags("ParamSet", 'view:"-" desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set -- can use multiple names separated by spaces (don\'t put spaces in ParamSet names!)"')

        self.Cycle = int()
        self.SetTags("Cycle", 'inactive:"+" desc:"current cycle of updating"')

        # internal state - view:"-"
        self.Win = 0
        self.SetTags("Win", 'view:"-" desc:"main GUI window"')
        self.NetView = 0
        self.SetTags("NetView", 'view:"-" desc:"the network viewer"')
        self.ToolBar = 0
        self.SetTags("ToolBar", 'view:"-" desc:"the master toolbar"')
        self.TstCycPlot = 0
        self.SetTags("TstCycPlot", 'view:"-" desc:"the test-trial plot"')
        self.SpikeVsRatePlot = 0
        self.SetTags("SpikeVsRatePlot", 'view:"-" desc:"the spike vs. rate plot"')
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
        ss.Params.OpenJSON("neuron.params")

    def Defaults(ss):
        """
        Defaults sets default params
        """
        ss.SpikeParams.Defaults()
        ss.UpdateInterval = 10
        ss.Cycle = 0
        ss.Spike = True
        ss.GbarE = 0.3
        ss.GbarL = 0.3
        ss.ErevE = 1
        ss.ErevL = 0.3
        ss.Noise = 0
        ss.KNaAdapt = True
        ss.NCycles = 200
        ss.OnCycle = 10
        ss.OffCycle = 160

    def Config(ss):
        """
        Config configures all the elements using the standard functions
        """
        ss.Defaults()
        ss.InitParams()
        ss.ConfigNet(ss.Net)
        ss.ConfigTstCycLog(ss.TstCycLog)
        ss.ConfigSpikeVsRateLog(ss.SpikeVsRateLog)

    def ConfigNet(ss, net):
        net.InitName(net, "Neuron")
        net.AddLayer2D("Neuron", 1, 1, emer.Hidden)

        net.Defaults()
        ss.SetParams("Network", False)
        net.Build()
        ss.InitWts(net)

    def InitWts(ss, net):
        """
        InitWts loads the saved weights
        """
        net.InitWts()

    def Init(ss):
        """
        Init restarts the run, and initializes everything, including network weights
        and resets the epoch log table
        """
        ss.Cycle = 0
        ss.InitWts(ss.Net)
        ss.StopNow = False
        ss.SetParams("", False)
        ss.UpdateView()

    def Counters(ss):
        """
        Counters returns a string of the current counter state
        use tabs to achieve a reasonable formatting overall
        and add a few tabs at the end to allow for expansion..
        """
        return "Cycle:\t%d\t\t\t" % (ss.Cycle)

    def UpdateView(ss):
        if ss.NetView != 0 and ss.NetView.IsVisible():
            ss.NetView.Record(ss.Counters())
            ss.NetView.GoUpdate()

    def RunCycles(ss):
        """
        RunCycles updates neuron over specified number of cycles
        """
        ss.Init()
        ss.StopNow = False
        ss.Net.InitActs()
        ss.SetParams("", False)
        ly = leabra.Layer(ss.Net.LayerByName("Neuron"))
        nrn = ly.Neurons[0]
        inputOn = False
        for cyc in range(ss.NCycles):
            if ss.Win != 0:
                ss.Win.PollEvents() # this is essential for GUI responsiveness while running
                
            ss.Cycle = cyc
            if cyc == ss.OnCycle:
                inputOn = True
            if cyc == ss.OffCycle:
                inputOn = False
            nrn.Noise = float(ly.Act.Noise.Gen(-1))
            if inputOn:
                nrn.Ge = 1
            else:
                nrn.Ge = 0
            nrn.Ge += nrn.Noise # GeNoise
            nrn.Gi = 0
            if ss.Spike:
                ss.SpikeUpdate(ss.Net, inputOn)
            else:
                ss.RateUpdate(ss.Net, inputOn)
            ss.LogTstCyc(ss.TstCycLog, ss.Cycle)
            if ss.Cycle%ss.UpdateInterval == 0:
                ss.UpdateView()
            if ss.StopNow:
                break
        ss.UpdateView()

    def RateUpdate(ss, nt, inputOn):
        """
        RateUpdate updates the neuron in rate-code mode
        this just calls the relevant activation code directly, bypassing most other stuff.
        """
        ly = leabra.Layer(ss.Net.LayerByName("Neuron"))
        nrn = leabra.Neuron(ly.Neurons[0])
        ly.Act.VmFmG(nrn)
        ly.Act.ActFmG(nrn)
        nrn.Ge = nrn.Ge * ly.Act.Gbar.E

    def SpikeUpdate(ss, nt, inputOn):
        """
        SpikeUpdate updates the neuron in spiking mode
        which is just computed directly as spiking is not yet implemented in main codebase
        """
        ly = leabra.Layer(ss.Net.LayerByName("Neuron"))
        nrn = ly.Neurons[0]
        ss.SpikeParams.SpikeVmFmG(nrn)
        ss.SpikeParams.SpikeActFmVm(nrn)
        nrn.Ge = nrn.Ge * ly.Act.Gbar.E

    def Stop(ss):
        """
        Stop tells the sim to stop running
        """
        ss.StopNow = True

    def SpikeVsRate(ss):
        """
        SpikeVsRate runs comparison between spiking vs. rate-code
        """
        row = 0
        nsamp = 100

        for gbarE in numpy.arange(0.1, 0.7, 0.025):
            ss.GbarE = float(gbarE)
            spike = float(0)
            ss.Noise = 0.1
            ss.Spike = True
            for ns in range(nsamp):
                ss.RunCycles()
                if ss.StopNow:
                    break
                act = ss.TstCycLog.CellFloat("Act", 159)
                spike += act
            rate = float(0)
            ss.Spike = False
            # ss.Noise = 0 // doesn't make much diff
            for ns in range(nsamp):
                ss.RunCycles()
                if ss.StopNow:
                    break
                act = ss.TstCycLog.CellFloat("Act", 159)
                rate += act
            if ss.StopNow:
                break
            spike /= float(nsamp)
            rate /= float(nsamp)
            ss.LogSpikeVsRate(ss.SpikeVsRateLog, row, gbarE, spike, rate)
            row += 1
        ss.Defaults()
        ss.SpikeVsRatePlot.GoUpdate()

    def SetParams(ss, sheet, setMsg):
        """
        SetParams sets the params for "Base" and then current ParamSet.

        # this is important for catching typos and ensuring that all sheets can be used
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
        ly = leabra.Layer(ss.Net.LayerByName("Neuron"))
        ly.Act.Gbar.E = float(ss.GbarE)
        ly.Act.Gbar.L = float(ss.GbarL)
        ly.Act.Erev.E = float(ss.ErevE)
        ly.Act.Erev.L = float(ss.ErevL)
        ly.Act.Noise.Var = float(ss.Noise)
        ly.Act.KNa.On = ss.KNaAdapt
        ly.Act.Update()
        ss.SpikeParams.CopyFromAct(ly.Act) # keep sync'd

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

    def LogTstCyc(ss, dt, cyc):
        """
        LogTstCyc adds data from current cycle to the TstCycLog table.
        """
        if dt.Rows <= cyc:
            dt.SetNumRows(cyc + 1)
        row = cyc

        ly = leabra.Layer(ss.Net.LayerByName("Neuron"))
        nrn = leabra.Neuron(ly.Neurons[0])

        dt.SetCellFloat("Cycle", row, float(cyc))
        dt.SetCellFloat("Ge", row, float(nrn.Ge))
        dt.SetCellFloat("Inet", row, float(nrn.Inet))
        dt.SetCellFloat("Vm", row, float(nrn.Vm))
        dt.SetCellFloat("Act", row, float(nrn.Act))
        dt.SetCellFloat("Spike", row, float(nrn.Spike))
        dt.SetCellFloat("Gk", row, float(nrn.Gk))
        dt.SetCellFloat("ISI", row, float(nrn.ISI))
        dt.SetCellFloat("AvgISI", row, float(nrn.ISIAvg))

        # note: essential to use Go version of update when called from another goroutine
        if cyc%ss.UpdateInterval == 0:
            ss.TstCycPlot.GoUpdate()

    def ConfigTstCycLog(ss, dt):
        dt.SetMetaData("name", "TstCycLog")
        dt.SetMetaData("desc", "Record of testing per cycle")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        nt = ss.NCycles # max cycles
        sch = etable.Schema(
            [etable.Column("Cycle", etensor.INT64, go.nil, go.nil),
            etable.Column("Ge", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("Inet", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("Vm", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("Act", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("Spike", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("Gk", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("ISI", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("AvgISI", etensor.FLOAT64, go.nil, go.nil)]
        )

        dt.SetFromSchema(sch, nt)

    def ConfigTstCycPlot(ss, plt, dt):
        plt.Params.Title = "Neuron Test Cycle Plot"
        plt.Params.XAxisCol = "Cycle"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Cycle", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Ge", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("Inet", eplot.On, eplot.FixMin, -.2, eplot.FixMax, 1)
        plt.SetColParams("Vm", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("Act", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("Spike", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("Gk", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("ISI", eplot.Off, eplot.FixMin, -2, eplot.FloatMax, 1)
        plt.SetColParams("AvgISI", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
        return plt

    def ResetTstCycPlot(ss):
        ss.TstCycLog.SetNumRows(0)
        ss.TstCycPlot.Update()

    def LogSpikeVsRate(ss, dt, row, gbarE, spike, rate):
        """
        LogSpikeVsRate adds data from current cycle to the SpikeVsRateLog table.
        """
        if dt.Rows <= row:
            dt.SetNumRows(row + 1)
        dt.SetCellFloat("GBarE", row, gbarE)
        dt.SetCellFloat("Spike", row, spike)
        dt.SetCellFloat("Rate", row, rate)

    def ConfigSpikeVsRateLog(ss, dt):
        dt.SetMetaData("name", "SpikeVsRateLog")
        dt.SetMetaData("desc", "Record spiking vs. rate-code activation")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        nt = 24 # typical number
        sch = etable.Schema(
            [etable.Column("GBarE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("Spike", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("Rate", etensor.FLOAT64, go.nil, go.nil)]
        )
        dt.SetFromSchema(sch, nt)

    def ConfigSpikeVsRatePlot(ss, plt, dt):
        plt.Params.Title = "Neuron Spike Vs. Rate-Code Plot"
        plt.Params.XAxisCol = "GBarE"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("GBarE", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("Spike", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("Rate", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        return plt

    def ConfigNetView(ss, nv):
        nv.ViewDefaults()

    def ConfigGui(ss):
        """
        ConfigGui configures the GoGi gui interface for this simulation,
        """
        width = 1600
        height = 1200

        gi.SetAppName("neuron")
        gi.SetAppAbout('This simulation illustrates the basic properties of neural spiking and rate-code activation, reflecting a balance of excitatory and inhibitory influences (including leak and synaptic inhibition). See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch2/neuron/README.md">README.md on GitHub</a>.</p>')

        win = gi.NewMainWindow("neuron", "Neuron", width, height)
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
        split.SetStretchMaxWidth()
        split.SetStretchMaxHeight()

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
        tv.AddTab(plt, "TstCycPlot")
        ss.TstCycPlot = ss.ConfigTstCycPlot(plt, ss.TstCycLog)

        plt = eplot.Plot2D()
        tv.AddTab(plt, "SpikeVsRatePlot")
        ss.SpikeVsRatePlot = ss.ConfigSpikeVsRatePlot(plt, ss.SpikeVsRateLog)

        split.SetSplitsList(go.Slice_float32([.2, .8]))

        recv = win.This()
        
        tbar.AddAction(gi.ActOpts(Label="Init", Icon="update", Tooltip="Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc=UpdateFuncNotRunning), recv, InitCB)

        tbar.AddAction(gi.ActOpts(Label="Stop", Icon="stop", Tooltip="Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc=UpdateFuncRunning), recv, StopCB)
        
        tbar.AddAction(gi.ActOpts(Label="Run Cycles", Icon="step-fwd", Tooltip="Runs neuron updating over NCycles.", UpdateFunc=UpdateFuncNotRunning), recv, RunCyclesCB)
        
        tbar.AddAction(gi.ActOpts(Label="Reset Plot", Icon="update", Tooltip="Reset TstCycPlot", UpdateFunc=UpdateFuncNotRunning), recv, ResetPlotCB)

        tbar.AddAction(gi.ActOpts(Label="Spike Vs Rate", Icon="play", Tooltip="Runs Spike vs Rate Test", UpdateFunc=UpdateFuncNotRunning), recv, SpikeVsRateCB)

        tbar.AddSeparator("log")
        
        tbar.AddAction(gi.ActOpts(Label= "Defaults", Icon= "update", Tooltip= "Restore initial default parameters.", UpdateFunc= UpdateFuncNotRunning), recv, DefaultsCB)

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

