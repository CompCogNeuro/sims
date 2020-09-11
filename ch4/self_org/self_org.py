#!/usr/local/bin/pyleabra -i

# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# use:
# just type file name to run, or:
# pyleabra -i <file>.py 

# self_org illustrates how self-organizing learning emerges from
# the interactions between inhibitory competition, rich-get-richer
# Hebbian learning, and homeostasis (negative feedback).

from leabra import go, leabra, emer, relpos, eplot, env, agg, patgen, prjn, etable, efile, split, etensor, params, netview, rand, erand, gi, giv, epygiv, mat32, simat, etview, norm, metric

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
    TheSim.ClassView.Update()
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
        TheSim.ClassView.Update()
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

def ResetRunLogCB(recv, send, sig, data):
    TheSim.RunLog.SetNumRows(0)
    TheSim.RunPlot.Update()

def NewRndSeedCB(recv, send, sig, data):
    TheSim.NewRndSeed()

def ReadmeCB(recv, send, sig, data):
    gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch4/self_org/README.md")

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
        ss.AvgLGain   = 2.5
        ss.InputNoise = 0.0
        ss.TrainGi    = 1.8
        ss.TestGi     = 2.5
        ss.Net = leabra.Network()
        ss.Lines2     = etable.Table()
        ss.Lines1     = etable.Table()
        ss.TrnEpcLog = etable.Table()
        ss.TstEpcLog = etable.Table()
        ss.TstTrlLog = etable.Table()
        ss.HidFmInputWts = etensor.Float32()
        ss.RunLog = etable.Table()
        ss.RunStats = etable.Table()
        ss.SimMat   = simat.SimMat()
        ss.Params    = params.Sets()
        ss.ParamSet = ""
        ss.MaxRuns  = 8
        ss.MaxEpcs  = 30
        ss.TrainEnv = env.FixedTable()
        ss.TestEnv  = env.FixedTable()
        ss.Time     = leabra.Time()
        ss.ViewOn   = True
        ss.TrainUpdt = leabra.AlphaCycle
        ss.TestUpdt = leabra.Cycle
        ss.TestInterval = 1
        ss.TstRecLays  = go.Slice_string(["Input", "Hidden"])
        
        # statistics
        ss.UniqPats   = 0.0
        
        # internal state - view:"-"
        ss.Win        = 0
        ss.vp         = 0
        ss.ToolBar    = 0
        ss.NetView    = 0
        ss.WtsGrid    = 0
        ss.TrnEpcPlot = 0
        ss.TstEpcPlot = 0
        ss.TstTrlPlot = 0
        ss.RunPlot    = 0
        ss.TrnEpcFile = 0
        ss.RunFile    = 0

        ss.Win        = 0
        ss.vp         = 0
        ss.ToolBar    = 0
        ss.NetView    = 0
        ss.TstTrlPlot = 0
        ss.IsRunning    = False
        ss.StopNow    = False
        ss.NeedsNewRun = False
        ss.RndSeed    = 0
        ss.ValsTsrs   = {}
       
        # ClassView tags for controlling display of fields
        ss.Tags = {
            'ParamSet': 'view:"-"',
            'UniqPats': 'inactive:"+"',
            'Win': 'view:"-"',
            'vp': 'view:"-"',
            'ToolBar': 'view:"-"',
            'NetView': 'view:"-"',
            'TrnEpcPlot': 'view:"-"',
            'TstEpcPlot': 'view:"-"',
            'TstTrlPlot': 'view:"-"',
            'RunPlot': 'view:"-"',
            'TrnEpcFile': 'view:"-"',
            'RunFile': 'view:"-"',
            'SaveWts': 'view:"-"',
            'NoGui': 'view:"-"',
            'IsRunning': 'view:"-"',
            'StopNow': 'view:"-"',
            'RndSeed': 'view:"-"',
            'NeedsNewRun': 'view:"-"',
            'ValsTsrs': 'view:"-"',
            'ClassView': 'view:"-"',
            'Tags': 'view:"-"',
        }
    
    def InitParams(ss):
        """
        Sets the default set of parameters -- Base is always applied, and others can be optionally
        selected to apply on top of that
        """
        ss.Params.OpenJSON("self_org.params")

    def Defaults(ss):
        ss.AvgLGain = 2.5
        ss.InputNoise = 0
        ss.TrainGi = 1.8
        ss.TestGi = 2.5

    def Config(ss):
        """
        Config configures all the elements using the standard functions
        """
        ss.InitParams()
        ss.Defaults()
        ss.OpenPats()
        ss.ConfigEnv()
        ss.ConfigNet(ss.Net)
        ss.ConfigTrnEpcLog(ss.TrnEpcLog)
        ss.ConfigTstEpcLog(ss.TstEpcLog)
        ss.ConfigTstTrlLog(ss.TstTrlLog)
        ss.ConfigRunLog(ss.RunLog)

    def ConfigEnv(ss):
        if ss.MaxRuns == 0:
            ss.MaxRuns = 8
        if ss.MaxEpcs == 0: # allow user override
            ss.MaxEpcs = 30

        ss.TrainEnv.Nm = "TrainEnv"
        ss.TrainEnv.Dsc = "training params and state"
        ss.TrainEnv.Table = etable.NewIdxView(ss.Lines2)
        ss.TrainEnv.Validate()
        ss.TrainEnv.Run.Max = ss.MaxRuns # note: we are not setting epoch max -- do that manually

        ss.TestEnv.Nm = "TestEnv"
        ss.TestEnv.Dsc = "testing params and state"
        ss.TestEnv.Table = etable.NewIdxView(ss.Lines1)
        ss.TestEnv.Sequential = True
        ss.TestEnv.Validate()

        ss.TrainEnv.Init(0)
        ss.TestEnv.Init(0)

    def ConfigNet(ss, net):
        net.InitName(net, "SelfOrg")
        inp = net.AddLayer2D("Input", 5, 5, emer.Input)
        hid = net.AddLayer2D("Hidden", 4, 5, emer.Hidden)

        net.ConnectLayers(inp, hid, prjn.NewFull(), emer.Forward)

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
            return "Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t" % (ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TestEnv.Trial.Cur, ss.Time.Cycle, ss.TestEnv.TrialName.Cur)

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
        viewUpdt = ss.TrainUpdt
        if not train:
            viewUpdt = ss.TestUpdt

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

        # going to the same layers, but good practice and cheap anyway
        args so that it can be used for various different contexts
        (training, testing, etc).
        """
        ss.Net.InitExt()

        lays = go.Slice_string(["Input"])
        for lnm in lays:
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            pats = en.State(ly.Nm)
            if pats == 0:
                continue
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
            if ss.ViewOn and ss.TrainUpdt > leabra.AlphaCycle:
                ss.UpdateView(True)
            if epc%ss.TestInterval == 0: # note: epc is *next* so won't trigger first time
                ss.TestAll()
            ss.LogTrnEpc(ss.TrnEpcLog)
            if epc >= ss.MaxEpcs:
                # done with training..
                ss.RunEnd()
                if ss.TrainEnv.Run.Incr(): # we are done!
                    ss.StopNow = True
                    return
                else:
                    ss.NeedsNewRun = True
                    return

        ss.SetParamsSet("Hidden2Act", "Network", False) # 2 units active for training
        ss.ApplyInputs(ss.TrainEnv)
        ss.AlphaCyc(True)   # train
        ss.TrialStats(True) # accumulate

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
        ss.InitStats()
        ss.TrnEpcLog.SetNumRows(0)
        ss.TstEpcLog.SetNumRows(0)
        ss.NeedsNewRun = False

    def InitStats(ss):
        """
        InitStats initializes all the statistics, especially important for the
        cumulative epoch stats -- called at start of new run
        """
        pass

    def TrialStats(ss, accum):
        pass

    def UniquePatStat(ss, dt):
        """
        UniquePatStat analyzes the hidden activity patterns for the single-line test inputs
        to determine how many such lines have a distinct hidden pattern, as computed
        from the similarity matrix across patterns
        """
        hc = etensor.Float64(dt.ColByName("Hidden"))
        norm.Binarize64(hc.Values, .5, 1, 0)
        ix = etable.NewIdxView(dt)
        ss.SimMat.TableColStd(ix, "Hidden", "TrialName", False, metric.SumSquares)
        dm = ss.SimMat.Mat
        nrow = dm.Dim(0)
        uniq = 0
        for row in range(nrow):
            tsr = etensor.Float64(dm.SubSpace(go.Slice_int([row])))
            nzero = 0
            for vl in tsr.Values:
                if vl == 0:
                    nzero += 1
            if nzero == 1:
                uniq += 1
        return float(uniq)

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
            ss.ClassView.Update()

    def SaveWeights(ss, filename):
        """
        SaveWeights saves the network weights -- when called with giv.CallMethod
        it will auto-prompt for filename
        """
        ss.Net.SaveWtsJSON(filename)

    def TestTrial(ss, returnOnChg):
        """
        TestTrial runs one trial of testing -- always sequentially presented inputs
        """
        ss.TestEnv.Step()

        chg = env.CounterChg(ss.TestEnv, env.Epoch)
        if chg:
            if ss.ViewOn and ss.TestUpdt > leabra.AlphaCycle:
                ss.UpdateView(False)
            ss.LogTstEpc(ss.TstEpcLog)
            if returnOnChg:
                return
        ss.SetParamsSet("Hidden1Act", "Network", False)

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
        ss.TestEnv.SetTrialName()
        ss.SetParamsSet("Hidden1Act", "Network", False)
        ss.ApplyInputs(ss.TestEnv)
        ss.AlphaCyc(False)
        ss.TrialStats(False)
        ss.TestEnv.Trial.Cur = cur

    def TestAll(ss):
        """
        TestAll runs through the full set of testing items
        """
        ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
        while True:
            ss.TestTrial(True)
            chg = env.CounterChg(ss.TestEnv, env.Epoch)
            if chg or ss.StopNow:
                break
        ss.UniqPats = ss.UniquePatStat(ss.TstTrlLog)

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
        ly = leabra.Layer(ss.Net.LayerByName("Hidden"))
        ly.Learn.AvgL.Gain = ss.AvgLGain
        inp = leabra.Layer(ss.Net.LayerByName("Input"))
        if ss.InputNoise == 0:
            inp.Act.Noise.Var = 0
            inp.Act.Noise.Type = leabra.NoNoise
        else:
            inp.Act.Noise.Var = float(ss.InputNoise)
            inp.Act.Noise.Type = leabra.ActNoise

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

    def OpenPats(ss):
        ss.Lines2.SetMetaData("name", "Lines2")
        ss.Lines2.SetMetaData("desc", "Lines2 Training patterns")
        ss.Lines2.OpenCSV("lines_5x5x2.tsv", etable.Tab)

        ss.Lines1.SetMetaData("name", "Lines1")
        ss.Lines1.SetMetaData("desc", "Lines1 Training patterns")
        ss.Lines1.OpenCSV("lines_5x5x1.tsv", etable.Tab)

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
        """
        row = dt.Rows
        dt.SetNumRows(row + 1)

        epc = ss.TrainEnv.Epoch.Prv
        # nt := float(ss.TrainEnv.Table.Len()) // number of trials in view

        ss.HidFmInput(ss.HidFmInputWts)
        if ss.WtsGrid != 0:
            ss.WtsGrid.UpdateSig()

        dt.SetCellFloat("Run", row, float(ss.TrainEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellFloat("UniqPats", row, ss.UniqPats)
        dt.SetCellTensor("HidFmInputWts", row, ss.HidFmInputWts)

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

        sch = etable.Schema()
        sch.append(etable.Column("Run", etensor.INT64, go.nil, go.nil))
        sch.append(etable.Column("Epoch", etensor.INT64, go.nil, go.nil))
        sch.append(etable.Column("UniqPats", etensor.FLOAT64, go.nil, go.nil))
        sch.append(etable.Column("HidFmInputWts", etensor.FLOAT32, go.Slice_int([4, 5, 5, 5]), go.nil))
        dt.SetFromSchema(sch, 0)
        ss.ConfigHidFmInput(ss.HidFmInputWts)

    def ConfigTrnEpcPlot(ss, plt, dt):
        plt.Params.Title = "Self Organizing Epoch Plot"
        plt.Params.XAxisCol = "Epoch"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("UniqPats", eplot.On, eplot.FixMin, 0, eplot.FixMax, 10)
        plt.SetColParams("HidFmInputWts", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

        return plt

    def HidFmInput(ss, dt):
        vals = dt.Values
        inp = leabra.Layer(ss.Net.LayerByName("Input"))
        isz = inp.Shape().Len()
        hid = leabra.Layer(ss.Net.LayerByName("Hidden"))
        ysz = hid.Shape().Dim(0)
        xsz = hid.Shape().Dim(1)
        for y in range(ysz):
            for x in range(xsz):
                ui = (y*xsz + x)
                ust = ui * isz
                vls = vals[ust : ust+isz]
                inp.SendPrjnVals(vls, "Wt", hid, ui, "")

    def ConfigHidFmInput(ss, dt):
        dt.SetShape(go.Slice_int([4, 5, 5, 5]), go.nil, go.nil)

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
        dt.SetCellString("TrialName", row, ss.TestEnv.TrialName.Cur)

        for lnm in ss.TstRecLays:
            tsr = ss.ValsTsr(lnm)
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            ly.UnitValsTensor(tsr, "Act")
            dt.SetCellTensor(lnm, row, tsr)

        # note: essential to use Go version of update when called from another goroutine
        ss.TstTrlPlot.GoUpdate()

    def ConfigTstTrlLog(ss, dt):
        dt.SetMetaData("name", "TstTrlLog")
        dt.SetMetaData("desc", "Record of testing per input pattern")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        nt = ss.TestEnv.Table.Len() # number in view
        sch = etable.Schema()
        sch.append(etable.Column("Run", etensor.INT64, go.nil, go.nil))
        sch.append(etable.Column("Epoch", etensor.INT64, go.nil, go.nil))
        sch.append(etable.Column("Trial", etensor.INT64, go.nil, go.nil))
        sch.append(etable.Column("TrialName", etensor.STRING, go.nil, go.nil))
        for lnm in ss.TstRecLays:
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            sch.append(etable.Column(lnm, etensor.FLOAT64, ly.Shp.Shp, go.nil))
        dt.SetFromSchema(sch, nt)

    def ConfigTstTrlPlot(ss, plt, dt):
        plt.Params.Title = "Self Organizing Test Trial Plot"
        plt.Params.XAxisCol = "Trial"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

        for lnm in ss.TstRecLays:
            plt.SetColParams(lnm, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        return plt

    def LogTstEpc(ss, dt):
        row = dt.Rows
        dt.SetNumRows(row + 1)

        # trl := ss.TstTrlLog
        # tix := etable.NewIdxView(trl)
        epc = ss.TrainEnv.Epoch.Prv # ?

        # note: this shows how to use agg methods to compute summary data from another
        # data table, instead of incrementing on the Sim
        dt.SetCellFloat("Run", row, float(ss.TrainEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))

        # note: essential to use Go version of update when called from another goroutine
        if ss.TstEpcPlot != 0:
            ss.TstEpcPlot.GoUpdate()

    def ConfigTstEpcLog(ss, dt):
        dt.SetMetaData("name", "TstEpcLog")
        dt.SetMetaData("desc", "Summary stats for testing trials")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        sch = etable.Schema()
        sch.append(etable.Column("Run", etensor.INT64, go.nil, go.nil))
        sch.append(etable.Column("Epoch", etensor.INT64, go.nil, go.nil))
        dt.SetFromSchema(sch, 0)
        
    def ConfigTstEpcPlot(ss, plt, dt):
        plt.Params.Title = "Self Organizing Testing Epoch Plot"
        plt.Params.XAxisCol = "Epoch"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
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
        nlast = 1
        if nlast > epcix.Len()-1:
            nlast = epcix.Len() - 1
        epcix.Idxs = epcix.Idxs[epcix.Len()-nlast-1:]

        params = "Std"
        if ss.AvgLGain != 2.5:
            params += "_AvgLGain=%s" % (ss.AvgLGain)
        if ss.InputNoise != 0:
            params += "_InVar=%s" % (ss.InputNoise)

        dt.SetCellFloat("Run", row, float(run))
        dt.SetCellString("Params", row, params)
        dt.SetCellFloat("UniqPats", row, agg.Mean(epcix, "UniqPats")[0])

        runix = etable.NewIdxView(dt)
        spl = split.GroupBy(runix, go.Slice_string(["Params"]))
        split.Desc(spl, "UniqPats")
        ss.RunStats = spl.AggsToTable(etable.AddAggName)

        # note: essential to use Go version of update when called from another goroutine
        ss.RunPlot.GoUpdate()

    def ConfigRunLog(ss, dt):
        dt.SetMetaData("name", "RunLog")
        dt.SetMetaData("desc", "Record of performance at end of training")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        sch = etable.Schema()
        sch.append(etable.Column("Run", etensor.INT64, go.nil, go.nil))
        sch.append(etable.Column("Params", etensor.STRING, go.nil, go.nil))
        sch.append(etable.Column("UniqPats", etensor.FLOAT64, go.nil, go.nil))
        dt.SetFromSchema(sch, 0)

    def ConfigRunPlot(ss, plt, dt):
        plt.Params.Title = "Self Organizing Run Plot"
        plt.Params.XAxisCol = "Run"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("UniqPats", eplot.On, eplot.FixMin, 0, eplot.FixMax, 10)
        return plt

    def ConfigGui(ss):
        """
        ConfigGui configures the GoGi gui interface for this simulation,
        """
        width = 1600
        height = 1200

        gi.SetAppName("self_org")
        gi.SetAppAbout('illustrates how self-organizing learning emerges from the interactions between inhibitory competition, rich-get-richer Hebbian learning, and homeostasis (negative feedback). See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch4/self_org/README.md">README.md on GitHub</a>.</p>')

        win = gi.NewMainWindow("self_org", "Self Organizing", width, height)
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

        plt = eplot.Plot2D()
        tv.AddTab(plt, "TrnEpcPlot")
        ss.TrnEpcPlot = ss.ConfigTrnEpcPlot(plt, ss.TrnEpcLog)

        tg = etview.TensorGrid()
        tv.AddTab(tg, "Weights")
        tg.SetStretchMax()
        ss.WtsGrid = tg
        tg.SetTensor(ss.HidFmInputWts)

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


