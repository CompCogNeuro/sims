#!/usr/local/bin/pyleabra -i

# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# use:
# just type file name to run, or:
# pyleabra -i <file>.py 

# attn: This simulation illustrates how object recognition (ventral, what) and
# spatial (dorsal, where) pathways interact to produce spatial attention
# effects, and accurately capture the effects of brain damage to the
# spatial pathway.

from leabra import go, leabra, emer, relpos, eplot, env, agg, patgen, prjn, etable, efile, split, etensor, params, netview, rand, erand, gi, giv, pygiv, pyparams, mat32, simat, metric, clust

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
    gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch6/attn/README.md")

def UpdtFuncNotRunning(act):
    act.SetActiveStateUpdt(not TheSim.IsRunning)
    
def UpdtFuncRunning(act):
    act.SetActiveStateUpdt(TheSim.IsRunning)

#####################################################    
#     Enums
    
class TestType(Enum):
    """
    TestType is the type of testing patterns
    """
    MultiObjs = 0
    StdPosner = 1
    ClosePosner = 2
    ReversePosner = 3
    ObjAttn = 4

class LesionType(Enum):
    """
    LesionType is the type of lesion
    """
    NoLesion = 0
    LesionSpat1 = 1
    LesionSpat2 = 2
    LesionSpat12 = 3

NoLesion = 0
LesionSpat1 = 1
LesionSpat2 = 2
LesionSpat12 = 3
    
class LesionSize(Enum):
    LesionHalf = 0
    LesionFull = 1

LesionHalf = 0
LesionFull = 1

class LesionParams(pygiv.ClassViewObj):
    def __init__(self):
        super(LesionParams, self).__init__()
        self.Layers = LesionType.NoLesion
        self.Locations = LesionSize.LesionHalf
        self.Units = LesionSize.LesionHalf

TheLesion = LesionParams()        
        
def LesionCB2(recv, send, sig, data):
    TheSim.Lesion(TheLesion.Layers.value, TheLesion.Locations.value, TheLesion.Units.value)

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
    LesionDialog(TheSim.vp, TheLesion, "les", {}, giv.DlgOpts(Title="Lesion", Prompt="Lesion spatial pathways:"))

    
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
        self.SpatToObj = float(2)
        self.SetTags("SpatToObj", 'def:"2" desc:"spatial to object projection WtScale.Rel strength -- reduce to 1.5, 1 to test"')
        self.V1ToSpat1 = float(0.6)
        self.SetTags("V1ToSpat1", 'def:"0.6" desc:"V1 to Spat1 projection WtScale.Rel strength -- reduce to .55, .5 to test"')
        self.KNaAdapt = False
        self.SetTags("KNaAdapt", 'def:"false" desc:"sodium (Na) gated potassium (K) channels that cause neurons to fatigue over time"')
        self.CueDur = int(100)
        self.SetTags("CueDur", 'def:"100" desc:"number of cycles to present the cue -- 100 by default, 50 to 300 for KNa adapt testing"')
        self.Net = leabra.Network()
        self.SetTags("Net", 'view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"')
        self.Test = TestType.MultiObjs
        self.SetTags("Test", 'desc:"select which type of test (input patterns) to use"')
        self.MultiObjs = etable.Table()
        self.SetTags("MultiObjs", 'view:"no-inline" desc:"click to see these testing input patterns"')
        self.StdPosner = etable.Table()
        self.SetTags("StdPosner", 'view:"no-inline" desc:"click to see these testing input patterns"')
        self.ClosePosner = etable.Table()
        self.SetTags("ClosePosner", 'view:"no-inline" desc:"click to see these testing input patterns"')
        self.ReversePosner = etable.Table()
        self.SetTags("ReversePosner", 'view:"no-inline" desc:"click to see these testing input patterns"')
        self.ObjAttn = etable.Table()
        self.SetTags("ObjAttn", 'view:"no-inline" desc:"click to see these testing input patterns"')
        self.TstTrlLog = etable.Table()
        self.SetTags("TstTrlLog", 'view:"no-inline" desc:"testing trial-level log data -- click to see record of network\'s response to each input"')
        self.TstStats = etable.Table()
        self.SetTags("TstStats", 'view:"no-inline" desc:"aggregate stats on testing data"')
        self.Params = params.Sets()
        self.SetTags("Params", 'view:"no-inline" desc:"full collection of param sets -- not really interesting for this model"')
        self.ParamSet = str()
        self.SetTags("ParamSet", 'view:"-" desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set -- can use multiple names separated by spaces (don\'t put spaces in ParamSet names!)"')
        self.TestEnv = env.FixedTable()
        self.SetTags("TestEnv", 'desc:"Testing environment -- manages iterating over testing"')
        self.Time = leabra.Time()
        self.SetTags("Time", 'desc:"leabra timing parameters and state"')
        self.ViewUpdt = leabra.TimeScales.FastSpike
        self.SetTags("ViewUpdt", 'desc:"at what time scale to update the display during testing?  Change to AlphaCyc to make display updating go faster"')
        self.TstRecLays = go.Slice_string(["Input", "V1", "Spat1", "Spat2", "Obj1", "Obj2", "Output"])
        self.SetTags("TstRecLays", 'desc:"names of layers to record activations etc of during testing"')

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
        ss.Params.OpenJSON("attn.params")
        ss.Defaults()

    def Defaults(ss):
        """
        Defaults sets default params
        """
        ss.SpatToObj = 2
        ss.V1ToSpat1 = 0.6
        ss.KNaAdapt = False
        ss.CueDur = 100

    def Config(ss):
        """
        Config configures all the elements using the standard functions
        """
        ss.InitParams()
        ss.OpenPats()
        ss.ConfigEnv()
        ss.ConfigNet(ss.Net)
        ss.ConfigTstTrlLog(ss.TstTrlLog)

    def ConfigEnv(ss):
        ss.TestEnv.Nm = "TestEnv"
        ss.TestEnv.Dsc = "testing params and state"
        ss.TestEnv.Table = etable.NewIdxView(ss.MultiObjs)
        ss.TestEnv.Sequential = True
        ss.TestEnv.Validate()
        ss.TestEnv.Init(0)

    def UpdateEnv(ss):
        if ss.Test == TestType.MultiObjs:
            ss.TestEnv.Table = etable.NewIdxView(ss.MultiObjs)
        if ss.Test == TestType.StdPosner:
            ss.TestEnv.Table = etable.NewIdxView(ss.StdPosner)
        if ss.Test == TestType.ClosePosner:
            ss.TestEnv.Table = etable.NewIdxView(ss.ClosePosner)
        if ss.Test == TestType.ReversePosner:
            ss.TestEnv.Table = etable.NewIdxView(ss.ReversePosner)
        if ss.Test == TestType.ObjAttn:
            ss.TestEnv.Table = etable.NewIdxView(ss.ObjAttn)

    def ConfigNet(ss, net):
        net.InitName(net, "AttnNet")
        inp = net.AddLayer4D("Input", 1, 7, 2, 1, emer.Input)
        v1 = net.AddLayer4D("V1", 1, 7, 2, 1, emer.Hidden)
        sp1 = net.AddLayer4D("Spat1", 1, 5, 2, 1, emer.Hidden)
        sp2 = net.AddLayer4D("Spat2", 1, 3, 2, 1, emer.Hidden)
        ob1 = net.AddLayer4D("Obj1", 1, 5, 2, 1, emer.Hidden)
        out = net.AddLayer2D("Output", 2, 1, emer.Compare)
        ob2 = net.AddLayer4D("Obj2", 1, 3, 2, 1, emer.Hidden)

        ob1.SetClass("Object")
        ob2.SetClass("Object")
        sp1.SetClass("Spatial")
        sp2.SetClass("Spatial")

        full = prjn.NewFull()
        net.ConnectLayers(inp, v1, prjn.NewOneToOne(), emer.Forward)

        rec3sp = prjn.NewRect()
        rec3sp.Size.Set(3, 2)
        rec3sp.Scale.Set(1, 0)
        rec3sp.Start.Set(0, 0)

        rec3sptd = prjn.NewRect()
        rec3sptd.Size.Set(3, 2)
        rec3sptd.Scale.Set(1, 0)
        rec3sptd.Start.Set(-2, 0)
        rec3sptd.Wrap = False

        net.BidirConnectLayersPy(v1, sp1, full)
        v1sp1 = v1.SendPrjns().RecvName(sp1.Name())
        sp1v1 = v1.RecvPrjns().SendName(sp1.Name())
        v1sp1.SetPattern(rec3sp)
        sp1v1.SetPattern(rec3sptd)

        net.BidirConnectLayersPy(sp1, sp2, full)
        sp1sp2 = sp1.SendPrjns().RecvName(sp2.Name())
        sp2sp1 = sp1.RecvPrjns().SendName(sp2.Name())
        sp1sp2.SetPattern(rec3sp)
        sp2sp1.SetPattern(rec3sptd)

        rec3ob = prjn.NewRect()
        rec3ob.Size.Set(3, 1)
        rec3ob.Scale.Set(1, 1)
        rec3ob.Start.Set(0, 0)

        rec3obtd = prjn.NewRect()
        rec3obtd.Size.Set(3, 1)
        rec3obtd.Scale.Set(1, 1)
        rec3obtd.Start.Set(-2, 0)
        rec3obtd.Wrap = False

        net.BidirConnectLayersPy(v1, ob1, full)
        v1ob1 = v1.SendPrjns().RecvName(ob1.Name())
        ob1v1 = v1.RecvPrjns().SendName(ob1.Name())
        v1ob1.SetPattern(rec3ob)
        ob1v1.SetPattern(rec3obtd)

        net.BidirConnectLayersPy(ob1, ob2, full)
        ob1ob2 = ob1.SendPrjns().RecvName(ob2.Name())
        ob2ob1 = ob1.RecvPrjns().SendName(ob2.Name())
        ob1ob2.SetPattern(rec3ob)
        ob2ob1.SetPattern(rec3obtd)

        recout = prjn.NewRect()
        recout.Size.Set(1, 1)
        recout.Scale.Set(0, 1)
        recout.Start.Set(0, 0)

        net.BidirConnectLayersPy(ob2, out, full)
        ob2out = ob2.SendPrjns().RecvName(out.Name())
        outob2 = ob2.RecvPrjns().SendName(out.Name())
        ob2out.SetPattern(rec3ob)
        outob2.SetPattern(recout)

        p1to1 = prjn.NewPoolOneToOne()
        net.BidirConnectLayersPy(sp1, ob1, p1to1)
        spob1 = sp1.SendPrjns().RecvName(ob1.Name())
        obsp1 = sp1.RecvPrjns().SendName(ob1.Name())
        net.BidirConnectLayersPy(sp2, ob2, p1to1)
        spob2 = sp2.SendPrjns().RecvName(ob2.Name())
        obsp2 = sp2.RecvPrjns().SendName(ob2.Name())

        spob1.SetClass("SpatToObj")
        spob2.SetClass("SpatToObj")
        obsp1.SetClass("ObjToSpat")
        obsp2.SetClass("ObjToSpat")

        rec1slf = prjn.NewRect()
        rec1slf.Size.Set(1, 2)
        rec1slf.Scale.Set(1, 0)
        rec1slf.Start.Set(0, 0)
        rec1slf.SelfCon = False
        net.ConnectLayers(sp1, sp1, rec1slf, emer.Lateral)
        net.ConnectLayers(sp2, sp2, rec1slf, emer.Lateral)

        sp1.SetRelPos(relpos.Rel(Rel= relpos.Above, Other= "V1", YAlign= relpos.Front, XAlign= relpos.Left, YOffset= 1))
        sp2.SetRelPos(relpos.Rel(Rel= relpos.Above, Other= "Spat1", YAlign= relpos.Front, XAlign= relpos.Left, Space= 1))
        ob1.SetRelPos(relpos.Rel(Rel= relpos.RightOf, Other= "Spat1", YAlign= relpos.Front, Space= 1))
        out.SetRelPos(relpos.Rel(Rel= relpos.RightOf, Other= "Spat2", YAlign= relpos.Front, Space= 1))
        ob2.SetRelPos(relpos.Rel(Rel= relpos.RightOf, Other= "Output", YAlign= relpos.Front, Space= 1))

        net.Defaults()
        ss.SetParams("Network", False) # only set Network params
        net.Build()
        ss.InitWts(net)

    def InitWts(ss, net):
        """
        InitWts loads the saved weights
        """
        net.InitWts()

    def LesionUnit(ss, lay, unx, uny):
        """
        LesionUnit lesions given unit number in given layer by setting all weights to 0
        """
        ui = etensor.Prjn2DIdx(lay.Shape(), False, uny, unx)
        rpj = lay.RecvPrjns()
        for pji in rpj:
            pj = leabra.Prjn(handle=pji)  # todo: not clear why handle needed here?
            nc = int(pj.RConN[ui])
            st = int(pj.RConIdxSt[ui])
            for ci in range(nc):
                rsi = pj.RSynIdx[st+ci]
                sy = pj.Syns[rsi]
                sy.Wt = 0
                pj.Learn.LWtFmWt(sy)

    def Lesion(ss, lay, locations, units):
        """
        Lesion lesions given set of layers (or unlesions for NoLesion) and
        locations and number of units (Half = partial = 1/2 units, Full = both units)
        """
        ss.InitWts(ss.Net)
        if lay == NoLesion:
            return
        if lay == LesionSpat1 or lay == LesionSpat12:
            sp1 = leabra.LeabraLayer(ss.Net.LayerByName("Spat1"))
            ss.LesionUnit(sp1, 3, 1)
            ss.LesionUnit(sp1, 4, 1)
            if units == LesionFull:
                ss.LesionUnit(sp1, 3, 0)
                ss.LesionUnit(sp1, 4, 0)
            if locations == LesionFull:
                ss.LesionUnit(sp1, 0, 1)
                ss.LesionUnit(sp1, 1, 1)
                ss.LesionUnit(sp1, 2, 1)
                if units == LesionFull:
                    ss.LesionUnit(sp1, 0, 0)
                    ss.LesionUnit(sp1, 1, 0)
                    ss.LesionUnit(sp1, 2, 0)
        if lay == LesionSpat2 or lay == LesionSpat12:
            sp2 = leabra.LeabraLayer(ss.Net.LayerByName("Spat2"))
            ss.LesionUnit(sp2, 2, 1)
            if units == LesionFull:
                ss.LesionUnit(sp2, 2, 0)
            if locations == LesionFull:
                ss.LesionUnit(sp2, 0, 1)
                ss.LesionUnit(sp2, 1, 1)
                if units == LesionFull:
                    ss.LesionUnit(sp2, 0, 0)
                    ss.LesionUnit(sp2, 1, 0)

    def Init(ss):
        """
        Init restarts the run, and initializes everything, including network weights
        and resets the epoch log table
        """
        ss.UpdateEnv()
        ss.TestEnv.Init(0)
        ss.Time.Reset()
        ss.Time.CycPerQtr = 55

        ss.StopNow = False
        ss.SetParams("", False)
        ss.TstTrlLog.SetNumRows(0)
        ss.UpdateView()

    def Counters(ss):
        """
        Counters returns a string of the current counter state
        use tabs to achieve a reasonable formatting overall
        and add a few tabs at the end to allow for expansion..
        """
        nm = ss.TestEnv.GroupName.Cur
        if ss.TestEnv.TrialName.Cur != nm:
            nm += ": " + ss.TestEnv.TrialName.Cur
        return "Trial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t" % (ss.TestEnv.Trial.Cur, ss.Time.Cycle, nm)

    def UpdateView(ss):
        if ss.NetView != 0 and ss.NetView.IsVisible():
            ss.NetView.Record(ss.Counters())
            ss.NetView.GoUpdate() # note: using counters is significantly slower..

    def AlphaCyc(ss):
        """
        AlphaCyc runs one alpha-cycle (100 msec, 4 quarters) of processing.
        External inputs must have already been applied prior to calling,
        using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
        Handles netview updating within scope of AlphaCycle
        """

        if ss.Win != 0:
            ss.Win.PollEvents() # this is essential for GUI responsiveness while running
        viewUpdt = ss.ViewUpdt.value

        out = leabra.Layer(ss.Net.LayerByName("Output"))

        ss.Net.AlphaCycInit(False)
        ss.Time.AlphaCycStart()
        overThresh = False
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
                trgact = out.Neurons[1].Act
                if trgact > 0.5:
                    overThresh = True
                    break
            ss.Net.QuarterFinal(ss.Time)
            ss.Time.QuarterInc()
            if viewUpdt <= leabra.Quarter:
                ss.UpdateView()
            if viewUpdt == leabra.Phase:
                if qtr >= 2:
                    ss.UpdateView()
            if overThresh:
                break

        ss.UpdateView()

    def AlphaCycCue(ss):
        """
        AlphaCycCue just runs over fixed number of cycles -- for Cue trials
        """
        ss.Net.AlphaCycInit(False)
        ss.Time.AlphaCycStart()
        for cyc in range(ss.CueDur):
            ss.Net.Cycle(ss.Time)
            ss.Time.CycleInc()
            if (cyc+1)%10 == 0:
                ss.UpdateView()
        ss.Net.QuarterFinal(ss.Time)
        ss.Time.QuarterInc()

        ss.UpdateView()

    def ApplyInputs(ss, en):
        """
        ApplyInputs applies input patterns from given envirbonment.
        It is good practice to have this be a separate method with appropriate
        args so that it can be used for various different contexts
        (training, testing, etc).
        """
        ss.Net.InitExt()

        lays = go.Slice_string(["Input", "Output"])
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

    def SaveWts(ss, filename):
        """
        SaveWts saves the network weights -- when called with giv.CallMethod
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

        isCue = (ss.TestEnv.TrialName.Cur == "Cue")

        if ss.TestEnv.TrialName.Prv != "Cue":
            ss.Net.InitActs()
        ss.ApplyInputs(ss.TestEnv)
        if isCue:
            ss.AlphaCycCue()
        else:
            ss.AlphaCyc()
            ss.LogTstTrl(ss.TstTrlLog)

    def TestTrialGUI(ss):
        """
        TestTrialGUI runs one trial of testing -- always sequentially presented inputs
        """
        ss.TestTrial()
        ss.Stopped()

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

    def TestItemGUI(ss, idx):
        """
        TestItemGUI tests given item which is at given index in test item list
        """
        ss.TestItem(idx)
        ss.Stopped()

    def TestAll(ss):
        """
        TestAll runs through the full set of testing items
        """
        ss.SetParams("", False)
        ss.UpdateEnv()
        ss.TestEnv.Init(0)
        while True:
            ss.TestTrial()
            chg = env.CounterChg(ss.TestEnv, env.Epoch)
            if chg or ss.StopNow:
                break
        ss.TestStats()

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

        spo = ss.Params.SetByName("Base").SheetByName("Network").SelByName(".SpatToObj")
        spo.Params.SetParamByName("Prjn.WtScale.Rel", ("%g" % (ss.SpatToObj)))

        vsp = ss.Params.SetByName("Base").SheetByName("Network").SelByName("#V1ToSpat1")
        vsp.Params.SetParamByName("Prjn.WtScale.Rel", ("%g" % (ss.V1ToSpat1)))

        ss.SetParamsSet("Base", sheet, setMsg)
        if ss.ParamSet != "" and ss.ParamSet != "Base":
            sps = ss.ParamSet.split()
            for ps in sps:
                ss.SetParamsSet(ps, sheet, setMsg)
        if ss.KNaAdapt:
            ss.SetParamsSet("KNaAdapt", sheet, setMsg)

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

    def OpenPatFile(ss, dt, fnm, name, desc):
        """
        OpenPatFile opens pattern file from file
        """
        dt.SetMetaData("name", name)
        dt.SetMetaData("desc", desc)
        dt.OpenCSV(fnm, etable.Tab)
        for i in range(len(dt.Cols)):
            dt.Cols[i].SetMetaData("grid-fill", "0.9")

    def OpenPats(ss):
        ss.OpenPatFile(ss.MultiObjs, "multi_objs.tsv", "MultiObjs", "multiple object filtering")
        ss.OpenPatFile(ss.StdPosner, "std_posner.tsv", "StdPosner", "standard Posner spatial cuing task")
        ss.OpenPatFile(ss.ClosePosner, "close_posner.tsv", "ClosePosner", "close together Posner spatial cuing task")
        ss.OpenPatFile(ss.ReversePosner, "reverse_posner.tsv", "ReversePosner", "reverse position Posner spatial cuing task")
        ss.OpenPatFile(ss.ObjAttn, "obj_attn.tsv", "ObjAttn", "object-based attention")

    def ValsTsr(ss, name):
        """
        ValsTsr gets value tensor of given name, creating if not yet made
        """
        if name in ss.ValsTsrs:
            return ss.ValsTsrs[name]
        tsr = etensor.Float32()
        ss.ValsTsrs[name] = tsr
        return tsr

    def LogTstTrl(ss, dt):
        """
        LogTstTrl adds data from current trial to the TstTrlLog table.
        log always contains number of testing items
        """
        row = dt.Rows
        if dt.Rows <= row:
            dt.SetNumRows(row + 1)

        trl = row % 3

        dt.SetCellFloat("Trial", row, float(trl))
        dt.SetCellString("TrialName", row, ss.TestEnv.GroupName.Cur)
        dt.SetCellFloat("Cycle", row, float(ss.Time.Cycle))

        for lnm in ss.TstRecLays :
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

        nt = ss.TestEnv.Table.Len()
        sch = etable.Schema(
            [etable.Column("Trial", etensor.INT64, go.nil, go.nil),
            etable.Column("TrialName", etensor.STRING, go.nil, go.nil),
            etable.Column("Cycle", etensor.INT64, go.nil, go.nil)]
        )
        for lnm in ss.TstRecLays:
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            sch.append( etable.Column(lnm, etensor.FLOAT64, ly.Shp.Shp, go.nil))
        dt.SetFromSchema(sch, nt)

    def ConfigTstTrlPlot(ss, plt, dt):
        plt.Params.Title = "Attn Test Trial Plot"
        plt.Params.XAxisCol = "Trial"
        plt.SetTable(dt)
        plt.Params.Points = True

        plt.SetColParams("Trial", eplot.Off, eplot.FixMin, -0.5, eplot.FixMax, 2.5)
        plt.SetColParams("TrialName", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Cycle", eplot.On, eplot.FixMin, 0, eplot.FixMax, 220)

        for lnm in ss.TstRecLays :
            cp = plt.SetColParams(lnm, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
            cp.TensorIdx = -1
        return plt

    def TestStats(ss):
        dt = ss.TstTrlLog
        runix = etable.NewIdxView(dt)
        spl = split.GroupBy(runix, go.Slice_string(["TrialName"]))
        split.Desc(spl, "Cycle")
        ss.TstStats = spl.AggsToTable(etable.AddAggName)

    def ConfigNetView(ss, nv):
        nv.ViewDefaults()
        nv.Scene().Camera.Pose.Pos.Set(0, 1.2, 3.0) # more "head on" than default which is more "top down"
        nv.Scene().Camera.LookAt(mat32.Vec3(0, 0, 0), mat32.Vec3(0, 1, 0))
        nv.SetMaxRecs(1100)

    def ConfigGui(ss):
        """
        ConfigGui configures the GoGi gui interface for this simulation,
        """
        width = 1600
        height = 1200

        gi.SetAppName("attn")
        gi.SetAppAbout('attn: This simulation illustrates how object recognition (ventral, what) and spatial (dorsal, where) pathways interact to produce spatial attention effects, and accurately capture the effects of brain damage to the spatial pathway. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch6/attn/README.md">README.md on GitHub</a>.</p>')

        win = gi.NewMainWindow("attn", "Attention", width, height)
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
        tv.AddTab(plt, "TstTrlPlot")
        ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

        split.SetSplitsList(go.Slice_float32([.2, .8]))

        recv = win.This()
        
        tbar.AddAction(gi.ActOpts(Label="Init", Icon="update", Tooltip="Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc=UpdtFuncNotRunning), recv, InitCB)

        tbar.AddAction(gi.ActOpts(Label="Stop", Icon="stop", Tooltip="Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc=UpdtFuncRunning), recv, StopCB)
        
        tbar.AddAction(gi.ActOpts(Label="Test Trial", Icon="step-fwd", Tooltip="Runs the next testing trial.", UpdateFunc=UpdtFuncNotRunning), recv, TestTrialCB)

        tbar.AddAction(gi.ActOpts(Label="Test All", Icon="step-fwd", Tooltip="Runs all testing trials.", UpdateFunc=UpdtFuncNotRunning), recv, TestAllCB)
        
        tbar.AddSeparator("log")
        
        tbar.AddAction(gi.ActOpts(Label= "Lesion", Icon= "cut", Tooltip= "Lesion spatial pathways.", UpdateFunc=UpdtFuncNotRunning), recv, LesionCB)

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

