// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
attn: This simulation illustrates how object recognition (ventral, what) and
spatial (dorsal, where) pathways interact to produce spatial attention
effects, and accurately capture the effects of brain damage to the
spatial pathway.
*/
package main

import (
	"bytes"
	"fmt"
	"log"
	"strconv"
	"strings"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/split"
	"github.com/emer/leabra/leabra"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// this is the stub main for gogi that calls our actual mainrun function, at end of file
func main() {
	gimain.Main(func() {
		mainrun()
	})
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// TestType is the type of testing patterns
type TestType int32

//go:generate stringer -type=TestType

var KiT_TestType = kit.Enums.AddEnum(TestTypeN, kit.NotBitFlag, nil)

func (ev TestType) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *TestType) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

const (
	MultiObjs TestType = iota
	StdPosner
	ClosePosner
	ReversePosner
	ObjAttn
	TestTypeN
)

// LesionType is the type of lesion
type LesionType int32

//go:generate stringer -type=LesionType

var KiT_LesionType = kit.Enums.AddEnum(LesionTypeN, kit.NotBitFlag, nil)

func (ev LesionType) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *LesionType) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

const (
	NoLesion LesionType = iota
	LesionSpat1
	LesionSpat2
	LesionSpat12
	LesionTypeN
)

// LesionSize is the size of lesion
type LesionSize int32

//go:generate stringer -type=LesionSize

var KiT_LesionSize = kit.Enums.AddEnum(LesionSizeN, kit.NotBitFlag, nil)

func (ev LesionSize) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *LesionSize) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

const (
	LesionHalf LesionSize = iota
	LesionFull
	LesionSizeN
)

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "no learning",
				Params: params.Params{
					"Prjn.Learn.Learn": "false",
					"Prjn.WtInit.Mean": "0.5",
					"Prjn.WtInit.Var":  "0",
					"Prjn.WtInit.Sym":  "false", // for lesions, just in case
				}},
			{Sel: "Layer", Desc: "fix expected activity levels, reduce leak",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init":  "0.05",
					"Layer.Inhib.ActAvg.Fixed": "true",
					"Layer.Inhib.Layer.FBTau":  "3",   // slower better for small nets
					"Layer.Act.Gbar.L":         "0.1", // needs lower leak
					"Layer.Act.Dt.VmTau":       "7",   // slower
					"Layer.Act.Dt.GTau":        "3",   // slower
					"Layer.Act.Noise.Dist":     "Gaussian",
					"Layer.Act.Noise.Var":      "0.001",
					"Layer.Act.Noise.Type":     "GeNoise",
					"Layer.Act.Noise.Fixed":    "false",
					"Layer.Act.Init.Decay":     "0",
					"Layer.Act.KNa.On":         "false",
				}},
			{Sel: "#Input", Desc: "no noise",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "2.0",
					"Layer.Act.Noise.Type":    "NoNoise",
					"Layer.Inhib.ActAvg.Init": "0.07",
				}},
			{Sel: "#V1", Desc: "specific inhibition",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "2.0",
				}},
			{Sel: ".Object", Desc: "specific inhbition",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "0.85",
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Pool.Gi":     "1",
					"Layer.Inhib.Pool.FB":     "0.5", // presumably important
					"Layer.Inhib.ActAvg.Init": "0.1",
				}},
			{Sel: ".Spatial", Desc: "specific inhbition",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "1",
					"Layer.Inhib.ActAvg.Init": "0.4",
				}},
			{Sel: "#Spat2", Desc: "specific inhbition",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.6667",
				}},
			{Sel: "#Output", Desc: "specific inhbition",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "1.8",
					"Layer.Inhib.ActAvg.Init": "0.5",
				}},
			{Sel: ".Back", Desc: "all top-downs",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.25",
				}},
			{Sel: ".Lateral", Desc: "spatial self",
				Params: params.Params{
					"Prjn.WtScale.Abs": "0.4",
				}},
			{Sel: ".SpatToObj", Desc: "spatial to obj",
				Params: params.Params{
					"Prjn.WtScale.Rel": "2", // note: controlled by Sim param
				}},
			{Sel: ".ObjToSpat", Desc: "obj to spatial",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.5",
				}},
			{Sel: "#InputToV1", Desc: "wt scale",
				Params: params.Params{
					"Prjn.WtScale.Rel": "3",
				}},
			{Sel: "#V1ToSpat1", Desc: "wt scale",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.6", // note: controlled by Sim param
				}},
			{Sel: "#Spat1ToV1", Desc: "stronger spatial top-down wt scale -- key param for invalid effect",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.4",
				}},
			{Sel: "#Spat2ToSpat1", Desc: "stronger spatial top-down wt scale -- key param for invalid effect",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.4",
				}},
		},
	}},
	{Name: "KNaAdapt", Desc: "Turn on KNa adaptation", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "KNa adapt on",
				Params: params.Params{
					"Layer.Act.KNa.On": "true",
				}},
		},
	}},
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	SpatToObj     float32           `def:"2" desc:"spatial to object projection WtScale.Rel strength -- reduce to 1.5, 1 to test"`
	V1ToSpat1     float32           `def:"0.6" desc:"V1 to Spat1 projection WtScale.Rel strength -- reduce to .55, .5 to test"`
	KNaAdapt      bool              `def:"false" desc:"sodium (Na) gated potassium (K) channels that cause neurons to fatigue over time"`
	CueDur        int               `def:"100" desc:"number of cycles to present the cue -- 100 by default, 50 to 300 for KNa adapt testing"`
	Net           *leabra.Network   `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Test          TestType          `desc:"select which type of test (input patterns) to use"`
	MultiObjs     *etable.Table     `view:"no-inline" desc:"click to see these testing input patterns"`
	StdPosner     *etable.Table     `view:"no-inline" desc:"click to see these testing input patterns"`
	ClosePosner   *etable.Table     `view:"no-inline" desc:"click to see these testing input patterns"`
	ReversePosner *etable.Table     `view:"no-inline" desc:"click to see these testing input patterns"`
	ObjAttn       *etable.Table     `view:"no-inline" desc:"click to see these testing input patterns"`
	TstTrlLog     *etable.Table     `view:"no-inline" desc:"testing trial-level log data -- click to see record of network's response to each input"`
	TstStats      *etable.Table     `view:"no-inline" desc:"aggregate stats on testing data"`
	Params        params.Sets       `view:"no-inline" desc:"full collection of param sets -- not really interesting for this model"`
	ParamSet      string            `view:"-" desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set -- can use multiple names separated by spaces (don't put spaces in ParamSet names!)"`
	TestEnv       env.FixedTable    `desc:"Testing environment -- manages iterating over testing"`
	Time          leabra.Time       `desc:"leabra timing parameters and state"`
	ViewUpdt      leabra.TimeScales `desc:"at what time scale to update the display during testing?  Change to AlphaCyc to make display updating go faster"`
	TstRecLays    []string          `desc:"names of layers to record activations etc of during testing"`

	// internal state - view:"-"
	Win        *gi.Window                  `view:"-" desc:"main GUI window"`
	NetView    *netview.NetView            `view:"-" desc:"the network viewer"`
	ToolBar    *gi.ToolBar                 `view:"-" desc:"the master toolbar"`
	TstTrlPlot *eplot.Plot2D               `view:"-" desc:"the test-trial plot"`
	ValsTsrs   map[string]*etensor.Float32 `view:"-" desc:"for holding layer values"`
	IsRunning  bool                        `view:"-" desc:"true if sim is running"`
	StopNow    bool                        `view:"-" desc:"flag to stop running"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Defaults()
	ss.Net = &leabra.Network{}
	ss.Test = MultiObjs
	ss.MultiObjs = &etable.Table{}
	ss.StdPosner = &etable.Table{}
	ss.ClosePosner = &etable.Table{}
	ss.ReversePosner = &etable.Table{}
	ss.ObjAttn = &etable.Table{}
	ss.TstTrlLog = &etable.Table{}
	ss.TstStats = &etable.Table{}
	ss.Params = ParamSets
	ss.ViewUpdt = leabra.FastSpike
	ss.TstRecLays = []string{"Input", "V1", "Spat1", "Spat2", "Obj1", "Obj2", "Output"}
	ss.Defaults()
}

// Defaults sets default params
func (ss *Sim) Defaults() {
	ss.SpatToObj = 2
	ss.V1ToSpat1 = 0.6
	ss.KNaAdapt = false
	ss.CueDur = 100
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.OpenPats()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
}

func (ss *Sim) ConfigEnv() {
	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Dsc = "testing params and state"
	ss.TestEnv.Table = etable.NewIdxView(ss.MultiObjs)
	ss.TestEnv.Sequential = true
	ss.TestEnv.Validate()
	ss.TestEnv.Init(0)
}

func (ss *Sim) UpdateEnv() {
	switch ss.Test {
	case MultiObjs:
		ss.TestEnv.Table = etable.NewIdxView(ss.MultiObjs)
	case StdPosner:
		ss.TestEnv.Table = etable.NewIdxView(ss.StdPosner)
	case ClosePosner:
		ss.TestEnv.Table = etable.NewIdxView(ss.ClosePosner)
	case ReversePosner:
		ss.TestEnv.Table = etable.NewIdxView(ss.ReversePosner)
	case ObjAttn:
		ss.TestEnv.Table = etable.NewIdxView(ss.ObjAttn)
	}
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.InitName(net, "AttnNet")
	inp := net.AddLayer4D("Input", 1, 7, 2, 1, emer.Input)
	v1 := net.AddLayer4D("V1", 1, 7, 2, 1, emer.Hidden)
	sp1 := net.AddLayer4D("Spat1", 1, 5, 2, 1, emer.Hidden)
	sp2 := net.AddLayer4D("Spat2", 1, 3, 2, 1, emer.Hidden)
	ob1 := net.AddLayer4D("Obj1", 1, 5, 2, 1, emer.Hidden)
	out := net.AddLayer2D("Output", 2, 1, emer.Compare)
	ob2 := net.AddLayer4D("Obj2", 1, 3, 2, 1, emer.Hidden)

	ob1.SetClass("Object")
	ob2.SetClass("Object")
	sp1.SetClass("Spatial")
	sp2.SetClass("Spatial")

	full := prjn.NewFull()
	net.ConnectLayers(inp, v1, prjn.NewOneToOne(), emer.Forward)

	rec3sp := prjn.NewRect()
	rec3sp.Size.Set(3, 2)
	rec3sp.Scale.Set(1, 0)
	rec3sp.Start.Set(0, 0)

	rec3sptd := prjn.NewRect()
	rec3sptd.Size.Set(3, 2)
	rec3sptd.Scale.Set(1, 0)
	rec3sptd.Start.Set(-2, 0)
	rec3sptd.Wrap = false

	v1sp1, sp1v1 := net.BidirConnectLayers(v1, sp1, full)
	v1sp1.SetPattern(rec3sp)
	sp1v1.SetPattern(rec3sptd)

	sp1sp2, sp2sp1 := net.BidirConnectLayers(sp1, sp2, full)
	sp1sp2.SetPattern(rec3sp)
	sp2sp1.SetPattern(rec3sptd)

	rec3ob := prjn.NewRect()
	rec3ob.Size.Set(3, 1)
	rec3ob.Scale.Set(1, 1)
	rec3ob.Start.Set(0, 0)

	rec3obtd := prjn.NewRect()
	rec3obtd.Size.Set(3, 1)
	rec3obtd.Scale.Set(1, 1)
	rec3obtd.Start.Set(-2, 0)
	rec3obtd.Wrap = false

	v1ob1, ob1v1 := net.BidirConnectLayers(v1, ob1, full)
	v1ob1.SetPattern(rec3ob)
	ob1v1.SetPattern(rec3obtd)

	ob1ob2, ob2ob1 := net.BidirConnectLayers(ob1, ob2, full)
	ob1ob2.SetPattern(rec3ob)
	ob2ob1.SetPattern(rec3obtd)

	recout := prjn.NewRect()
	recout.Size.Set(1, 1)
	recout.Scale.Set(0, 1)
	recout.Start.Set(0, 0)

	ob2out, outob2 := net.BidirConnectLayers(ob2, out, full)
	ob2out.SetPattern(rec3ob)
	outob2.SetPattern(recout)

	// between pathways
	p1to1 := prjn.NewPoolOneToOne()
	spob1, obsp1 := net.BidirConnectLayers(sp1, ob1, p1to1)
	spob2, obsp2 := net.BidirConnectLayers(sp2, ob2, p1to1)

	spob1.SetClass("SpatToObj")
	spob2.SetClass("SpatToObj")
	obsp1.SetClass("ObjToSpat")
	obsp2.SetClass("ObjToSpat")

	// self cons
	rec1slf := prjn.NewRect()
	rec1slf.Size.Set(1, 2)
	rec1slf.Scale.Set(1, 0)
	rec1slf.Start.Set(0, 0)
	rec1slf.SelfCon = false
	net.ConnectLayers(sp1, sp1, rec1slf, emer.Lateral)
	net.ConnectLayers(sp2, sp2, rec1slf, emer.Lateral)

	sp1.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "V1", YAlign: relpos.Front, XAlign: relpos.Left, YOffset: 1})
	sp2.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Spat1", YAlign: relpos.Front, XAlign: relpos.Left, Space: 1})
	ob1.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Spat1", YAlign: relpos.Front, Space: 1})
	out.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Spat2", YAlign: relpos.Front, Space: 1})
	ob2.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Output", YAlign: relpos.Front, Space: 1})

	net.Defaults()
	ss.SetParams("Network", false) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	ss.InitWts(net)
}

// InitWts loads the saved weights
func (ss *Sim) InitWts(net *leabra.Network) {
	net.InitWts()
}

// LesionUnit lesions given unit number in given layer by setting all weights to 0
func (ss *Sim) LesionUnit(lay *leabra.Layer, unx, uny int) {
	ui := etensor.Prjn2DIdx(&lay.Shp, false, uny, unx)
	rpj := lay.RecvPrjns()
	for _, pji := range *rpj {
		pj := pji.(*leabra.Prjn)
		nc := int(pj.RConN[ui])
		st := int(pj.RConIdxSt[ui])
		for ci := 0; ci < nc; ci++ {
			rsi := pj.RSynIdx[st+ci]
			sy := &pj.Syns[rsi]
			sy.Wt = 0
			pj.Learn.LWtFmWt(sy)
		}
	}
}

// Lesion lesions given set of layers (or unlesions for NoLesion) and
// locations and number of units (Half = partial = 1/2 units, Full = both units)
func (ss *Sim) Lesion(lay LesionType, locations LesionSize, units LesionSize) {
	ss.InitWts(ss.Net)
	if lay == NoLesion {
		return
	}
	if lay == LesionSpat1 || lay == LesionSpat12 {
		sp1 := ss.Net.LayerByName("Spat1").(leabra.LeabraLayer).AsLeabra()
		ss.LesionUnit(sp1, 3, 1)
		ss.LesionUnit(sp1, 4, 1)
		if units == LesionFull {
			ss.LesionUnit(sp1, 3, 0)
			ss.LesionUnit(sp1, 4, 0)
		}
		if locations == LesionFull {
			ss.LesionUnit(sp1, 0, 1)
			ss.LesionUnit(sp1, 1, 1)
			ss.LesionUnit(sp1, 2, 1)
			if units == LesionFull {
				ss.LesionUnit(sp1, 0, 0)
				ss.LesionUnit(sp1, 1, 0)
				ss.LesionUnit(sp1, 2, 0)
			}
		}
	}
	if lay == LesionSpat2 || lay == LesionSpat12 {
		sp2 := ss.Net.LayerByName("Spat2").(leabra.LeabraLayer).AsLeabra()
		ss.LesionUnit(sp2, 2, 1)
		if units == LesionFull {
			ss.LesionUnit(sp2, 2, 0)
		}
		if locations == LesionFull {
			ss.LesionUnit(sp2, 0, 1)
			ss.LesionUnit(sp2, 1, 1)
			if units == LesionFull {
				ss.LesionUnit(sp2, 0, 0)
				ss.LesionUnit(sp2, 1, 0)
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.UpdateEnv()
	ss.TestEnv.Init(0)
	ss.Time.Reset()
	ss.Time.CycPerQtr = 55 // 220 total
	// ss.InitWts(ss.Net)
	ss.StopNow = false
	ss.SetParams("", false) // all sheets
	ss.TstTrlLog.SetNumRows(0)
	ss.UpdateView()
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters() string {
	nm := ss.TestEnv.GroupName.Cur
	if ss.TestEnv.TrialName.Cur != nm {
		nm += ": " + ss.TestEnv.TrialName.Cur
	}
	return fmt.Sprintf("Trial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t", ss.TestEnv.Trial.Cur, ss.Time.Cycle, nm)
}

func (ss *Sim) UpdateView() {
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters())
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate() // note: using counters is significantly slower..
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// AlphaCyc runs one alpha-cycle (100 msec, 4 quarters)			 of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// Handles netview updating within scope of AlphaCycle
func (ss *Sim) AlphaCyc() {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdt := ss.ViewUpdt

	// note: this has no learning calls
	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()

	ss.Net.AlphaCycInit(false)
	ss.Time.AlphaCycStart()
	overThresh := false
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			ss.Time.CycleInc()
			switch viewUpdt {
			case leabra.Cycle:
				if cyc != ss.Time.CycPerQtr-1 { // will be updated by quarter
					ss.UpdateView()
				}
			case leabra.FastSpike:
				if (cyc+1)%10 == 0 {
					ss.UpdateView()
				}
			}
			trgact := out.Neurons[1].Act
			if trgact > 0.5 {
				overThresh = true
				break
			}
		}
		ss.Net.QuarterFinal(&ss.Time)
		ss.Time.QuarterInc()
		switch {
		case viewUpdt <= leabra.Quarter:
			ss.UpdateView()
		case viewUpdt == leabra.Phase:
			if qtr >= 2 {
				ss.UpdateView()
			}
		}
		if overThresh {
			break
		}
	}

	ss.UpdateView()
}

// AlphaCycCue just runs over fixed number of cycles -- for Cue trials
func (ss *Sim) AlphaCycCue() {
	ss.Net.AlphaCycInit(false)
	ss.Time.AlphaCycStart()
	for cyc := 0; cyc < ss.CueDur; cyc++ {
		ss.Net.Cycle(&ss.Time)
		ss.Time.CycleInc()
		if (cyc+1)%10 == 0 {
			ss.UpdateView()
		}
	}
	ss.Net.QuarterFinal(&ss.Time) // whatever.
	ss.Time.QuarterInc()

	ss.UpdateView()
}

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"Input", "Output"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := en.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.StopNow = true
}

// Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
func (ss *Sim) Stopped() {
	ss.IsRunning = false
	if ss.Win != nil {
		vp := ss.Win.WinViewport2D()
		if ss.ToolBar != nil {
			ss.ToolBar.UpdateActions()
		}
		vp.SetNeedsFullRender()
	}
}

// SaveWts saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWts(filename gi.FileName) {
	ss.Net.SaveWtsJSON(filename)
}

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial() {
	ss.TestEnv.Step()

	// Query counters FIRST
	_, _, chg := ss.TestEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewUpdt > leabra.AlphaCycle {
			ss.UpdateView()
		}
		return
	}

	isCue := (ss.TestEnv.TrialName.Cur == "Cue")

	if ss.TestEnv.TrialName.Prv != "Cue" {
		ss.Net.InitActs()
	}
	ss.ApplyInputs(&ss.TestEnv)
	if isCue {
		ss.AlphaCycCue()
	} else {
		ss.AlphaCyc()
		ss.LogTstTrl(ss.TstTrlLog)
	}
}

// TestTrialGUI runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrialGUI() {
	ss.TestTrial()
	ss.Stopped()
}

// TestItem tests given item which is at given index in test item list
func (ss *Sim) TestItem(idx int) {
	cur := ss.TestEnv.Trial.Cur
	ss.TestEnv.Trial.Cur = idx
	ss.TestEnv.SetTrialName()
	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc() // !train
	ss.TestEnv.Trial.Cur = cur
}

// TestItemGUI tests given item which is at given index in test item list
func (ss *Sim) TestItemGUI(idx int) {
	ss.TestItem(idx)
	ss.Stopped()
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.SetParams("", false) // in case params were changed
	ss.UpdateEnv()
	ss.TestEnv.Init(0)
	for {
		ss.TestTrial()
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}
	ss.TestStats()
}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.StopNow = false
	ss.TestAll()
	ss.Stopped()
}

/////////////////////////////////////////////////////////////////////////
//   Params setting

// SetParams sets the params for "Base" and then current ParamSet.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParams(sheet string, setMsg bool) error {
	if sheet == "" {
		// this is important for catching typos and ensuring that all sheets can be used
		ss.Params.ValidateSheets([]string{"Network", "Sim"})
	}

	spo := ss.Params.SetByName("Base").SheetByName("Network").SelByName(".SpatToObj")
	spo.Params.SetParamByName("Prjn.WtScale.Rel", fmt.Sprintf("%g", ss.SpatToObj))

	vsp := ss.Params.SetByName("Base").SheetByName("Network").SelByName("#V1ToSpat1")
	vsp.Params.SetParamByName("Prjn.WtScale.Rel", fmt.Sprintf("%g", ss.V1ToSpat1))

	err := ss.SetParamsSet("Base", sheet, setMsg)
	if ss.ParamSet != "" && ss.ParamSet != "Base" {
		sps := strings.Fields(ss.ParamSet)
		for _, ps := range sps {
			err = ss.SetParamsSet(ps, sheet, setMsg)
		}
	}

	if ss.KNaAdapt {
		err = ss.SetParamsSet("KNaAdapt", sheet, setMsg)
	}
	return err
}

// SetParamsSet sets the params for given params.Set name.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParamsSet(setNm string, sheet string, setMsg bool) error {
	pset, err := ss.Params.SetByNameTry(setNm)
	if err != nil {
		return err
	}
	if sheet == "" || sheet == "Network" {
		netp, ok := pset.Sheets["Network"]
		if ok {
			ss.Net.ApplyParams(netp, setMsg)
		}
	}

	if sheet == "" || sheet == "Sim" {
		simp, ok := pset.Sheets["Sim"]
		if ok {
			simp.Apply(ss, setMsg)
		}
	}
	// note: if you have more complex environments with parameters, definitely add
	// sheets for them, e.g., "TrainEnv", "TestEnv" etc
	return err
}

// OpenPatAsset opens pattern file from embedded assets
func (ss *Sim) OpenPatAsset(dt *etable.Table, fnm, name, desc string) error {
	dt.SetMetaData("name", name)
	dt.SetMetaData("desc", desc)
	ab, err := Asset(fnm)
	if err != nil {
		log.Println(err)
		return err
	}
	err = dt.ReadCSV(bytes.NewBuffer(ab), etable.Tab)
	if err != nil {
		log.Println(err)
	} else {
		for i := 1; i < len(dt.Cols); i++ {
			dt.Cols[i].SetMetaData("grid-fill", "0.9")
		}
	}
	return err
}

func (ss *Sim) OpenPats() {
	// patgen.ReshapeCppFile(ss.MultiObjs, "MultiObjs.dat", "multi_objs.tsv")             // one-time reshape
	// patgen.ReshapeCppFile(ss.StdPosner, "StdPosner.dat", "std_posner.tsv")             // one-time reshape
	// patgen.ReshapeCppFile(ss.ClosePosner, "ClosePosner.dat", "close_posner.tsv")       // one-time reshape
	// patgen.ReshapeCppFile(ss.ReversePosner, "ReversePosner.dat", "reverse_posner.tsv") // one-time reshape
	// patgen.ReshapeCppFile(ss.ObjAttn, "ObjAttn.dat", "obj_attn.tsv")                   // one-time reshape

	ss.OpenPatAsset(ss.MultiObjs, "multi_objs.tsv", "MultiObjs", "multiple object filtering")
	ss.OpenPatAsset(ss.StdPosner, "std_posner.tsv", "StdPosner", "standard Posner spatial cuing task")
	ss.OpenPatAsset(ss.ClosePosner, "close_posner.tsv", "ClosePosner", "close together Posner spatial cuing task")
	ss.OpenPatAsset(ss.ReversePosner, "reverse_posner.tsv", "ReversePosner", "reverse position Posner spatial cuing task")
	ss.OpenPatAsset(ss.ObjAttn, "obj_attn.tsv", "ObjAttn", "object-based attention")
}

//////////////////////////////////////////////
//  TstTrlLog

// ValsTsr gets value tensor of given name, creating if not yet made
func (ss *Sim) ValsTsr(name string) *etensor.Float32 {
	if ss.ValsTsrs == nil {
		ss.ValsTsrs = make(map[string]*etensor.Float32)
	}
	tsr, ok := ss.ValsTsrs[name]
	if !ok {
		tsr = &etensor.Float32{}
		ss.ValsTsrs[name] = tsr
	}
	return tsr
}

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *etable.Table) {
	row := dt.Rows
	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	trl := row % 3 // every third item is a new trial type

	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("TrialName", row, ss.TestEnv.GroupName.Cur)
	dt.SetCellFloat("Cycle", row, float64(ss.Time.Cycle))

	for _, lnm := range ss.TstRecLays {
		tsr := ss.ValsTsr(lnm)
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		ly.UnitValsTensor(tsr, "Act")
		dt.SetCellTensor(lnm, row, tsr)
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TstTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTstTrlLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TestEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"Cycle", etensor.INT64, nil, nil},
	}
	for _, lnm := range ss.TstRecLays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		sch = append(sch, etable.Column{lnm, etensor.FLOAT64, ly.Shp.Shp, nil})
	}
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Attn Test Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	plt.Params.Points = true
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Trial", eplot.Off, eplot.FixMin, -0.5, eplot.FixMax, 2.5)
	plt.SetColParams("TrialName", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Cycle", eplot.On, eplot.FixMin, 0, eplot.FixMax, 220)

	for _, lnm := range ss.TstRecLays {
		cp := plt.SetColParams(lnm, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
		cp.TensorIdx = -1 // plot all
	}
	return plt
}

func (ss *Sim) TestStats() {
	dt := ss.TstTrlLog
	runix := etable.NewIdxView(dt)
	spl := split.GroupBy(runix, []string{"TrialName"})
	split.Desc(spl, "Cycle")
	ss.TstStats = spl.AggsToTable(etable.AddAggName)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.ViewDefaults()
	nv.Scene().Camera.Pose.Pos.Set(0, 1.2, 3.0) // more "head on" than default which is more "top down"
	nv.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
	nv.SetMaxRecs(1100)
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("attn")
	gi.SetAppAbout(`attn: This simulation illustrates how object recognition (ventral, what) and spatial (dorsal, where) pathways interact to produce spatial attention effects, and accurately capture the effects of brain damage to the spatial pathway. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch6/attn/README.md">README.md on GitHub</a>.</p>`)

	win := gi.NewMainWindow("attn", "Attention", width, height)
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = mat32.X
	split.SetStretchMax()

	sv := giv.AddNewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := gi.AddNewTabView(split, "tv")

	nv := tv.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	nv.Var = "Act"
	nv.SetNet(ss.Net)
	ss.NetView = nv
	ss.ConfigNetView(nv)

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TstTrlPlot").(*eplot.Plot2D)
	ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

	split.SetSplits(.2, .8)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Stop()
	})

	tbar.AddAction(gi.ActOpts{Label: "Test Trial", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TestTrialGUI()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Test All", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestAll()
		}
	})

	tbar.AddSeparator("msep")

	tbar.AddAction(gi.ActOpts{Label: "Lesion", Icon: "cut", Tooltip: "Lesion spatial pathways.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		giv.CallMethod(ss, "Lesion", vp)
	})

	tbar.AddAction(gi.ActOpts{Label: "Defaults", Icon: "update", Tooltip: "Restore default parameters.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Defaults()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch6/attn/README.md")
		})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(win)

	inQuitPrompt := false
	gi.SetQuitReqFunc(func() {
		if inQuitPrompt {
			return
		}
		inQuitPrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Quit?",
			Prompt: "Are you <i>sure</i> you want to quit and lose any unsaved params, weights, logs, etc?"}, gi.AddOk, gi.AddCancel,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inQuitPrompt = false
				}
			})
	})

	// gi.SetQuitCleanFunc(func() {
	// 	fmt.Printf("Doing final Quit cleanup here..\n")
	// })

	inClosePrompt := false
	win.SetCloseReqFunc(func(w *gi.Window) {
		if inClosePrompt {
			return
		}
		inClosePrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Close Window?",
			Prompt: "Are you <i>sure</i> you want to close the window?  This will Quit the App as well, losing all unsaved params, weights, logs, etc"}, gi.AddOk, gi.AddCancel,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inClosePrompt = false
				}
			})
	})

	win.SetCloseCleanFunc(func(w *gi.Window) {
		go gi.Quit() // once main window is closed, quit
	})

	win.MainMenuUpdated()
	return win
}

// These props register Save methods so they can be used
var SimProps = ki.Props{
	"CallMethods": ki.PropSlice{
		{"SaveWts", ki.Props{
			"desc": "save network weights to file",
			"icon": "file-save",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".wts",
				}},
			},
		}},
		{"Lesion", ki.Props{
			"desc": "lesions given set of layers (or unlesions for NoLesion) and locations and number of units (Half = partial = 1/2 units, Full = both units)",
			"icon": "cut",
			"Args": ki.PropSlice{
				{"Layers", ki.Props{}},
				{"Locations", ki.Props{}},
				{"Units", ki.Props{}},
			},
		}},
	},
}

func mainrun() {
	TheSim.New()
	TheSim.Config()

	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}
