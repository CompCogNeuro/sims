// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// attn: This simulation illustrates how object recognition (ventral, what) and
// spatial (dorsal, where) pathways interact to produce spatial attention
// effects, and accurately capture the effects of brain damage to the
// spatial pathway.
package main

//go:generate core generate -add-types

import (
	"embed"
	"fmt"
	"math"
	"strings"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/randx"
	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tensor/table"
	"cogentcore.org/core/tree"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/estats"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/netview"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/leabra/v2/leabra"
)

//go:embed multi_objs.tsv std_posner.tsv close_posner.tsv reverse_posner.tsv obj_attn.tsv
var content embed.FS

// TestType is the type of testing patterns
type TestType int32 //enums:enum

const (
	MultiObjs TestType = iota
	StdPosner
	ClosePosner
	ReversePosner
	ObjAttn
)

// LesionType is the type of lesion
type LesionType int32 //enums:enum

const (
	NoLesion LesionType = iota
	LesionSpat1
	LesionSpat2
	LesionSpat12
)

// LesionSize is the size of lesion
type LesionSize int32 //enums:enum

const (
	LesionHalf LesionSize = iota
	LesionFull
)

func main() {
	sim := &Sim{}
	sim.New()
	sim.ConfigAll()
	sim.RunGUI()
}

// ParamSets is the default set of parameters.
// Base is always applied, and others can be optionally
// selected to apply on top of that.
var ParamSets = params.Sets{
	"Base": {
		{Sel: "Path", Desc: "no learning",
			Params: params.Params{
				"Path.Learn.Learn": "false",
				"Path.WtInit.Mean": "0.5",
				"Path.WtInit.Var":  "0",
				"Path.WtInit.Sym":  "false", // for lesions, just in case
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
		{Sel: ".BackPath", Desc: "all top-downs",
			Params: params.Params{
				"Path.WtScale.Rel": "0.25",
			}},
		{Sel: ".LateralPath", Desc: "spatial self",
			Params: params.Params{
				"Path.WtScale.Abs": "0.4",
			}},
		{Sel: ".SpatToObj", Desc: "spatial to obj",
			Params: params.Params{
				"Path.WtScale.Rel": "2", // note: controlled by Sim param
			}},
		{Sel: ".ObjToSpat", Desc: "obj to spatial",
			Params: params.Params{
				"Path.WtScale.Rel": "0.5",
			}},
		{Sel: "#InputToV1", Desc: "wt scale",
			Params: params.Params{
				"Path.WtScale.Rel": "3",
			}},
		{Sel: "#V1ToSpat1", Desc: "wt scale",
			Params: params.Params{
				"Path.WtScale.Rel": "0.6", // note: controlled by Sim param
			}},
		{Sel: "#Spat1ToV1", Desc: "stronger spatial top-down wt scale -- key param for invalid effect",
			Params: params.Params{
				"Path.WtScale.Rel": "0.4",
			}},
		{Sel: "#Spat2ToSpat1", Desc: "stronger spatial top-down wt scale -- key param for invalid effect",
			Params: params.Params{
				"Path.WtScale.Rel": "0.4",
			}},
	},
	"KNaAdapt": {
		{Sel: "Layer", Desc: "KNa adapt on",
			Params: params.Params{
				"Layer.Act.KNa.On": "true",
			}},
	},
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	// select which type of test (input patterns) to use
	Test TestType

	// spatial to object projection WtScale.Rel strength -- reduce to 1.5, 1 to test
	SpatToObj float32 `default:"2"`

	// V1 to Spat1 projection WtScale.Rel strength -- reduce to .55, .5 to test
	V1ToSpat1 float32 `default:"0.6"`

	// sodium (Na) gated potassium (K) channels that cause neurons to fatigue over time
	KNaAdapt bool `default:"false"`

	// number of cycles to present the cue; 100 by default, 50 to 300 for KNa adapt testing
	CueCycles int `default:"100"`

	// number of cycles to present a target; 220 by default, 50 to 300 for KNa adapt testing
	TargetCycles int `default:"220"`

	// click to see these testing input patterns
	MultiObjs *table.Table `view:"no-inline"`

	// click to see these testing input patterns
	StdPosner *table.Table `view:"no-inline"`

	// click to see these testing input patterns
	ClosePosner *table.Table `view:"no-inline"`

	// click to see these testing input patterns
	ReversePosner *table.Table `view:"no-inline"`

	// click to see these testing input patterns
	ObjAttn *table.Table `view:"no-inline"`

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *leabra.Network `new-window:"+" display:"no-inline"`

	// network parameter management
	Params emer.NetParams `display:"add-fields"`

	// contains looper control loops for running sim
	Loops *looper.Manager

	// contains computed statistic values
	Stats estats.Stats `display:"-"`

	// Contains all the logs and information about the logs.'
	Logs elog.Logs `display:"+"`

	// Environments
	Envs env.Envs `new-window:"+" display:"no-inline"`

	// leabra timing parameters and state
	Context leabra.Context `display:"-"`

	// netview update parameters
	ViewUpdate netview.ViewUpdate `display:"add-fields"`

	// manages all the gui elements
	GUI egui.GUI `display:"-"`

	// a list of random seeds to use for each run
	RandSeeds randx.Seeds `display:"-"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Defaults()
	ss.MultiObjs = &table.Table{}
	ss.StdPosner = &table.Table{}
	ss.ClosePosner = &table.Table{}
	ss.ReversePosner = &table.Table{}
	ss.ObjAttn = &table.Table{}
	ss.Net = leabra.NewNetwork("Attn")
	ss.Params.Config(ParamSets, "", "", ss.Net)
	ss.Stats.Init()
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
}

func (ss *Sim) Defaults() {
	ss.SpatToObj = 2
	ss.V1ToSpat1 = 0.6
	ss.KNaAdapt = false
	ss.CueCycles = 100
	ss.TargetCycles = 220
}

//////////////////////////////////////////////////////////////////////////////
// 		Configs

// ConfigAll configures all the elements using the standard functions
func (ss *Sim) ConfigAll() {
	ss.OpenPatterns()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
}

// OpenPatternAsset opens pattern file from embedded assets
func (ss *Sim) OpenPatternAsset(dt *table.Table, fnm, name, desc string) error {
	dt.SetMetaData("name", name)
	dt.SetMetaData("desc", desc)
	err := dt.OpenFS(content, fnm, table.Tab)
	if errors.Log(err) == nil {
		for i := 1; i < len(dt.Columns); i++ {
			dt.Columns[i].SetMetaData("grid-fill", "0.9")
		}
	}
	return err
}

func (ss *Sim) OpenPatterns() {
	ss.OpenPatternAsset(ss.MultiObjs, "multi_objs.tsv", "MultiObjs", "multiple object filtering")
	ss.OpenPatternAsset(ss.StdPosner, "std_posner.tsv", "StdPosner", "standard Posner spatial cuing task")
	ss.OpenPatternAsset(ss.ClosePosner, "close_posner.tsv", "ClosePosner", "close together Posner spatial cuing task")
	ss.OpenPatternAsset(ss.ReversePosner, "reverse_posner.tsv", "ReversePosner", "reverse position Posner spatial cuing task")
	ss.OpenPatternAsset(ss.ObjAttn, "obj_attn.tsv", "ObjAttn", "object-based attention")
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var tst *env.FixedTable
	if len(ss.Envs) == 0 {
		tst = &env.FixedTable{}
	} else {
		tst = ss.Envs.ByMode(etime.Test).(*env.FixedTable)
	}

	tst.Name = etime.Test.String()
	tst.Config(table.NewIndexView(ss.MultiObjs))
	tst.Sequential = true
	tst.Validate()

	tst.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(tst)
}

func (ss *Sim) UpdateEnv() {
	ev := ss.Envs.ByMode(etime.Test).(*env.FixedTable)
	switch ss.Test {
	case MultiObjs:
		ev.Table = table.NewIndexView(ss.MultiObjs)
	case StdPosner:
		ev.Table = table.NewIndexView(ss.StdPosner)
	case ClosePosner:
		ev.Table = table.NewIndexView(ss.ClosePosner)
	case ReversePosner:
		ev.Table = table.NewIndexView(ss.ReversePosner)
	case ObjAttn:
		ev.Table = table.NewIndexView(ss.ObjAttn)
	}
	ev.Init(0)
	if ss.Loops != nil {
		tst := ss.Loops.Stacks[etime.Test]
		tst.Loops[etime.Trial].Counter.Max = ev.Table.Len()
	}
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	inp := net.AddLayer4D("Input", 1, 7, 2, 1, leabra.InputLayer)
	v1 := net.AddLayer4D("V1", 1, 7, 2, 1, leabra.SuperLayer)
	sp1 := net.AddLayer4D("Spat1", 1, 5, 2, 1, leabra.SuperLayer)
	sp2 := net.AddLayer4D("Spat2", 1, 3, 2, 1, leabra.SuperLayer)
	ob1 := net.AddLayer4D("Obj1", 1, 5, 2, 1, leabra.SuperLayer)
	out := net.AddLayer2D("Output", 2, 1, leabra.CompareLayer)
	ob2 := net.AddLayer4D("Obj2", 1, 3, 2, 1, leabra.SuperLayer)

	ob1.AddClass("Object")
	ob2.AddClass("Object")
	sp1.AddClass("Spatial")
	sp2.AddClass("Spatial")

	full := paths.NewFull()
	net.ConnectLayers(inp, v1, paths.NewOneToOne(), leabra.ForwardPath)

	rec3sp := paths.NewRect()
	rec3sp.Size.Set(3, 2)
	rec3sp.Scale.Set(1, 0)
	rec3sp.Start.Set(0, 0)

	rec3sptd := paths.NewRect()
	rec3sptd.Size.Set(3, 2)
	rec3sptd.Scale.Set(1, 0)
	rec3sptd.Start.Set(-2, 0)
	rec3sptd.Wrap = false

	v1sp1, sp1v1 := net.BidirConnectLayers(v1, sp1, full)
	v1sp1.Pattern = rec3sp
	sp1v1.Pattern = rec3sptd

	sp1sp2, sp2sp1 := net.BidirConnectLayers(sp1, sp2, full)
	sp1sp2.Pattern = rec3sp
	sp2sp1.Pattern = rec3sptd

	rec3ob := paths.NewRect()
	rec3ob.Size.Set(3, 1)
	rec3ob.Scale.Set(1, 1)
	rec3ob.Start.Set(0, 0)

	rec3obtd := paths.NewRect()
	rec3obtd.Size.Set(3, 1)
	rec3obtd.Scale.Set(1, 1)
	rec3obtd.Start.Set(-2, 0)
	rec3obtd.Wrap = false

	v1ob1, ob1v1 := net.BidirConnectLayers(v1, ob1, full)
	v1ob1.Pattern = rec3ob
	ob1v1.Pattern = rec3obtd

	ob1ob2, ob2ob1 := net.BidirConnectLayers(ob1, ob2, full)
	ob1ob2.Pattern = rec3ob
	ob2ob1.Pattern = rec3obtd

	recout := paths.NewRect()
	recout.Size.Set(1, 1)
	recout.Scale.Set(0, 1)
	recout.Start.Set(0, 0)

	ob2out, outob2 := net.BidirConnectLayers(ob2, out, full)
	ob2out.Pattern = rec3ob
	outob2.Pattern = recout

	// between pathways
	p1to1 := paths.NewPoolOneToOne()
	spob1, obsp1 := net.BidirConnectLayers(sp1, ob1, p1to1)
	spob2, obsp2 := net.BidirConnectLayers(sp2, ob2, p1to1)

	spob1.AddClass("SpatToObj")
	spob2.AddClass("SpatToObj")
	obsp1.AddClass("ObjToSpat")
	obsp2.AddClass("ObjToSpat")

	// self cons
	rec1slf := paths.NewRect()
	rec1slf.Size.Set(1, 2)
	rec1slf.Scale.Set(1, 0)
	rec1slf.Start.Set(0, 0)
	rec1slf.SelfCon = false
	net.ConnectLayers(sp1, sp1, rec1slf, leabra.LateralPath)
	net.ConnectLayers(sp2, sp2, rec1slf, leabra.LateralPath)

	sp1.PlaceAbove(v1)
	sp2.PlaceAbove(sp1)
	ob1.PlaceRightOf(sp1, 1)
	out.PlaceRightOf(sp2, 1)
	ob2.PlaceRightOf(out, 1)

	net.Build()
	net.Defaults()
	ss.ApplyParams()
	ss.InitWeights(net)
}

// InitWeights initializes weights to digit 8
func (ss *Sim) InitWeights(net *leabra.Network) {
	net.InitWeights()
}

// LesionUnit lesions given unit number in given layer by setting all weights to 0
func (ss *Sim) LesionUnit(lay *leabra.Layer, unx, uny int) {
	ui := tensor.Projection2DIndex(&lay.Shape, false, uny, unx)
	for _, pj := range lay.RecvPaths {
		nc := int(pj.RConN[ui])
		st := int(pj.RConIndexSt[ui])
		for ci := 0; ci < nc; ci++ {
			rsi := pj.RSynIndex[st+ci]
			sy := &pj.Syns[rsi]
			sy.Wt = 0
			pj.Learn.LWtFromWt(sy)
		}
	}
}

// Lesion lesions given set of layers (or unlesions for NoLesion) and
// locations and number of units (Half = partial = 1/2 units, Full = both units)
func (ss *Sim) Lesion(lay LesionType, locations LesionSize, units LesionSize) {
	ss.InitWeights(ss.Net)
	if lay == NoLesion {
		return
	}
	if lay == LesionSpat1 || lay == LesionSpat12 {
		sp1 := ss.Net.LayerByName("Spat1")
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
		sp2 := ss.Net.LayerByName("Spat2")
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

func (ss *Sim) ApplyParams() {
	spo, _ := errors.Log1(ss.Params.Params.SheetByName("Base")).SelByName(".SpatToObj")
	spo.Params.SetByName("Path.WtScale.Rel", fmt.Sprintf("%g", ss.SpatToObj))

	vsp, _ := errors.Log1(ss.Params.Params.SheetByName("Base")).SelByName("#V1ToSpat1")
	vsp.Params.SetByName("Path.WtScale.Rel", fmt.Sprintf("%g", ss.V1ToSpat1))

	ss.Params.SetAll()

	if ss.KNaAdapt {
		ss.Params.SetAllSheet("KNaAdapt")
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Loops.ResetCounters()
	// ss.InitRandSeed(0)
	ss.GUI.StopNow = false
	ss.ApplyParams()
	ss.NewRun()
	ss.ViewUpdate.RecordSyns()
	ss.ViewUpdate.Update()
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	ss.RandSeeds.Set(run)
	ss.RandSeeds.Set(run, &ss.Net.Rand)
}

// CycleThresholdStop
func (ss *Sim) CycleThresholdStop() {
	trlnm := ss.Stats.String("TrialName")
	if strings.Contains(trlnm, "Cue") {
		return
	}
	cyc := ss.Loops.Stacks[etime.Test].Loops[etime.Cycle]
	out := ss.Net.LayerByName("Output")
	act := out.Neurons[1].Act
	if act > 0.5 {
		ss.Stats.SetFloat("RT", float64(cyc.Counter.Cur))
		cyc.SkipToMax()
	}
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()

	ntrls := 6
	cycles := ss.TargetCycles
	man.AddStack(etime.Test).
		AddTime(etime.Epoch, 10).
		AddTime(etime.Trial, ntrls).
		AddTime(etime.Cycle, cycles)

	leabra.LooperStdPhases(man, &ss.Context, ss.Net, cycles-25, cycles-1)
	leabra.LooperSimCycleAndLearn(man, ss.Net, &ss.Context, &ss.ViewUpdate) // std algo code

	for m, _ := range man.Stacks {
		stack := man.Stacks[m]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
		stack.Loops[etime.Cycle].Main.Add("CycleThresholdStop", func() {
			ss.CycleThresholdStop()
		})
	}

	/////////////////////////////////////////////
	// Logging

	man.AddOnEndToAll("Log", ss.Log)
	// leabra.LooperResetLogBelow(man, &ss.Logs)

	////////////////////////////////////////////
	// GUI

	leabra.LooperUpdateNetView(man, &ss.ViewUpdate, ss.Net, ss.NetViewCounters)
	leabra.LooperUpdatePlots(man, &ss.GUI)
	ss.Loops = man
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	ctx := &ss.Context
	net := ss.Net
	net.InitExt()
	ev := ss.Envs.ByMode(ctx.Mode).(*env.FixedTable)
	ev.Step()
	ss.Stats.SetFloat("RT", math.NaN())
	lays := net.LayersByType(leabra.InputLayer, leabra.TargetLayer)
	net.InitExt()
	ss.Stats.SetString("TrialName", ev.TrialName.Cur)
	ss.Stats.SetString("GroupName", ev.GroupName.Cur)
	if !strings.Contains(ev.TrialName.Prv, "Cue") {
		net.InitActs()
	}
	maxCyc := ss.TargetCycles
	if strings.Contains(ev.TrialName.Cur, "Cue") {
		maxCyc = ss.CueCycles
	}
	ss.Loops.Stacks[etime.Test].Loops[etime.Cycle].Counter.Max = maxCyc
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm)
		pats := ev.State(ly.Name)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ctx := &ss.Context
	ctx.Reset()
	ctx.Mode = etime.Test
	ss.InitWeights(ss.Net)
	ss.UpdateEnv()
	ss.InitStats()
	ss.StatCounters()
	// ss.Logs.ResetLog(etime.Test, etime.Trial)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetString("TrialName", "")
	ss.Stats.SetString("GroupName", "")
	ss.Stats.SetFloat("RT", math.NaN())
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them for ViewUpdate.Text
func (ss *Sim) StatCounters() {
	ctx := &ss.Context
	mode := ctx.Mode
	ss.Loops.Stacks[mode].CountersToStats(&ss.Stats)
	trl := ss.Stats.Int("Trial")
	ss.Stats.SetInt("Trial", trl)
	ss.Stats.SetInt("Cycle", int(ctx.Cycle))
}

func (ss *Sim) NetViewCounters(tm etime.Times) {
	if ss.ViewUpdate.View == nil {
		return
	}
	// if tm == etime.Trial {
	// 	ss.TrialStats() // get trial stats for current di
	// }
	ss.StatCounters()
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Trial", "GroupName", "TrialName", "Cycle"})
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Logs.AddCounterItems(etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.Test, etime.Trial, "GroupName")
	ss.Logs.AddStatStringItem(etime.Test, etime.Trial, "TrialName")

	ss.Logs.AddStatAggItem("RT", etime.Epoch, etime.Trial)

	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "InputLayer")

	ss.Logs.PlotItems("RT", "GroupName")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	ss.Logs.NoPlot(etime.Test, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Epoch)
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	ctx := &ss.Context
	if mode != etime.Analyze {
		ctx.Mode = mode // Also set specifically in a Loop callback.
	}
	dt := ss.Logs.Table(mode, time)
	if dt == nil {
		return
	}
	row := dt.Rows
	ss.StatCounters()

	switch {
	case time == etime.Trial:
		ss.StatCounters()
		if !strings.Contains(ss.Stats.String("TrialName"), "Cue") {
			if math.IsNaN(ss.Stats.Float("RT")) { // didn't stop
				ss.Stats.SetFloat("RT", float64(ss.TargetCycles))
			}
			ss.Logs.Log(mode, time)
		}
		return
	}
	ss.Logs.LogRow(mode, time, row)
}

////////////////////////////////////////////////////////////////
// 		GUI

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "Attn"
	ss.GUI.MakeBody(ss, "attn", title, `attn: This simulation illustrates how object recognition (ventral, what) and spatial (dorsal, where) pathways interact to produce spatial attention effects, and accurately capture the effects of brain damage to the spatial pathway. See <a href="https://github.com/CompCogNeuro/sims/blob/main/ch6/attn/README.md">README.md on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 1100
	nv.SetNet(ss.Net)
	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1.2, 3.0)
	nv.SceneXYZ().Camera.LookAt(math32.Vector3{0, 0, 0}, math32.Vector3{0, 1, 0})
	ss.ViewUpdate.Config(nv, etime.Cycle, etime.Cycle)
	ss.GUI.ViewUpdate = &ss.ViewUpdate
	nv.Current()

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Init", Icon: icons.Update,
		Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Init()
			ss.GUI.UpdateWindow()
		},
	})

	ss.GUI.AddLooperCtrl(p, ss.Loops, []etime.Modes{etime.Test})

	tree.Add(p, func(w *core.Separator) {})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Reset Log",
		Icon:    icons.Reset,
		Tooltip: "Reset the accumulated trial log",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.Logs.ResetLog(etime.Test, etime.Trial)
			ss.GUI.UpdatePlot(etime.Test, etime.Trial)
		},
	})

	////////////////////////////////////////////////
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Defaults", Icon: icons.Update,
		Tooltip: "Restore initial default parameters.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Defaults()
			ss.Init()
			ss.GUI.SimForm.Update()
			ss.GUI.UpdateWindow()
		},
	})
	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "README",
		Icon:    icons.FileMarkdown,
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.TheApp.OpenURL("https://github.com/CompCogNeuro/sims/blob/main/ch6/attn/README.md")
		},
	})
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}
