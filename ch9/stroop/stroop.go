// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// stroop illustrates how the PFC can produce top-down biasing for
// executive control, in the context of the widely studied Stroop task.
package main

//go:generate core generate -add-types

import (
	"embed"
	"fmt"
	"math"
	"reflect"
	"strings"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/math32/minmax"
	"cogentcore.org/core/styles"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/base/randx"
	"github.com/emer/emergent/v2/econfig"
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
	"github.com/emer/etensor/tensor/table"
	"github.com/emer/leabra/v2/leabra"
)

//go:embed stroop_train.tsv stroop_test.tsv stroop_soa.tsv
var content embed.FS

//go:embed *.png README.md
var readme embed.FS

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
		{Sel: "Path", Desc: "lower lrate, uniform init",
			Params: params.Params{
				"Path.Learn.Norm.On":     "false",
				"Path.Learn.Momentum.On": "false",
				"Path.Learn.WtBal.On":    "false",
				"Path.Learn.Lrate":       "0.04",
				"Path.WtInit.Mean":       "0.25",
				"Path.WtInit.Var":        "0",
			}},
		{Sel: "Layer", Desc: "high inhibition, layer act avg",
			Params: params.Params{
				"Layer.Act.XX1.Gain":       "40",
				"Layer.Learn.AvgL.Gain":    "1", // critical params here
				"Layer.Learn.AvgL.Init":    "0.2",
				"Layer.Learn.AvgL.Min":     "0.05",
				"Layer.Learn.AvgL.LrnMin":  "0.05",
				"Layer.Learn.AvgL.LrnMax":  "0.05",
				"Layer.Inhib.Layer.Gi":     "2.1",
				"Layer.Inhib.Layer.FB":     "0.5",
				"Layer.Inhib.ActAvg.Init":  "0.4",
				"Layer.Inhib.ActAvg.Fixed": "true",
			}},
		{Sel: "#Hidden", Desc: "higher inhibition",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi":     "3",
				"Layer.Inhib.ActAvg.Init":  "0.5",
				"Layer.Inhib.ActAvg.Fixed": "true",
			}},
		{Sel: "#Colors", Desc: "layer act avg",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Init": "0.5",
			}},
		{Sel: "#Words", Desc: "layer act avg",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Init": "0.5",
			}},
		{Sel: "#PFCToHidden", Desc: "PFC top-down projection",
			Params: params.Params{
				"Path.WtScale.Rel":        "0.3",
				"Path.Learn.Lrate":        "0.01", // even slower
				"Path.Learn.XCal.SetLLrn": "true",
				"Path.Learn.XCal.LLrn":    "0.1",
			}},
		{Sel: "#OutputToHidden", Desc: "Output top-down projection",
			Params: params.Params{
				"Path.WtScale.Rel":        "0.2",
				"Path.Learn.Lrate":        "0.04",
				"Path.Learn.XCal.SetLLrn": "true",
				"Path.Learn.XCal.LLrn":    "0.1",
			}},
		{Sel: "#HiddenToOutput", Desc: "to output",
			Params: params.Params{
				"Path.Learn.Lrate":        "0.08",
				"Path.Learn.XCal.SetLLrn": "true",
				"Path.Learn.XCal.LLrn":    "0.1",
			}},
	},
	"Training": {
		{Sel: "Layer", Desc: "faster time constant",
			Params: params.Params{
				"Layer.Act.Init.Decay": "1",
				"Layer.Act.Dt.VmTau":   "3.3",
			}},
	},
	"Testing": {
		{Sel: "Layer", Desc: "slower time constant",
			Params: params.Params{
				"Layer.Act.Init.Decay": "1",
				"Layer.Act.Dt.VmTau":   "30",
			}},
	},
	"SOATesting": {
		{Sel: "Layer", Desc: "no decay",
			Params: params.Params{
				"Layer.Act.Init.Decay": "0",
			}},
	},
}

// Config has config parameters related to running the sim
type Config struct {
	// total number of runs to do when running Train
	NRuns int `default:"1" min:"1"`

	// total number of epochs per run
	NEpochs int `default:"55"`

	// how often to run through all the test patterns, in terms of training epochs.
	// can use 0 or -1 for no testing.
	TestInterval int `default:"5"`
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {

	// strength of projection from PFC to Hidden -- reduce to simulate PFC damage
	FromPFC float32 `def:"0.3" step:"0.01"`

	// time constant for updating the network
	DtVmTau float32 `def:"30" step:"5"`

	// Config contains misc configuration parameters for running the sim
	Config Config `new-window:"+" display:"no-inline"`

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *leabra.Network `new-window:"+" display:"no-inline"`

	// network parameter management
	Params emer.NetParams `display:"add-fields"`

	// training patterns
	Train *table.Table `new-window:"+" display:"no-inline"`

	// testing patterns
	Test *table.Table `new-window:"+" display:"no-inline"`

	// SOA testing patterns
	SOA *table.Table `new-window:"+" display:"no-inline"`

	// contains looper control loops for running sim
	Loops *looper.Stacks `new-window:"+" display:"no-inline"`

	// contains computed statistic values
	Stats estats.Stats `new-window:"+"`

	// Contains all the logs and information about the logs.'
	Logs elog.Logs `new-window:"+"`

	// Environments
	Envs env.Envs `new-window:"+" display:"no-inline"`

	// leabra timing parameters and state
	Context leabra.Context `new-window:"+"`

	// netview update parameters
	ViewUpdate netview.ViewUpdate `display:"add-fields"`

	// manages all the gui elements
	GUI egui.GUI `display:"-"`

	// a list of random seeds to use for each run
	RandSeeds randx.Seeds `display:"-"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	econfig.Config(&ss.Config, "config.toml")
	ss.Defaults()
	ss.Net = leabra.NewNetwork("Stroop")
	ss.Params.Config(ParamSets, "", "", ss.Net)
	ss.Stats.Init()
	ss.Train = &table.Table{}
	ss.Test = &table.Table{}
	ss.SOA = &table.Table{}
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
}

func (ss *Sim) Defaults() {
	ss.FromPFC = 0.3
	ss.DtVmTau = 30
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

// OpenPatAsset opens pattern file from embedded assets
func (ss *Sim) OpenPatAsset(dt *table.Table, fnm, name, desc string) error {
	dt.SetMetaData("name", name)
	dt.SetMetaData("desc", desc)
	err := dt.OpenFS(content, fnm, table.Tab)
	if errors.Log(err) == nil {
		for i := 1; i < dt.NumColumns(); i++ {
			dt.Columns[i].SetMetaData("grid-fill", "0.9")
		}
	}
	return err
}

func (ss *Sim) OpenPatterns() {
	ss.OpenPatAsset(ss.Train, "stroop_train.tsv", "Train", "Stroop Training Patterns")
	ss.OpenPatAsset(ss.Test, "stroop_test.tsv", "Test", "Stroop Testing Patterns")
	ss.OpenPatAsset(ss.SOA, "stroop_soa.tsv", "SOA", "Stroop SOA Testing Patterns")
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var trn *env.FreqTable
	var tst, soa *env.FixedTable
	if len(ss.Envs) == 0 {
		trn = &env.FreqTable{}
		tst = &env.FixedTable{}
		soa = &env.FixedTable{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*env.FreqTable)
		tst = ss.Envs.ByMode(etime.Test).(*env.FixedTable)
		soa = ss.Envs.ByMode(etime.Validate).(*env.FixedTable)
	}

	// note: names must be standard here!
	trn.Name = etime.Train.String()
	trn.Table = table.NewIndexView(ss.Train)
	trn.NSamples = 1

	tst.Name = etime.Test.String()
	tst.Config(table.NewIndexView(ss.Test))
	tst.Sequential = true

	soa.Name = etime.Validate.String()
	soa.Config(table.NewIndexView(ss.SOA))
	soa.Sequential = true

	trn.Init(0)
	tst.Init(0)
	soa.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn, tst, soa)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	clr := net.AddLayer2D("Colors", 1, 2, leabra.InputLayer)
	wrd := net.AddLayer2D("Words", 1, 2, leabra.InputLayer)
	hid := net.AddLayer4D("Hidden", 1, 2, 1, 2, leabra.SuperLayer)
	pfc := net.AddLayer2D("PFC", 1, 2, leabra.InputLayer)
	out := net.AddLayer2D("Output", 1, 2, leabra.TargetLayer)

	full := paths.NewFull()
	clr2hid := paths.NewOneToOne()
	wrd2hid := paths.NewOneToOne()
	wrd2hid.RecvStart = 2

	pfc2hid := paths.NewRect()
	pfc2hid.Scale.Set(0.5, 0.5)
	pfc2hid.Size.Set(1, 1)

	net.ConnectLayers(clr, hid, clr2hid, leabra.ForwardPath)
	net.ConnectLayers(wrd, hid, wrd2hid, leabra.ForwardPath)
	net.ConnectLayers(pfc, hid, pfc2hid, leabra.BackPath)
	net.BidirConnectLayers(hid, out, full)

	wrd.PlaceRightOf(clr, 1)
	out.PlaceRightOf(wrd, 1)
	hid.PlaceAbove(clr)
	pfc.PlaceRightOf(hid, 1)

	net.Build()
	net.Defaults()
	ss.ApplyParams()
	net.InitWeights()
}

func (ss *Sim) ApplyParams() {
	spo, _ := ss.Params.Params["Testing"].SelByName("Layer")
	spo.Params.SetByName("Layer.Act.Dt.VmTau", fmt.Sprintf("%g", ss.DtVmTau))
	ss.Params.SetAll()
	ss.Params.SetAllSheet("Training")

	ss.SetPFCParams()

	if ss.Loops != nil {
		trn := ss.Loops.Stacks[etime.Train]
		trn.Loops[etime.Run].Counter.Max = ss.Config.NRuns
		trn.Loops[etime.Epoch].Counter.Max = ss.Config.NEpochs
	}
}

func (ss *Sim) SetPFCParams() {
	hid := ss.Net.LayerByName("Hidden")
	fmpfc := errors.Log1(hid.RecvPathBySendName("PFC")).(*leabra.Path)
	fmpfc.WtScale.Rel = ss.FromPFC
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // in case user interactively changes tag
	ss.Loops.ResetCounters()
	ss.InitRandSeed(0)
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
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

func (ss *Sim) TestInit() {
	ss.ApplyParams()
	ss.Params.SetAllSheet("Testing")
	ss.SetPFCParams()
	ev := ss.Envs.ByMode(etime.Validate)
	ev.Init(0)
}

func (ss *Sim) SOATestInit() {
	ss.ApplyParams()
	ss.Params.SetAllSheet("Testing")
	ss.Params.SetAllSheet("SOATesting")
	ss.SetPFCParams()
	ev := ss.Envs.ByMode(etime.Validate)
	ev.Init(0)
}

// CycleThresholdStop
func (ss *Sim) CycleThresholdStop() {
	if ss.Context.Mode == etime.Train {
		return
	}
	mode := ss.Context.Mode
	cycl := ss.Loops.Stacks[mode].Loops[etime.Cycle]
	cyc := cycl.Counter.Cur
	out := ss.Net.LayerByName("Output")
	outact := out.Pools[0].Inhib.Act.Max
	if mode == etime.Validate {
		tbl := ss.SOA
		trl := ss.Loops.Stacks[mode].Loops[etime.Trial].Counter.Cur
		// soa := int(tbl.Float("SOA", trl))
		mxc := int(tbl.Float("MaxCycles", trl))
		islate := strings.Contains(ss.Stats.String("TrialName"), "latestim")
		if islate {
			if outact > 0.51 {
				ss.Stats.SetFloat("RT", float64(cyc))
				cycl.SkipToMax()
			}
		} else {
			if cyc > mxc {
				cycl.SkipToMax()
			}
		}
		// fmt.Println(trl, tnm, soa, mxc)
	} else {
		if outact > 0.51 || cyc == cycl.Counter.Max-1 {
			ss.Stats.SetFloat("RT", float64(cyc))
			cycl.SkipToMax()
		}
	}
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	ls := looper.NewStacks()

	trls := 16

	ls.AddStack(etime.Train).
		AddTime(etime.Run, ss.Config.NRuns).
		AddTime(etime.Epoch, ss.Config.NEpochs).
		AddTime(etime.Trial, trls).
		AddTime(etime.Cycle, 100)

	ls.AddStack(etime.Test).
		AddTime(etime.Epoch, 1).
		AddTime(etime.Trial, ss.Test.Rows).
		AddTime(etime.Cycle, 250)

	ls.AddStack(etime.Validate).
		AddTime(etime.Epoch, 1).
		AddTime(etime.Trial, ss.SOA.Rows).
		AddTime(etime.Cycle, 250)

	leabra.LooperStdPhases(ls, &ss.Context, ss.Net, 75, 99)                // plus phase timing
	leabra.LooperSimCycleAndLearn(ls, ss.Net, &ss.Context, &ss.ViewUpdate) // std algo code

	ls.Stacks[etime.Train].OnInit.Add("Init", func() { ss.Init() })
	ls.Stacks[etime.Test].OnInit.Add("Init", func() { ss.TestInit() })
	ls.Stacks[etime.Validate].OnInit.Add("Init", func() { ss.SOATestInit() })

	for m, _ := range ls.Stacks {
		stack := ls.Stacks[m]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
		if m == etime.Test || m == etime.Validate {
			stack.Loops[etime.Cycle].OnEnd.Add("CycleThresholdStop", func() {
				ss.CycleThresholdStop()
			})
		}
	}

	ls.Loop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	/////////////////////////////////////////////
	// Logging

	ls.Loop(etime.Test, etime.Epoch).OnEnd.Add("LogTestErrors", func() {
		leabra.LogTestErrors(&ss.Logs)
	})
	ls.AddOnEndToAll("Log", func(mode, time enums.Enum) {
		ss.Log(mode.(etime.Modes), time.(etime.Times))
	})
	leabra.LooperResetLogBelow(ls, &ss.Logs)
	ls.Loop(etime.Train, etime.Run).OnEnd.Add("RunStats", func() {
		ss.Logs.RunStats("PctCor", "FirstZero", "LastZero")
	})

	////////////////////////////////////////////
	// GUI

	leabra.LooperUpdateNetView(ls, &ss.ViewUpdate, ss.Net, ss.NetViewCounters)
	leabra.LooperUpdatePlots(ls, &ss.GUI)
	ls.Stacks[etime.Train].OnInit.Add("GUI-Init", func() { ss.GUI.UpdateWindow() })
	ls.Stacks[etime.Test].OnInit.Add("GUI-Init", func() { ss.GUI.UpdateWindow() })

	// fmt.Println(ls.DocString())
	ss.Loops = ls
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	ctx := &ss.Context
	net := ss.Net
	evi := ss.Envs.ByMode(ctx.Mode)
	evi.Step()
	ss.Stats.SetFloat("RT", math.NaN())
	out := ss.Net.LayerByName("Output")

	lays := net.LayersByType(leabra.InputLayer, leabra.TargetLayer)
	net.InitExt()
	if ctx.Mode == etime.Train {
		out.Type = leabra.TargetLayer
		// ss.Stats.SetString("TrialName", evi.(*env.FreqTable).String())
	} else {
		out.Type = leabra.CompareLayer
		trlnm := evi.(*env.FixedTable).TrialName.Cur
		ss.Stats.SetString("TrialName", trlnm)
		if ctx.Mode == etime.Validate {
			if !strings.Contains(trlnm, "latestim") || strings.Contains(trlnm, "Both") {
				ss.Net.InitActs()
			}
		}
	}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm)
		pats := evi.State(ly.Name)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ctx := &ss.Context
	ss.InitRandSeed(ss.Loops.Loop(etime.Train, etime.Run).Counter.Cur)
	ss.Envs.ByMode(etime.Train).Init(0)
	ss.Envs.ByMode(etime.Test).Init(0)
	ctx.Reset()
	ctx.Mode = etime.Train
	ss.Net.InitWeights()
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.Envs.ByMode(etime.Test).Init(0)
	ss.Loops.ResetAndRun(etime.Test)
	ss.Loops.Mode = etime.Train // Important to reset Mode back to Train because this is called from within the Train Run.
}

/////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetFloat("SSE", 0.0)
	ss.Stats.SetFloat("RT", math.NaN())
	ss.Stats.SetFloat("SOA", 0)
	ss.Stats.SetString("TrialName", "")
	ss.Stats.SetString("GroupName", "")
	ss.Logs.InitErrStats() // inits TrlErr, FirstZero, LastZero, NZero
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them for ViewUpdate.Text
func (ss *Sim) StatCounters() {
	ctx := &ss.Context
	mode := ctx.Mode
	ss.Loops.Stacks[mode].CountersToStats(&ss.Stats)
	// always use training epoch..
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	trl := ss.Stats.Int("Trial")
	if ctx.Mode == etime.Test {
		trl = trl % 3
	} else if ctx.Mode == etime.Validate {
		trl := ss.Loops.Stacks[mode].Loops[etime.Trial].Counter.Cur
		soa := ss.SOA.Float("SOA", trl)
		ss.Stats.SetFloat("SOA", soa)
		gp := trl / 22
		conds := []string{"Color_Conf", "Color_Cong", "Word_Conf", "Word_Cong"}
		ss.Stats.SetString("GroupName", conds[gp])
		trl = trl % 22
	}
	ss.Stats.SetInt("Trial", trl)
	ss.Stats.SetInt("Cycle", int(ctx.Cycle))
}

func (ss *Sim) NetViewCounters(tm etime.Times) {
	if ss.ViewUpdate.View == nil {
		return
	}
	if tm == etime.Trial {
		ss.TrialStats() // get trial stats for current di
	}
	ss.StatCounters()
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "TrialName", "Cycle", "SSE", "TrlErr"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	out := ss.Net.LayerByName("Output")
	ctx := &ss.Context
	if ctx.Mode != etime.Train {
		for i := range out.Neurons {
			out.Neurons[i].ActM = out.Neurons[i].Act // use final state for err measure
		}
	}

	sse, avgsse := out.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
	ss.Stats.SetFloat("SSE", sse)
	ss.Stats.SetFloat("AvgSSE", avgsse)
	if sse > 0 {
		ss.Stats.SetFloat("TrlErr", 1)
	} else {
		ss.Stats.SetFloat("TrlErr", 0)
	}
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName", "GroupName")

	ss.Logs.AddStatAggItem("SSE", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("AvgSSE", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddErrStatAggItems("TrlErr", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddStatAggItem("RT", etime.Epoch, etime.Trial)

	ss.Logs.AddItem(&elog.Item{
		Name:   "SOA",
		Type:   reflect.Float64,
		FixMax: true,
		Range:  minmax.F32{Max: 220},
		Write: elog.WriteMap{
			etime.Scope(etime.Validate, etime.Trial): func(ctx *elog.Context) {
				ctx.SetFloat64(ss.Stats.Float("SOA"))
			}}})

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.PlotItems("RT")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Train, etime.Run)
	ss.Logs.NoPlot(etime.Test, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Epoch)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	ss.Logs.SetMetaScope(etime.Scope(etime.Validate, etime.Trial), "Plot", "true")
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")

	ss.Logs.SetMeta(etime.Test, etime.Trial, "Points", "true")

	ss.Logs.SetMeta(etime.Test, etime.Trial, "Trial:FixMin", "true")
	ss.Logs.SetMeta(etime.Test, etime.Trial, "Trial:FixMax", "true")
	ss.Logs.SetMeta(etime.Test, etime.Trial, "Trial:Min", "-0.3")
	ss.Logs.SetMeta(etime.Test, etime.Trial, "Trial:Max", "2.3")

	ss.Logs.SetMeta(etime.Test, etime.Trial, "RT:FixMin", "true")
	ss.Logs.SetMeta(etime.Test, etime.Trial, "RT:FixMax", "true")
	ss.Logs.SetMeta(etime.Test, etime.Trial, "RT:Min", "0")
	ss.Logs.SetMeta(etime.Test, etime.Trial, "RT:Max", "250")
	ss.Logs.SetMeta(etime.Test, etime.Trial, "TrialName:On", "+")

	ss.Logs.SetMeta(etime.Train, etime.Epoch, "PctErr:On", "+")

	ss.Logs.SetMeta(etime.Validate, etime.Trial, "XAxis", "SOA")
	ss.Logs.SetMeta(etime.Validate, etime.Trial, "Legend", "GroupName")
	ss.Logs.SetMeta(etime.Validate, etime.Trial, "Points", "true")

	ss.Logs.SetMeta(etime.Validate, etime.Trial, "SOA:Min", "-22")
	ss.Logs.SetMeta(etime.Validate, etime.Trial, "SOA:Max", "22")
	ss.Logs.SetMeta(etime.Validate, etime.Trial, "SOA:FixMin", "true")
	ss.Logs.SetMeta(etime.Validate, etime.Trial, "SOA:FixMax", "true")
	ss.Logs.SetMeta(etime.Validate, etime.Trial, "RT:FixMin", "true")
	ss.Logs.SetMeta(etime.Validate, etime.Trial, "RT:FixMax", "true")
	ss.Logs.SetMeta(etime.Validate, etime.Trial, "RT:Min", "0")
	ss.Logs.SetMeta(etime.Validate, etime.Trial, "RT:Max", "250")
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

	switch {
	case time == etime.Cycle:
		return
	case time == etime.Trial:
		if mode == etime.Validate {
			if !strings.Contains(ss.Stats.String("TrialName"), "latestim") {
				return
			}
		}
		ss.TrialStats()
		ss.StatCounters()
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

//////////////////////////////////////////////////////////////////////
// 		GUI

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.SceneXYZ().Camera.Pose.Pos.Set(0.1, 1.8, 3.5)
	nv.SceneXYZ().Camera.LookAt(math32.Vector3{0.1, 0.15, 0}, math32.Vector3{0, 1, 0})

	labs := []string{"     g      r", "   G       R", "  gr      rd", "     g      r         G      R", "  cn     wr"}
	nv.ConfigLabels(labs)

	lays := []string{"Colors", "Words", "Output", "Hidden", "PFC"}

	for li, lnm := range lays {
		ly := nv.LayerByName(lnm)
		lbl := nv.LabelByName(labs[li])
		lbl.Pose = ly.Pose
		lbl.Pose.Pos.Y += .2
		lbl.Pose.Pos.Z += .02
		lbl.Pose.Scale.SetMul(math32.Vector3{0.4, 0.06, 0.5})
		lbl.Styles.Text.WhiteSpace = styles.WhiteSpacePre
	}
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "Stroop"
	ss.GUI.MakeBody(ss, "stroop", title, `illustrates how the PFC can produce top-down biasing for executive control, in the context of the widely studied Stroop task. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch9/stroop/README.md">README.md on GitHub</a>.</p>`, readme)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 300
	nv.Options.Raster.Max = 100
	nv.SetNet(ss.Net)
	nv.Options.PathWidth = 0.005
	ss.ViewUpdate.Config(nv, etime.GammaCycle, etime.Cycle)
	ss.GUI.ViewUpdate = &ss.ViewUpdate
	nv.Current()

	ss.ConfigNetView(nv)

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.Loops)

	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Reset RunLog",
		Icon:    icons.Reset,
		Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.Logs.ResetLog(etime.Train, etime.Run)
			ss.GUI.UpdatePlot(etime.Train, etime.Run)
		},
	})
	////////////////////////////////////////////////
	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "New Seed",
		Icon:    icons.Add,
		Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.RandSeeds.NewSeeds()
		},
	})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "README",
		Icon:    icons.FileMarkdown,
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.TheApp.OpenURL("https://github.com/CompCogNeuro/sims/blob/main/ch9/stroop/README.md")
		},
	})
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}
