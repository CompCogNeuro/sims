// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// rl explores the temporal differences (TD) reinforcement learning algorithm
// under some basic Pavlovian conditioning environments.
package main

//go:generate core generate -add-types

import (
	"embed"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
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
	"github.com/emer/etensor/tensor"
	"github.com/emer/etensor/tensor/stats/metric"
	"github.com/emer/etensor/tensor/stats/norm"
	"github.com/emer/etensor/tensor/table"
	"github.com/emer/leabra/v2/leabra"
)

//go:embed README.md
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
		{Sel: "Path", Desc: "no extra learning factors",
			Params: params.Params{
				"Path.Learn.Norm.On":     "false",
				"Path.Learn.Momentum.On": "false",
				"Path.Learn.WtBal.On":    "false",
			}},
		{Sel: "Layer", Desc: "faster average",
			Params: params.Params{
				"Layer.Act.Dt.AvgTau": "200",
			}},
		{Sel: "#Input", Desc: "input fixed act",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Fixed": "true", // critical for ensuring weights have same impact!
				"Layer.Inhib.ActAvg.Init":  "0.015",
			}},
		{Sel: ".TDToInteg", Desc: "rew to integ",
			Params: params.Params{
				"Path.Learn.Learn": "false",
				"Path.WtInit.Mean": "1",
				"Path.WtInit.Var":  "0",
				"Path.WtInit.Sym":  "false",
			}},
		{Sel: "#InputToPred", Desc: "input to rewpred",
			Params: params.Params{
				"Path.WtInit.Mean": "0",
				"Path.WtInit.Var":  "0",
				"Path.WtInit.Sym":  "false",
				"Path.Learn.Lrate": "0.5",
			}},
		{Sel: "#Rew", Desc: "allow negative",
			Params: params.Params{
				"Layer.Act.Clamp.Range.Min": "-1",
				"Layer.Act.Clamp.Range.Max": "1",
				"Layer.Inhib.ActAvg.Fixed":  "true", // critical for ensuring weights have same impact!
				"Layer.Inhib.ActAvg.Init":   "1",
			}},
	},
}

// Config has config parameters related to running the sim
type Config struct {
	// total number of runs to do when running Train
	NRuns int `default:"1" min:"1"`

	// total number of epochs per run
	NEpochs int `default:"30"`

	// total number of trials per epoch
	NTrials int `default:"20"`
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	// discount factor for future rewards
	Discount float32 `def:"0.9"`

	// learning rate
	Lrate float32 `def:"0.5"`

	// Config contains misc configuration parameters for running the sim
	Config Config `new-window:"+" display:"no-inline"`

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *leabra.Network `new-window:"+" display:"no-inline"`

	// network parameter management
	Params emer.NetParams `display:"add-fields"`

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
	ss.Net = leabra.NewNetwork("RL")
	ss.Params.Config(ParamSets, "", "", ss.Net)
	ss.Stats.Init()
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
	ss.Defaults()
}

func (ss *Sim) Defaults() {
	ss.Discount = 0.9
	ss.Lrate = 0.5
}

//////////////////////////////////////////////////////////////////////////////
// 		Configs

// ConfigAll configures all the elements using the standard functions
func (ss *Sim) ConfigAll() {
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var trn *CondEnv
	if len(ss.Envs) == 0 {
		trn = &CondEnv{}
		trn.Defaults()
		trn.RewVal = 1
		trn.NoRewVal = 0
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*CondEnv)
	}

	// note: names must be standard here!
	trn.Name = etime.Train.String()
	trn.Trial.Max = ss.Config.NTrials
	trn.Init(0)
	ss.Envs.Add(trn)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	rew, rp, ri, td := net.AddTDLayers("", 4)
	_ = rew
	_ = ri
	inp := net.AddLayer2D("Input", 3, 20, leabra.InputLayer)

	net.ConnectLayers(inp, rp, paths.NewFull(), leabra.TDPredPath)

	rp.PlaceRightOf(rew, 4)
	ri.PlaceRightOf(rp, 4)
	td.PlaceRightOf(ri, 4)
	inp.PlaceAbove(rew)

	net.Build()
	net.Defaults()

	td.AddAllSendToBut() // send dopamine to all layers..

	ss.ApplyParams()
	net.InitWeights()
}

func (ss *Sim) ApplyParams() {
	ss.Params.SetAll()

	ri := ss.Net.LayerByName("Integ")
	ri.TD.Discount = ss.Discount

	rp := ss.Net.LayerByName("Pred")
	fmi := errors.Log1(rp.RecvPathBySendName("Input")).(*leabra.Path)
	fmi.Learn.Lrate = ss.Lrate
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // in case user interactively changes tag
	ss.Loops.ResetCounters()
	ss.InitRandSeed(0)
	ss.ConfigEnv()
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

// ConfigLoops configures the control loops: Training
func (ss *Sim) ConfigLoops() {
	ls := looper.NewStacks()

	ls.AddStack(etime.Train).
		AddTime(etime.Run, ss.Config.NRuns).
		AddTime(etime.Epoch, ss.Config.NEpochs).
		AddTime(etime.Trial, ss.Config.NTrials).
		AddTime(etime.Cycle, 100)

	leabra.LooperStdPhases(ls, &ss.Context, ss.Net, 75, 99)                // plus phase timing
	leabra.LooperSimCycleAndLearn(ls, ss.Net, &ss.Context, &ss.ViewUpdate) // std algo code
	ls.Stacks[etime.Train].OnInit.Add("Init", func() { ss.Init() })

	for m, _ := range ls.Stacks {
		stack := ls.Stacks[m]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
	}

	ls.Loop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	/////////////////////////////////////////////
	// Logging

	ls.AddOnEndToAll("Log", func(mode, time enums.Enum) {
		ss.Log(mode.(etime.Modes), time.(etime.Times))
	})
	// leabra.LooperResetLogBelow(man, &ss.Logs)
	ls.Loop(etime.Train, etime.Run).OnEnd.Add("RunStats", func() {
		ss.Logs.RunStats("UniqPats")
	})

	////////////////////////////////////////////
	// GUI

	leabra.LooperUpdateNetView(ls, &ss.ViewUpdate, ss.Net, ss.NetViewCounters)
	leabra.LooperUpdatePlots(ls, &ss.GUI)
	ls.Stacks[etime.Train].OnInit.Add("GUI-Init", func() { ss.GUI.UpdateWindow() })

	ss.Loops = ls
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	ctx := &ss.Context
	net := ss.Net
	ev := ss.Envs.ByMode(ctx.Mode).(*CondEnv)
	ev.Step()
	lays := net.LayersByType(leabra.InputLayer, leabra.ClampDaLayer)
	net.InitExt()
	ss.Stats.SetString("TrialName", ev.String())
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
	ss.InitRandSeed(ss.Loops.Loop(etime.Train, etime.Run).Counter.Cur)
	ss.Envs.ByMode(etime.Train).Init(0)
	ctx.Reset()
	ctx.Mode = etime.Train
	// ss.Net.InitWeights()
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.PredFromInput()
}

/////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetFloat("UniqPats", 0.0)
	ss.Stats.SetString("TrialName", "")
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
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "TrialName", "Cycle", "UniqPats"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
}

// UniquePatStat analyzes the hidden activity patterns for the single-line test inputs
// to determine how many such lines have a distinct hidden pattern, as computed
// from the similarity matrix across patterns
func (ss *Sim) UniquePatStat(dt *table.Table) float64 {
	if dt.Rows == 0 {
		return 0
	}
	hc := errors.Log1(dt.ColumnByName("Hidden_Act")).(*tensor.Float32)
	norm.Binarize32(hc.Values, .5, 1, 0)
	ix := table.NewIndexView(dt)
	sm := ss.Stats.SimMat("UniqPats")
	sm.TableColumn(ix, "Hidden_Act", "TrialName", false, metric.SumSquares64)
	dm := sm.Mat
	nrow := dm.DimSize(0)
	uniq := 0
	for row := 0; row < nrow; row++ {
		tsr := dm.SubSpace([]int{row}).(*tensor.Float64)
		nzero := 0
		for _, vl := range tsr.Values {
			if vl == 0 {
				nzero++
			}
		}
		if nzero == 1 { // one zero in dist matrix means it was only identical to itself
			uniq++
		}
	}
	return float64(uniq)
}

func (ss *Sim) PredFromInput() {
	if ss.GUI.Grids == nil {
		return
	}
	wg := ss.Stats.F32Tensor("PredFromInput")
	vals := wg.Values
	inp := ss.Net.LayerByName("Input")
	isz := inp.Shape.Len()
	hid := ss.Net.LayerByName("Pred")
	ysz := hid.Shape.DimSize(0)
	xsz := hid.Shape.DimSize(1)
	for y := 0; y < ysz; y++ {
		for x := 0; x < xsz; x++ {
			ui := (y*xsz + x)
			ust := ui * isz
			vls := vals[ust : ust+isz]
			inp.SendPathValues(&vls, "Wt", hid, ui, "")
		}
	}
	gv := ss.GUI.Grid("Weights")
	gv.Update()
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Train, etime.Trial, "TDDaLayer", "TDPredLayer", "TDIntegLayer")
	if li, ok := ss.Logs.ItemByName("TD_Act"); ok {
		li.FixMin = false
	}
	ss.Logs.PlotItems("TD_Act")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Train, etime.Epoch)
	ss.Logs.NoPlot(etime.Train, etime.Run)
	ss.Logs.SetMeta(etime.Train, etime.Trial, "TD_Act:FixMin", "+")
	ss.Logs.SetMeta(etime.Train, etime.Trial, "TD_Act:FixMax", "+")
	ss.Logs.SetMeta(etime.Train, etime.Trial, "TD_Act:Max", "1")
	ss.Logs.SetMeta(etime.Train, etime.Trial, "TD_Act:Min", "-1")
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
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
		ss.TrialStats()
		ss.StatCounters()
	case time == etime.Epoch:
		ss.PredFromInput()
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

//////////////////////////////////////////////////////////////////////
// 		GUI

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "RL"
	ss.GUI.MakeBody(ss, "rl", title, `explores the temporal differences (TD) reinforcement learning algorithm under some basic Pavlovian conditioning environments. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch8/rl/README.md">README.md on GitHub</a>.</p>`, readme)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 300
	nv.Options.Raster.Max = 100
	nv.SetNet(ss.Net)
	nv.Options.PathWidth = 0.005
	ss.ViewUpdate.Config(nv, etime.GammaCycle, etime.GammaCycle)
	ss.GUI.ViewUpdate = &ss.ViewUpdate

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 2.0, 2.4)
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))

	nv.Current()

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.AddTableView(&ss.Logs, etime.Train, etime.Trial)

	wgv := ss.GUI.AddGridTab("Weights")
	wg := ss.Stats.F32Tensor("PredFromInput")
	wg.SetShape([]int{1, 1, 3, 20})
	wgv.SetTensor(wg)

	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Init Weights", Icon: icons.Update,
		Tooltip: "Initialize the learned learned weights.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Net.InitWeights()
			ss.PredFromInput()
			ss.GUI.UpdateWindow()
		},
	})

	ss.GUI.AddLooperCtrl(p, ss.Loops)

	tree.Add(p, func(w *core.Separator) {})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Reset Trial Log",
		Icon:    icons.Reset,
		Tooltip: "Reset the accumulated train trial log",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.Logs.ResetLog(etime.Train, etime.Trial)
			ss.GUI.UpdatePlot(etime.Train, etime.Trial)
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
			core.TheApp.OpenURL("https://github.com/CompCogNeuro/sims/blob/main/ch8/rl/README.md")
		},
	})
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}
