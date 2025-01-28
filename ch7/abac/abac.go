// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// abac explores the classic paired associates learning task in a
// cortical-like network, which exhibits catastrophic levels of
// interference.
package main

//go:generate core generate -add-types

import (
	"embed"
	"fmt"
	"reflect"

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
	"github.com/emer/etensor/plot/plotcore"
	"github.com/emer/etensor/tensor/stats/clust"
	"github.com/emer/etensor/tensor/stats/metric"
	"github.com/emer/etensor/tensor/stats/split"
	"github.com/emer/etensor/tensor/stats/stats"
	"github.com/emer/etensor/tensor/table"
	"github.com/emer/leabra/v2/leabra"
	"gonum.org/v1/gonum/mat"
)

//go:embed ab_pats.tsv ac_pats.tsv
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
		{Sel: "Path", Desc: "fixed LLrn",
			Params: params.Params{
				"Path.Learn.Norm.On":      "false",
				"Path.Learn.Momentum.On":  "false",
				"Path.Learn.WtBal.On":     "true",
				"Path.WtInit.Var":         "0.25",
				"Path.Learn.XCal.SetLLrn": "true",
				"Path.Learn.XCal.LLrn":    "0.0003",
				"Path.Learn.Lrate":        "0.04",
			}},
		{Sel: "Layer", Desc: "Default learning, inhib params",
			Params: params.Params{
				"Layer.Learn.AvgL.Gain": "2.5",
				"Layer.Inhib.Layer.Gi":  "1.6", // output layer does better with this!
			}},
		{Sel: ".BackPath", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
			Params: params.Params{
				"Path.WtScale.Rel": "0.3", // note: this was missing in C++ version! makes this work much better
			}},
	},
}

// Config has config parameters related to running the sim
type Config struct {
	// total number of runs to do when running Train
	NRuns int `default:"10" min:"1"`

	// total number of epochs per run
	NEpochs int `default:"100"`

	// stop run after this number of perfect, zero-error epochs.
	NZero int `default:"1"`

	// how often to run through all the test patterns, in terms of training epochs.
	// can use 0 or -1 for no testing.
	TestInterval int `default:"1"`
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {

	// HiddenInhibGi is the hidden layer inhibition; increase to make sparser.
	HiddenInhibGi float32 `def:"1.8"`

	// WtInitVar is the random initial weight variance; increase to make more random.
	WtInitVar float32 `def:"0.25"`

	// FmContext is the relative WtScale.Rel from Context layer.
	FmContext float32 `def:"1"`

	// XCalLLrn is the amount of Hebbian BCM learning based on AvgL long-term average
	// activity. Increase to increase amount of hebbian.
	XCalLLrn float32 `min:"0" step:"0.0001" def:"0.0003"`

	// Lrate is the learning rate
	Lrate float32 `def:"0.04"`

	// Config contains misc configuration parameters for running the sim
	Config Config `new-window:"+" display:"no-inline"`

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *leabra.Network `new-window:"+" display:"no-inline"`

	// network parameter management
	Params emer.NetParams `display:"add-fields"`

	// AB training patterns
	ABPatterns *table.Table `new-window:"+" display:"no-inline"`

	// AC training patterns
	ACPatterns *table.Table `new-window:"+" display:"no-inline"`

	// ABAC testing patterns
	ABACPatterns *table.Table `new-window:"+" display:"no-inline"`

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
	ss.Defaults()
	econfig.Config(&ss.Config, "config.toml")
	ss.Net = leabra.NewNetwork("ABAC")
	ss.Params.Config(ParamSets, "", "", ss.Net)
	ss.Stats.Init()
	ss.Stats.SetInt("Expt", 0)
	ss.ABPatterns = &table.Table{}
	ss.ACPatterns = &table.Table{}
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
}

func (ss *Sim) Defaults() {
	ss.HiddenInhibGi = 1.8
	ss.WtInitVar = 0.25
	ss.FmContext = 1
	ss.XCalLLrn = 0.0003
	ss.Lrate = 0.04
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
	col := table.InsertColumn[string](dt, "Group", 1)
	for i := range col.DimSize(0) {
		col.SetString1D(i, name)
	}
	return err
}

func (ss *Sim) OpenPatterns() {
	ss.OpenPatAsset(ss.ABPatterns, "ab_pats.tsv", "AB", "AB Training Patterns")
	ss.OpenPatAsset(ss.ACPatterns, "ac_pats.tsv", "AC", "AC Training Patterns")

	ss.ABACPatterns = ss.ABPatterns.Clone()
	ss.ABACPatterns.SetMetaData("name", "ABAC")
	ss.ABACPatterns.AppendRows(ss.ACPatterns)
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var trn, tst *env.FixedTable
	if len(ss.Envs) == 0 {
		trn = &env.FixedTable{}
		tst = &env.FixedTable{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*env.FixedTable)
		tst = ss.Envs.ByMode(etime.Test).(*env.FixedTable)
	}

	// note: names must be standard here!
	trn.Name = etime.Train.String()
	trn.Config(table.NewIndexView(ss.ABPatterns))

	tst.Name = etime.Test.String()
	tst.Config(table.NewIndexView(ss.ABACPatterns))
	tst.Sequential = true

	trn.Init(0)
	tst.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	inp := net.AddLayer2D("Input", 5, 5, leabra.InputLayer)
	ctx := net.AddLayer2D("Context", 5, 5, leabra.InputLayer)
	hid := net.AddLayer2D("Hidden", 10, 15, leabra.SuperLayer)
	out := net.AddLayer2D("Output", 5, 5, leabra.TargetLayer)

	full := paths.NewFull()

	net.ConnectLayers(inp, hid, full, leabra.ForwardPath)
	net.ConnectLayers(ctx, hid, full, leabra.ForwardPath)
	net.BidirConnectLayers(hid, out, full)

	ctx.PlaceRightOf(inp, 2)
	hid.PlaceAbove(inp)

	net.Build()
	net.Defaults()
	ss.ApplyParams()
	net.InitWeights()
}

func (ss *Sim) ApplyParams() {
	if ss.Loops != nil {
		trn := ss.Loops.Stacks[etime.Train]
		trn.Loops[etime.Run].Counter.Max = ss.Config.NRuns
		trn.Loops[etime.Epoch].Counter.Max = ss.Config.NEpochs
	}

	spo := errors.Log1(errors.Log1(ss.Params.Params.SheetByName("Base")).SelByName("Path"))
	spo.Params.SetByName("Path.WtInit.Var", fmt.Sprintf("%g", ss.WtInitVar))
	spo.Params.SetByName("Path.Learn.XCal.LLrn", fmt.Sprintf("%g", ss.XCalLLrn))
	spo.Params.SetByName("Path.Learn.Lrate", fmt.Sprintf("%g", ss.Lrate))

	ss.Params.SetAll()

	hid := ss.Net.LayerByName("Hidden")
	hid.Inhib.Layer.Gi = ss.HiddenInhibGi

	fmc := errors.Log1(hid.RecvPathBySendName("Context")).(*leabra.Path)
	fmc.WtScale.Rel = ss.FmContext
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

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	ls := looper.NewStacks()

	trls := ss.ABPatterns.Rows

	ls.AddStack(etime.Train).
		AddTime(etime.Run, ss.Config.NRuns).
		AddTime(etime.Epoch, ss.Config.NEpochs).
		AddTime(etime.Trial, trls).
		AddTime(etime.Cycle, 100)

	ls.AddStack(etime.Test).
		AddTime(etime.Epoch, 1).
		AddTime(etime.Trial, ss.ABACPatterns.Rows).
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

	ls.Loop(etime.Train, etime.Run).OnEnd.Add("RunDone", func() {
		if ss.Stats.Int("Run") >= ss.Config.NRuns-1 {
			ss.RunStats()
			expt := ss.Stats.Int("Expt")
			ss.Stats.SetInt("Expt", expt+1)
		}
	})

	// Train stop early condition
	ls.Loop(etime.Train, etime.Epoch).IsDone.AddBool("NZeroStop", func() bool {
		// This is calculated in TrialStats
		stopNz := ss.Config.NZero
		if stopNz <= 0 {
			stopNz = 2
		}
		curNZero := ss.Stats.Int("NZero")
		stop := curNZero >= stopNz
		if !stop {
			return false
		}
		trn := ss.Envs.ByMode(etime.Train).(*env.FixedTable)
		if trn.Table.Table.MetaData["name"] == "AC" {
			return stop
		}
		epc := ss.Stats.Int("Epoch")
		if stop || epc >= 50 {
			ss.Stats.SetInt("FirstPerfect", epc)
			trn.Config(table.NewIndexView(ss.ACPatterns))
		}
		return false
	})

	// Add Testing
	trainEpoch := ls.Loop(etime.Train, etime.Epoch)
	trainEpoch.OnStart.Add("TestAtInterval", func() {
		if (ss.Config.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.Config.TestInterval == 0) {
			// Note the +1 so that it doesn't occur at the 0th timestep.
			ss.TestAll()
		}
	})

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

	ss.Loops = ls
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	ctx := &ss.Context
	net := ss.Net
	ev := ss.Envs.ByMode(ctx.Mode).(*env.FixedTable)
	ev.Step()

	lays := net.LayersByType(leabra.InputLayer, leabra.TargetLayer)
	net.InitExt()
	ss.Stats.SetString("TrialName", ev.TrialName.Cur)
	ss.Stats.SetString("GroupName", ev.GroupName.Cur)
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
	trn := ss.Envs.ByMode(etime.Train).(*env.FixedTable)
	trn.Config(table.NewIndexView(ss.ABPatterns))
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

////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetFloat("SSE", 0.0)
	ss.Stats.SetFloat("ABErr", 0.0)
	ss.Stats.SetFloat("ACErr", 0.0)
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
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "GroupName", "TrialName", "Cycle", "SSE", "TrlErr"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	params := fmt.Sprintf("hid_gi: %g, wt_var: %g, fm_ctxt: %g, lrate: %g", ss.HiddenInhibGi, ss.WtInitVar, ss.FmContext, ss.Lrate)
	ss.Stats.SetString("RunName", params)

	out := ss.Net.LayerByName("Output")

	sse, avgsse := out.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
	ss.Stats.SetFloat("SSE", sse)
	ss.Stats.SetFloat("AvgSSE", avgsse)
	if sse > 0 {
		ss.Stats.SetFloat("TrlErr", 1)
	} else {
		ss.Stats.SetFloat("TrlErr", 0)
	}
}

func (ss *Sim) TestStats() {
	trl := ss.Logs.Table(etime.Test, etime.Trial)
	if trl.Rows == 0 {
		return
	}
	trix := table.NewIndexView(trl)
	spl := split.GroupBy(trix, "GroupName")
	split.AggColumn(spl, "Err", stats.Mean)
	tsts := spl.AggsToTable(table.ColumnNameOnly)
	ss.Logs.MiscTables["TestEpoch"] = tsts
	ss.Stats.SetFloat("ABErr", tsts.Columns[1].Float1D(0))
	ss.Stats.SetFloat("ACErr", tsts.Columns[1].Float1D(1))
}

func (ss *Sim) RunStats() {
	dt := ss.Logs.Table(etime.Train, etime.Run)
	runix := table.NewIndexView(dt)
	spl := split.GroupBy(runix, "Expt")
	split.DescColumn(spl, "ABErr")
	st := spl.AggsToTableCopy(table.AddAggName)
	ss.Logs.MiscTables["RunStats"] = st
	plt := ss.GUI.Plots[etime.ScopeKey("RunStats")]

	st.SetMetaData("XAxis", "RunName")

	st.SetMetaData("Points", "true")

	st.SetMetaData("ABErr:Mean:On", "+")
	st.SetMetaData("ABErr:Mean:FixMin", "true")
	st.SetMetaData("ABErr:Mean:FixMax", "true")
	st.SetMetaData("ABErr:Mean:Min", "0")
	st.SetMetaData("ABErr:Mean:Max", "1")
	st.SetMetaData("ABErr:Min:On", "+")
	st.SetMetaData("ABErr:Count:On", "-")

	plt.SetTable(st)
	plt.GoUpdatePlot()
}

// RepsAnalysis analyzes the representations as captured in the Test Trial Log
func (ss *Sim) RepsAnalysis() {
	trl := ss.Logs.Table(etime.Test, etime.Trial)

	ss.Stats.SVD.Kind = mat.SVDFull // critical
	rels := table.NewIndexView(trl)
	rels.SortColumnName("TrialName", table.Ascending)
	ss.Stats.SimMat("Hidden").TableColumn(rels, "Hidden_ActM", "TrialName", true, metric.Correlation64)
	errors.Log(ss.Stats.SVD.TableColumn(rels, "Hidden_ActM", metric.Covariance64))
	svt := ss.Logs.MiscTable("HiddenPCA")
	ss.Stats.SVD.ProjectColumnToTable(svt, rels, "Hidden_ActM", "TrialName", []int{0, 1})
	estats.ConfigPCAPlot(ss.GUI.PlotByName("HiddenPCA"), svt, "Hidden Rel PCA")
	estats.ClusterPlot(ss.GUI.PlotByName("HiddenClust"), rels, "Hidden_ActM", "TrialName", clust.ContrastDist)
}

//////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.AllTimes, "Expt")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "GroupName", "TrialName")

	ss.Logs.AddStatAggItem("SSE", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("AvgSSE", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddErrStatAggItems("TrlErr", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddItem(&elog.Item{
		Name: "ABErr",
		Type: reflect.Float64,
		Plot: false,
		Write: elog.WriteMap{
			etime.Scope(etime.AllModes, etime.Epoch): func(ctx *elog.Context) {
				ctx.SetFloat64(ctx.Stats.Float("ABErr"))
			}, etime.Scope(etime.AllModes, etime.Run): func(ctx *elog.Context) {
				ctx.SetFloat64(ctx.Stats.Float("ABErr"))
			}}})

	ss.Logs.AddItem(&elog.Item{
		Name: "ACErr",
		Type: reflect.Float64,
		Plot: false,
		Write: elog.WriteMap{
			etime.Scope(etime.AllModes, etime.Epoch): func(ctx *elog.Context) {
				ctx.SetFloat64(ctx.Stats.Float("ACErr"))
			}, etime.Scope(etime.AllModes, etime.Run): func(ctx *elog.Context) {
				ctx.SetFloat64(ctx.Stats.Float("ACErr"))
			}}})

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddLayerTensorItems(ss.Net, "ActM", etime.Test, etime.Trial, "InputLayer", "SuperLayer", "TargetLayer")
	ss.Logs.AddLayerTensorItems(ss.Net, "Targ", etime.Test, etime.Trial, "TargetLayer")

	ss.Logs.PlotItems("ABErr", "ACErr")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Trial)
	ss.Logs.NoPlot(etime.Test, etime.Run)
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
	case time == etime.Epoch && mode == etime.Test:
		ss.TestStats()
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc

	if mode == etime.Test {
		ss.GUI.UpdateTableView(etime.Test, etime.Trial)
	}
}

//////////////////////////////////////////////////////////////////////
// 		GUI

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "ABAC"
	ss.GUI.MakeBody(ss, "abac", title, `abac explores the classic paired associates learning task in a cortical-like network, which exhibits catastrophic levels of interference. See <a href="https://github.com/CompCogNeuro/sims/blob/main/ch7/abac/README.md">README.md on GitHub</a>.</p>`, readme)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
	nv.Options.Raster.Max = 100
	nv.Options.MaxRecs = 300
	nv.SetNet(ss.Net)
	nv.Options.PathWidth = 0.003
	ss.ViewUpdate.Config(nv, etime.GammaCycle, etime.GammaCycle)
	ss.GUI.ViewUpdate = &ss.ViewUpdate
	nv.Current()

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1.15, 2.25)
	nv.SceneXYZ().Camera.LookAt(math32.Vector3{0, -0.15, 0}, math32.Vector3{0, 1, 0})

	ss.GUI.AddPlots(title, &ss.Logs)

	stnm := "RunStats"
	dt := ss.Logs.MiscTable(stnm)
	bcp, _ := ss.GUI.Tabs.NewTab(stnm + " Plot")
	plt := plotcore.NewSubPlot(bcp)
	ss.GUI.Plots[etime.ScopeKey(stnm)] = plt
	plt.Options.Title = "Run Stats"
	plt.Options.XAxis = "RunName"
	plt.SetTable(dt)

	ss.GUI.AddTableView(&ss.Logs, etime.Test, etime.Trial)

	ss.GUI.AddMiscPlotTab("HiddenPCA")
	ss.GUI.AddMiscPlotTab("HiddenClust")

	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.Loops)

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Reps Analysis",
		Icon:    icons.Reset,
		Tooltip: "analyzes the current testing Hidden activations, producing PCA, Cluster and Similarity Matrix (SimMat) plots",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.RepsAnalysis()
		},
	})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Build Net",
		Icon:    icons.Reset,
		Tooltip: "Build the network with updated hidden layer size",
		Active:  egui.ActiveAlways,
		Func: func() {
			net := ss.Net
			net.Build()
			net.Defaults()
			ss.ApplyParams()
			ss.ViewUpdate.View.SetNet(net)
			net.InitWeights()
		},
	})

	////////////////////////////////////////////////
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
			core.TheApp.OpenURL("https://github.com/CompCogNeuro/sims/blob/main/ch7/abac/README.md")
		},
	})
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}
