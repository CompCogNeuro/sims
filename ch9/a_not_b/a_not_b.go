// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// a_not_b explores how the development of PFC active maintenance abilities can help
// to make behavior more flexible, in the sense that it can rapidly shift with changes
// in the environment. The development of flexibility has been extensively explored
// in the context of Piaget's famous A-not-B task, where a toy is first hidden several
// times in one hiding location (A), and then hidden in a new location (B). Depending
// on various task parameters, young kids reliably reach back at A instead of updating to B.
package main

//go:generate core generate -add-types

import (
	"embed"
	"math"
	"strings"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
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

//go:embed a_not_b_delay3.tsv a_not_b_delay5.tsv a_not_b_delay1.tsv
var content embed.FS

//go:embed README.md
var readme embed.FS

// Delays is delay case to use
type Delays int32 //enums:enum

const (
	Delay3 Delays = iota
	Delay5
	Delay1
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
		{Sel: "Path", Desc: "lower lrate, uniform init, fixed LLrn, no MLrn",
			Params: params.Params{
				"Path.Learn.Norm.On":      "false",
				"Path.Learn.Momentum.On":  "false",
				"Path.Learn.WtBal.On":     "false",
				"Path.Learn.Lrate":        "0.02",
				"Path.WtInit.Mean":        "0.3",
				"Path.WtInit.Var":         "0",
				"Path.Learn.XCal.SetLLrn": "true",
				"Path.Learn.XCal.LLrn":    "1",
				"Path.Learn.XCal.MLrn":    "0",
			}},
		{Sel: "Layer", Desc: "high inhibition, layer act avg",
			Params: params.Params{
				"Layer.Act.XX1.Gain":       "20",
				"Layer.Act.Dt.VmTau":       "10",
				"Layer.Act.Init.Decay":     "0",
				"Layer.Learn.AvgL.Gain":    "0.6", // critical params here
				"Layer.Learn.AvgL.Init":    "0.2",
				"Layer.Learn.AvgL.Min":     "0.1",
				"Layer.Learn.AvgL.Tau":     "100",
				"Layer.Inhib.Layer.Gi":     "1.3",
				"Layer.Inhib.Layer.FB":     "0.5",
				"Layer.Inhib.ActAvg.Init":  "0.05",
				"Layer.Inhib.ActAvg.Fixed": "true",
			}},
		{Sel: "#Reach", Desc: "higher gain, inhib",
			Params: params.Params{
				"Layer.Act.XX1.Gain":   "100",
				"Layer.Inhib.Layer.Gi": "1.7",
				"Layer.Inhib.Layer.FB": "1",
			}},
		{Sel: "#GazeExpect", Desc: "higher inhib",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "2",
			}},
		{Sel: "#LocationToHidden", Desc: "strong",
			Params: params.Params{
				"Path.WtScale.Abs": "1.5",
			}},
		{Sel: "#CoverToHidden", Desc: "strong, slow",
			Params: params.Params{
				"Path.WtScale.Abs": "1.5",
				"Path.Learn.Lrate": "0.005",
			}},
		{Sel: "#ToyToHidden", Desc: "strong, slow",
			Params: params.Params{
				"Path.WtScale.Abs": "1.5",
				"Path.Learn.Lrate": "0.005",
			}},
		{Sel: "#HiddenToHidden", Desc: "recurrent",
			Params: params.Params{
				"Path.WtInit.Mean": "0.4",
				"Path.Learn.Learn": "false",
			}},
		{Sel: "#HiddenToGazeExpect", Desc: "strong",
			Params: params.Params{
				"Path.WtScale.Abs": "1.5",
			}},
		{Sel: "#GazeExpectToGazeExpect", Desc: "recurrent",
			Params: params.Params{
				"Path.WtInit.Mean": "0.3",
				"Path.Learn.Learn": "false",
			}},
		{Sel: "#HiddenToReach", Desc: "no learn to reach",
			Params: params.Params{
				"Path.Learn.Learn": "false",
			}},
	},
}

// Config has config parameters related to running the sim
type Config struct {
	// total number of runs to do when running Train
	NRuns int `default:"1" min:"1"`

	// total number of epochs per run
	NEpochs int `default:"1"`

	// how often to run through all the test patterns, in terms of training epochs.
	// can use 0 or -1 for no testing.
	TestInterval int `default:"-1"`
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {

	// which delay to use -- pres Init when changing
	Delay Delays

	// strength of recurrent weight in Hidden layer from each unit back to self
	RecurrentWt float32 `default:"0.4" step:"0.01"`

	// Config contains misc configuration parameters for running the sim
	Config Config `new-window:"+" display:"no-inline"`

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *leabra.Network `new-window:"+" display:"no-inline"`

	// network parameter management
	Params emer.NetParams `display:"add-fields"`

	// delay = 3 training patterns
	Delay3 *table.Table `new-window:"+" display:"no-inline"`

	// delay = 5 training patterns
	Delay5 *table.Table `new-window:"+" display:"no-inline"`

	// delay = 1 training patterns
	Delay1 *table.Table `new-window:"+" display:"no-inline"`

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
	ss.Net = leabra.NewNetwork("AnotB")
	ss.Params.Config(ParamSets, "", "", ss.Net)
	ss.Stats.Init()
	ss.Delay3 = &table.Table{}
	ss.Delay5 = &table.Table{}
	ss.Delay1 = &table.Table{}
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
}

func (ss *Sim) Defaults() {
	ss.RecurrentWt = 0.4
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
	ss.OpenPatAsset(ss.Delay3, "a_not_b_delay3.tsv", "AnotB Delay=3", "AnotB input patterns")
	ss.OpenPatAsset(ss.Delay5, "a_not_b_delay5.tsv", "AnotB Delay=5", "AnotB input patterns")
	ss.OpenPatAsset(ss.Delay1, "a_not_b_delay1.tsv", "AnotB Delay=1", "AnotB input patterns")
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var trn *env.FixedTable
	if len(ss.Envs) == 0 {
		trn = &env.FixedTable{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*env.FixedTable)
	}

	// note: names must be standard here!
	trn.Name = etime.Train.String()
	trn.Sequential = true

	switch ss.Delay {
	case Delay3:
		trn.Table = table.NewIndexView(ss.Delay3)
	case Delay5:
		trn.Table = table.NewIndexView(ss.Delay5)
	case Delay1:
		trn.Table = table.NewIndexView(ss.Delay1)
	}

	trn.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	loc := net.AddLayer2D("Location", 1, 3, leabra.InputLayer)
	cvr := net.AddLayer2D("Cover", 1, 2, leabra.InputLayer)
	toy := net.AddLayer2D("Toy", 1, 2, leabra.InputLayer)
	hid := net.AddLayer2D("Hidden", 1, 3, leabra.SuperLayer)
	gze := net.AddLayer2D("GazeExpect", 1, 3, leabra.SuperLayer)
	rch := net.AddLayer2D("Reach", 1, 3, leabra.CompareLayer)

	full := paths.NewFull()
	self := paths.NewOneToOne()
	net.ConnectLayers(loc, hid, full, leabra.ForwardPath)
	net.ConnectLayers(cvr, hid, full, leabra.ForwardPath)
	net.ConnectLayers(toy, hid, full, leabra.ForwardPath)
	net.ConnectLayers(hid, hid, self, leabra.LateralPath)
	net.ConnectLayers(hid, gze, full, leabra.ForwardPath)
	net.ConnectLayers(hid, rch, full, leabra.ForwardPath)
	net.ConnectLayers(gze, gze, self, leabra.LateralPath)

	cvr.PlaceRightOf(loc, 1)
	toy.PlaceRightOf(cvr, 1)
	hid.PlaceAbove(cvr)
	hid.Pos.XOffset = -1
	gze.PlaceAbove(hid)
	gze.Pos.XOffset = -4
	rch.PlaceRightOf(gze, 4)

	net.Build()
	net.Defaults()
	ss.ApplyParams()
	ss.InitWeights(net)
}

func (ss *Sim) InitWeights(net *leabra.Network) {
	net.InitWeights()
	hid := ss.Net.LayerByName("Hidden")
	fmloc := errors.Log1(hid.RecvPathBySendName("Location")).(*leabra.Path)
	gze := ss.Net.LayerByName("GazeExpect")
	hidgze := errors.Log1(gze.RecvPathBySendName("Hidden")).(*leabra.Path)
	rch := ss.Net.LayerByName("Reach")
	hidrch := errors.Log1(rch.RecvPathBySendName("Hidden")).(*leabra.Path)
	for i := 0; i < 3; i++ {
		fmloc.SetSynValue("Wt", i, i, 0.7)
		hidgze.SetSynValue("Wt", i, i, 0.7)
		hidrch.SetSynValue("Wt", i, i, 0.7)
	}
}

func (ss *Sim) ApplyParams() {
	ss.Params.SetAll()

	hid := ss.Net.LayerByName("Hidden")
	fmhid := errors.Log1(hid.RecvPathBySendName("Hidden")).(*leabra.Path)
	fmhid.WtInit.Mean = float64(ss.RecurrentWt)

	ev := ss.Envs.ByMode(etime.Train).(*env.FixedTable)

	if ss.Loops != nil {
		trn := ss.Loops.Stacks[etime.Train]
		trn.Loops[etime.Run].Counter.Max = ss.Config.NRuns
		trn.Loops[etime.Epoch].Counter.Max = ss.Config.NEpochs
		trn.Loops[etime.Trial].Counter.Max = ev.Table.Table.Rows
	}
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

	ev := ss.Envs.ByMode(etime.Train).(*env.FixedTable)
	trls := ev.Table.Table.Rows

	ls.AddStack(etime.Train).
		AddTime(etime.Run, ss.Config.NRuns).
		AddTime(etime.Epoch, ss.Config.NEpochs).
		AddTime(etime.Trial, trls).
		AddTime(etime.Cycle, 16)

	leabra.LooperStdPhases(ls, &ss.Context, ss.Net, 12, 15)                // plus phase timing
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
	leabra.LooperResetLogBelow(ls, &ss.Logs)

	////////////////////////////////////////////
	// GUI

	leabra.LooperUpdateNetView(ls, &ss.ViewUpdate, ss.Net, ss.NetViewCounters)
	leabra.LooperUpdatePlots(ls, &ss.GUI)
	ls.Stacks[etime.Train].OnInit.Add("GUI-Init", func() { ss.GUI.UpdateWindow() })

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
	ev := evi.(*env.FixedTable)
	trlnm := ev.TrialName.Cur
	gpnm := ev.GroupName.Cur

	rch := ss.Net.LayerByName("Reach")
	if strings.Contains(trlnm, "choice") {
		rch.Type = leabra.CompareLayer
	} else {
		rch.Type = leabra.InputLayer
	}

	if gpnm != ss.Stats.String("GroupName") { // init at start of new group
		net.InitActs()
	}
	train := true
	if strings.Contains(trlnm, "delay") {
		train = false
	}
	if train {
		net.LrateMult(1.0)
	} else {
		net.LrateMult(0.0)
	}

	ss.Stats.SetString("TrialName", trlnm)
	ss.Stats.SetString("GroupName", gpnm)

	lays := net.LayersByType(leabra.InputLayer, leabra.TargetLayer, leabra.CompareLayer)
	net.InitExt()
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
	ctx.Reset()
	ctx.Mode = etime.Train
	ss.InitWeights(ss.Net)
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
}

/////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetFloat("SSE", 0.0)
	ss.Stats.SetFloat("RT", math.NaN())
	ss.Stats.SetString("TrialName", "")
	ss.Stats.SetString("GroupName", "")
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
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "TrialName", "GroupName", "Cycle"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName", "GroupName")

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Train, etime.Trial, "InputLayer", "SuperLayer", "CompareLayer")

	ss.Logs.PlotItems("Reach_Act")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Train, etime.Run)
	ss.Logs.NoPlot(etime.Train, etime.Epoch)

	ss.Logs.SetMeta(etime.Train, etime.Trial, "Type", "Bar")
	ss.Logs.SetMeta(etime.Train, etime.Trial, "XAxis", "TrialName")
	ss.Logs.SetMeta(etime.Train, etime.Trial, "XAxisRotation", "-45")
	ss.Logs.SetMeta(etime.Train, etime.Trial, "Reach_Act:TensorIndex", "-1")
	ss.Logs.SetMeta(etime.Train, etime.Trial, "Reach_Act:Min", "0")
	ss.Logs.SetMeta(etime.Train, etime.Trial, "Reach_Act:Max", "1")
	ss.Logs.SetMeta(etime.Train, etime.Trial, "Reach_Act:FixMin", "true")
	ss.Logs.SetMeta(etime.Train, etime.Trial, "Reach_Act:FixMax", "true")
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
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
	if time == etime.Trial {
		ss.GUI.UpdateTableView(etime.Train, etime.Trial)
	}
}

//////////////////////////////////////////////////////////////////////
// 		GUI

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.SceneXYZ().Camera.Pose.Pos.Set(0.1, 1.8, 3.5)
	nv.SceneXYZ().Camera.LookAt(math32.Vector3{0.1, 0.15, 0}, math32.Vector3{0, 1, 0})

	labs := []string{"    A    B    C", " C1  C2", "T1  T2"}
	nv.ConfigLabels(labs)

	lays := []string{"Location", "Cover", "Toy"}

	for li, lnm := range lays {
		ly := nv.LayerByName(lnm)
		lbl := nv.LabelByName(labs[li])
		lbl.Pose = ly.Pose
		lbl.Pose.Pos.Y += .2
		lbl.Pose.Pos.Z += .02
		lbl.Pose.Scale.SetMul(math32.Vector3{0.6, 0.1, 0.6})
		lbl.Styles.Text.WhiteSpace = styles.WhiteSpacePre
	}
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "A not B"
	ss.GUI.MakeBody(ss, "a_not_b", title, `explores how the development of PFC active maintenance abilities can help to make behavior more flexible, in the sense that it can rapidly shift with changes in the environment. The development of flexibility has been extensively explored in the context of Piaget's famous A-not-B task, where a toy is first hidden several times in one hiding location (A), and then hidden in a new location (B). Depending on various task parameters, young kids reliably reach back at A instead of updating to B. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch9/a_not_b/README.md">README.md on GitHub</a>.</p>`, readme)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 300
	nv.Options.Raster.Max = 100
	nv.SetNet(ss.Net)
	nv.Options.PathWidth = 0.005
	ss.ViewUpdate.Config(nv, etime.GammaCycle, etime.Cycle)
	ss.GUI.ViewUpdate = &ss.ViewUpdate
	ss.ConfigNetView(nv)
	nv.Current()

	ss.GUI.AddPlots(title, &ss.Logs)

	tv := ss.GUI.AddTableView(&ss.Logs, etime.Train, etime.Trial)
	tv.TensorDisplay.GridMinSize = 32
	tv.TensorDisplay.GridMaxSize = 32

	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.Loops)

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
			core.TheApp.OpenURL("https://github.com/CompCogNeuro/sims/blob/main/ch9/a_not_b/README.md")
		},
	})
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}
