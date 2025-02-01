// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// cats_dogs: This project explores a simple **semantic network** intended
// to represent a (very small) set of relationships among different
// features used to represent a set of entities in the world.
// In our case, we represent some features of cats and dogs:
// their color, size, favorite food, and favorite toy.
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
	"github.com/emer/emergent/v2/relpos"
	"github.com/emer/etensor/tensor/table"
	"github.com/emer/leabra/v2/leabra"
)

//go:embed cats_dogs_pats.tsv cats_dogs.wts
var content embed.FS

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
		{Sel: "Path", Desc: "no learning",
			Params: params.Params{
				"Path.Learn.Learn": "false",
			}},
		{Sel: "Layer", Desc: "generic params for all layers: lower gain, slower, soft clamp",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Init":  "0.25",
				"Layer.Inhib.ActAvg.Fixed": "true",
				"Layer.Inhib.Layer.FBTau":  "3", // this is key for smoothing bumps
				"Layer.Act.Clamp.Hard":     "false",
				"Layer.Act.Clamp.Gain":     "1",
				"Layer.Act.XX1.Gain":       "40",  // more graded -- key
				"Layer.Act.Dt.VmTau":       "4",   // a bit slower -- not as effective as FBTau
				"Layer.Act.Gbar.L":         "0.1", // needs lower leak
			}},
		{Sel: ".Id", Desc: "specific inhibition for identity, name",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "4.0",
			}},
	},
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *leabra.Network `new-window:"+" display:"no-inline"`

	// network parameter management
	Params emer.NetParams `display:"add-fields"`

	// contains looper control loops for running sim
	Loops *looper.Stacks `display:"-"`

	// contains computed statistic values
	Stats estats.Stats `display:"-"`

	// Contains all the logs and information about the logs.'
	Logs elog.Logs `display:"+"`

	// the patterns to use
	Patterns *table.Table `new-window:"+" display:"no-inline"`

	// Environments
	Envs env.Envs `display:"-"`

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
	ss.Net = leabra.NewNetwork("CatsAndDogs")
	ss.Params.Config(ParamSets, "", "", ss.Net)
	ss.Stats.Init()
	ss.Patterns = &table.Table{}
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
}

func (ss *Sim) Defaults() {
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

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var tst *env.FixedTable
	if len(ss.Envs) == 0 {
		tst = &env.FixedTable{}
	} else {
		tst = ss.Envs.ByMode(etime.Test).(*env.FixedTable)
	}

	tst.Name = etime.Test.String()
	tst.Config(table.NewIndexView(ss.Patterns))
	tst.Sequential = true
	tst.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(tst)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	name := net.AddLayer2D("Name", 1, 10, leabra.InputLayer)
	iden := net.AddLayer2D("Identity", 1, 10, leabra.InputLayer)
	color := net.AddLayer2D("Color", 1, 4, leabra.InputLayer)
	food := net.AddLayer2D("FavoriteFood", 1, 4, leabra.InputLayer)
	size := net.AddLayer2D("Size", 1, 3, leabra.InputLayer)
	spec := net.AddLayer2D("Species", 1, 2, leabra.InputLayer)
	toy := net.AddLayer2D("FavoriteToy", 1, 4, leabra.InputLayer)

	name.AddClass("Id") // share params
	iden.AddClass("Id")

	one2one := paths.NewOneToOne()
	full := paths.NewFull()

	net.BidirConnectLayers(name, iden, one2one)
	net.BidirConnectLayers(color, iden, full)
	net.BidirConnectLayers(food, iden, full)
	net.BidirConnectLayers(size, iden, full)
	net.BidirConnectLayers(spec, iden, full)
	net.BidirConnectLayers(toy, iden, full)

	iden.PlaceAbove(name)
	color.PlaceAbove(iden)
	// gend.Pos.XAlign = relpos.Right
	food.PlaceAbove(iden)
	food.Pos.XAlign = relpos.Right
	size.PlaceAbove(color)
	spec.PlaceAbove(color)
	spec.Pos.XAlign = relpos.Right
	spec.Pos.XOffset = 2
	toy.PlaceAbove(food)
	toy.Pos.XAlign = relpos.Right
	toy.Pos.XOffset = 1

	net.Build()
	net.Defaults()
	ss.ApplyParams()
	ss.InitWeights(net)
}

// InitWeights initializes weights to digit 8
func (ss *Sim) InitWeights(net *leabra.Network) {
	net.InitWeights()
	net.OpenWeightsFS(content, "cats_dogs.wts")
}

func (ss *Sim) ApplyParams() {
	ss.Params.SetAll()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Loops.ResetCounters()
	ss.InitRandSeed(0)
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

	ev := ss.Envs.ByMode(etime.Test).(*env.FixedTable)
	ntrls := ev.Table.Len()

	cycles := 100
	ls.AddStack(etime.Test).
		AddTime(etime.Epoch, 1).
		AddTime(etime.Trial, ntrls).
		AddTime(etime.Cycle, cycles)

	leabra.LooperStdPhases(ls, &ss.Context, ss.Net, cycles-50, cycles-1)
	leabra.LooperSimCycleAndLearn(ls, ss.Net, &ss.Context, &ss.ViewUpdate) // std algo code
	ls.Stacks[etime.Test].OnInit.Add("Init", func() { ss.Init() })

	for m, _ := range ls.Stacks {
		stack := ls.Stacks[m]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
	}

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
	lays := net.LayersByType(leabra.InputLayer, leabra.CompareLayer)
	net.InitExt()
	ss.Stats.SetString("TrialName", ev.TrialName.Cur)
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
	ss.Envs.ByMode(etime.Test).Init(0)
	ctx.Reset()
	ctx.Mode = etime.Test
	ss.InitWeights(ss.Net)
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
}

/////////////////////////////////////////////////////////////////////////
//   Patterns

func (ss *Sim) OpenPatterns() {
	dt := ss.Patterns
	dt.SetMetaData("name", "CatsAndDogs")
	dt.SetMetaData("desc", "Face testing patterns")
	errors.Log(dt.OpenFS(content, "cats_dogs_pats.tsv", table.Tab))
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetString("TrialName", "")
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
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Trial", "TrialName", "Cycle"})
}

// Harmony computes the harmony (excitatory net input Ge * Act)
func (ss *Sim) Harmony(nt *leabra.Network) float32 {
	harm := float32(0)
	nu := 0
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		for i := range ly.Neurons {
			nrn := &(ly.Neurons[i])
			harm += nrn.Ge * nrn.Act
			nu++
		}
	}
	if nu > 0 {
		harm /= float32(nu)
	}
	return harm
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Logs.AddCounterItems(etime.Trial, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.Test, etime.Trial, "TrialName")
	ss.Logs.AddStatAggItem("Harmony", etime.Trial, etime.Cycle)

	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "InputLayer", "CompareLayer")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	ss.Logs.PlotItems("Harmony")
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
		ss.StatCounters()
		ss.Stats.SetFloat32("Harmony", ss.Harmony(ss.Net))
	case time == etime.Trial:
		ss.StatCounters()
		ss.Logs.Log(mode, time) // also logs to file, etc
		return
	}
	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

////////////////////////////////////////////////////////////////
// 		GUI

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	labs := []string{" Chloe Socks Sylv Garf Fuzz Daisy Fido Spot Snoop Penny",
		" black white brown orange", "bugs grass scraps shoe", "small  med  large", "cat     dog", "string feath bone shoe"}
	nv.ConfigLabels(labs)

	lays := []string{"Name", "Color", "FavoriteFood", "Size", "Species", "FavoriteToy"}

	for li, lnm := range lays {
		ly := nv.LayerByName(lnm)
		lbl := nv.LabelByName(labs[li])
		lbl.Pose = ly.Pose
		lbl.Pose.Pos.Y += .2
		lbl.Pose.Pos.Z += .02
		lbl.Pose.Scale.SetMul(math32.Vector3{0.4, 0.08, 0.5})
	}

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1.5, 3.0)
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "CatsAndDogs"
	ss.GUI.MakeBody(ss, "cats_dogs", title, `This project explores a simple **semantic network** intended to represent a (very small) set of relationships among different features used to represent a set of entities in the world.  In our case, we represent some features of cats and dogs: their color, size, favorite food, and favorite toy. See <a href="https://github.com/CompCogNeuro/sims/blob/main/ch3/cats_dogs/README.md">README.md on GitHub</a>.</p>`, readme)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 300
	nv.Options.Raster.Max = 100
	nv.SetNet(ss.Net)
	ss.ViewUpdate.Config(nv, etime.Cycle, etime.Cycle)
	ss.GUI.ViewUpdate = &ss.ViewUpdate
	nv.Current()
	ss.ConfigNetView(nv)

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.Loops)

	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "README",
		Icon:    icons.FileMarkdown,
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.TheApp.OpenURL("https://github.com/CompCogNeuro/sims/blob/main/ch3/cats_dogs/README.md")
		},
	})
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}
