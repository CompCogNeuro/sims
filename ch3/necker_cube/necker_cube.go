// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// necker_cube: This simulation explores the use of constraint satisfaction
// in processing ambiguous stimuli, in this case the *Necker cube*, which
// can be viewed as a cube in one of two orientations, where people flip back and forth.
package main

//go:generate core generate -add-types

import (
	"embed"

	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/base/randx"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/estats"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/netview"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/leabra/v2/leabra"
)

//go:embed necker_cube.wts
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
		{Sel: "Path", Desc: "no learning",
			Params: params.Params{
				"Path.Learn.Learn": "false",
			}},
		{Sel: "Layer", Desc: "generic params for all layers: lower gain, slower, soft clamp",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Init":  "0.35",
				"Layer.Inhib.ActAvg.Fixed": "true",
				"Layer.Inhib.Layer.Gi":     "1.4", // need this for FB = 0.5 -- 1 works otherwise but not with adapt
				"Layer.Inhib.Layer.FB":     "0.5", // this is better for adapt dynamics: 1.0 not as clean of dynamics
				"Layer.Act.Clamp.Hard":     "false",
				"Layer.Act.Clamp.Gain":     "0.1",
				"Layer.Act.Dt.VmTau":       "6", // a bit slower -- not as effective as FBTau
				"Layer.Act.Noise.Dist":     "Gaussian",
				"Layer.Act.Noise.Var":      "0.01",
				"Layer.Act.Noise.Type":     "GeNoise",
				"Layer.Act.Noise.Fixed":    "false",
				"Layer.Act.KNa.Slow.Rise":  "0.005",
				"Layer.Act.KNa.Slow.Max":   "0.2",
				"Layer.Act.Gbar.K":         "1.2",
				"Layer.Act.Gbar.L":         "0.1", // this is important relative to .2
			}},
	},
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	// the variance parameter for Gaussian noise added to unit activations on every cycle
	Noise float32 `min:"0" step:"0.01"`

	// apply sodium-gated potassium adaptation mechanisms that cause the neuron to reduce spiking over time
	KNaAdapt bool

	// total number of cycles to run per trial; increase to 1,000 when testing adaptation
	Cycles int `default:"100,1000"`

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
	ss.Net = leabra.NewNetwork("NeckerCube")
	ss.Params.Config(ParamSets, "", "", ss.Net)
	ss.Stats.Init()
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
}

func (ss *Sim) Defaults() {
	ss.Noise = 0.01
	ss.KNaAdapt = false
	ss.Cycles = 100
}

//////////////////////////////////////////////////////////////////////////////
// 		Configs

// ConfigAll configures all the elements using the standard functions
func (ss *Sim) ConfigAll() {
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	nc := net.AddLayer4D("NeckerCube", 1, 2, 4, 2, leabra.InputLayer)

	full := paths.NewFull()
	net.ConnectLayers(nc, nc, full, leabra.LateralPath)

	net.Build()
	net.Defaults()
	ss.ApplyParams()
	ss.InitWeights(net)
}

// InitWeights initializes weights to digit 8
func (ss *Sim) InitWeights(net *leabra.Network) {
	net.InitWeights()
	net.OpenWeightsFS(content, "necker_cube.wts")
}

func (ss *Sim) ApplyParams() {
	ss.Params.SetAll()
	ly := ss.Net.LayerByName("NeckerCube")
	ly.Act.Noise.Var = float64(ss.Noise)
	ly.Act.KNa.On = ss.KNaAdapt
	ly.Act.Update()
	if ss.Loops != nil {
		cyc := ss.Loops.Stacks[etime.Test].Loops[etime.Cycle]
		cyc.Counter.Max = ss.Cycles
		cyc.EventByName("Quarter1").AtCounter = ss.Cycles / 4
		cyc.EventByName("Quarter2").AtCounter = 2 * (ss.Cycles / 4)
		cyc.EventByName("MinusPhase:End").AtCounter = 3 * (ss.Cycles / 4)
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

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	ls := looper.NewStacks()

	ntrls := 100
	cycles := ss.Cycles
	ls.AddStack(etime.Test).
		AddTime(etime.Epoch, 1).
		AddTime(etime.Trial, ntrls).
		AddTime(etime.Cycle, cycles)

	leabra.LooperStdPhases(ls, &ss.Context, ss.Net, cycles-25, cycles-1)
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
	net := ss.Net
	net.InitExt()
	ly := net.LayerByName("NeckerCube")
	tsr := ss.Stats.F32Tensor("Inputs")
	tsr.SetShape([]int{16})
	if tsr.Float1D(0) != 1 {
		for i := range tsr.Values {
			tsr.Values[i] = 1
		}
	}
	ly.ApplyExt(tsr)
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ctx := &ss.Context
	ctx.Reset()
	ctx.Mode = etime.Test
	ss.InitWeights(ss.Net)
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
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
	ss.Logs.AddStatAggItem("GknaFast", etime.Trial, etime.Cycle)
	ss.Logs.AddStatAggItem("GknaMed", etime.Trial, etime.Cycle)
	ss.Logs.AddStatAggItem("GknaSlow", etime.Trial, etime.Cycle)

	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "InputLayer")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	ss.Logs.PlotItems("Harmony")
}

func (ss *Sim) CycleStats() {
	ss.Stats.SetFloat32("Harmony", ss.Harmony(ss.Net))
	ly := ss.Net.LayerByName("NeckerCube")
	ss.Stats.SetFloat32("GknaFast", ly.Neurons[0].GknaFast)
	ss.Stats.SetFloat32("GknaMed", ly.Neurons[0].GknaMed)
	ss.Stats.SetFloat32("GknaSlow", ly.Neurons[0].GknaSlow)
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
		ss.CycleStats()
	case time == etime.Trial:
		ss.StatCounters()
		ss.Logs.Log(mode, time) // also logs to file, etc
		return
	}
	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

////////////////////////////////////////////////////////////////
// 		GUI

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "NeckerCube"
	ss.GUI.MakeBody(ss, "necker_cube", title, `necker_cube: This simulation explores the use of constraint satisfaction in processing ambiguous stimuli, in this case the *Necker cube*, which can be viewed as a cube in one of two orientations, where people flip back and forth. See <a href="https://github.com/CompCogNeuro/sims/blob/main/ch3/necker_cube/README.md">README.md on GitHub</a>.</p>`, readme)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
	nv.Options.Raster.Max = 100
	nv.Options.MaxRecs = 300
	nv.SetNet(ss.Net)
	ss.ViewUpdate.Config(nv, etime.Cycle, etime.Cycle)
	ss.GUI.ViewUpdate = &ss.ViewUpdate
	nv.Current()

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.Loops)

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
			core.TheApp.OpenURL("https://github.com/CompCogNeuro/sims/blob/main/ch3/necker_cube/README.md")
		},
	})
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}
