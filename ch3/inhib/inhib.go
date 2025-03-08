// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// inhib: This simulation explores how inhibitory interneurons can dynamically
// control overall activity levels within the network, by providing both
// feedforward and feedback inhibition to excitatory pyramidal neurons.
package main

//go:generate core generate -add-types

import (
	"embed"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/icons"
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
	"github.com/emer/emergent/v2/patgen"
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/etensor/tensor"
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
		{Sel: "Path", Desc: "no learning",
			Params: params.Params{
				"Path.Learn.Learn": "false",
				"Path.WtInit.Dist": "Uniform",
				"Path.WtInit.Mean": "0.25",
				"Path.WtInit.Var":  "0.2",
			}},
		{Sel: "Layer", Desc: "generic params for all layers: lower gain, slower, soft clamp",
			Params: params.Params{
				"Layer.Inhib.Layer.On":     "false",
				"Layer.Inhib.ActAvg.Init":  "0.2",
				"Layer.Inhib.ActAvg.Fixed": "true",
				"Layer.Act.Dt.GTau":        "40",
				"Layer.Act.Gbar.I":         "0.4",
				"Layer.Act.Gbar.L":         "0.1",
			}},
		{Sel: ".InhibLay", Desc: "generic params for all layers: lower gain, slower, soft clamp",
			Params: params.Params{
				"Layer.Act.XX1.Thr": "0.4", // essential for getting active early
			}},
		{Sel: ".InhibPath", Desc: "inhibitory projections",
			Params: params.Params{
				"Path.WtInit.Dist": "Uniform",
				"Path.WtInit.Mean": "0.5",
				"Path.WtInit.Var":  "0",
				"Path.WtInit.Sym":  "false",
			}},
	},
	"Untrained": {
		{Sel: ".Excite", Desc: "excitatory connections",
			Params: params.Params{
				"Path.WtInit.Dist": "Uniform",
				"Path.WtInit.Mean": "0.25",
				"Path.WtInit.Var":  "0.2",
			}},
	},
	"Trained": {
		{Sel: ".Excite", Desc: "excitatory connections",
			Params: params.Params{
				"Path.WtInit.Dist": "Gaussian",
				"Path.WtInit.Mean": "0.25",
				"Path.WtInit.Var":  "0.7",
			}},
	},
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	// if true, use the bidirectionally connected network,
	// otherwise use the simpler feedforward network.
	BidirNet bool

	// simulate trained weights by having higher variance and Gaussian
	// distributed weight values -- otherwise lower variance, uniform.
	TrainedWts bool

	// percent of active units in input layer (literally number of active units,
	// because input has 100 units total).
	InputPct float32 `default:"20" min:"5" max:"50" step:"1"`

	// use feedforward, feedback (FFFB) computed inhibition instead
	// of unit-level inhibition.
	FFFBInhib bool `default:"false"`

	// inhibitory conductance strength for inhibition into Hidden layer.
	HiddenGbarI float32 `default:"0.4" min:"0" step:"0.05"`

	// inhibitory conductance strength for inhibition into Inhib layer
	// (self-inhibition -- tricky!).
	InhibGbarI float32 `default:"0.75" min:"0" step:"0.05"`

	// feedforward (FF) inhibition relative strength: for FF projections into Inhib neurons.
	FFinhibWtScale float32 `default:"1" min:"0" step:"0.1"`

	// feedback (FB) inhibition relative strength: for projections into Inhib neurons.
	FBinhibWtScale float32 `default:"1" min:"0" step:"0.1"`

	// time constant (tau) for updating G conductances into Hidden neurons
	// Much slower than std default of 1.4.
	HiddenGTau float32 `default:"40" min:"1" step:"1"`

	// time constant (tau) for updating G conductances into Inhib neurons.
	// Much slower than std default of 1.4, but 2x faster than Hidden.
	InhibGTau float32 `default:"20" min:"1" step:"1"`

	// absolute weight scaling of projections from inhibition onto
	// hidden and inhib layers.  This must be set to 0 to turn off the
	// connection-based inhibition when using the FFFBInhib computed inbhition.
	FmInhibWtScaleAbs float32 `default:"1"`

	// the feedforward network -- click to view / edit parameters for layers, paths, etc
	NetFF *leabra.Network `new-window:"+" display:"no-inline"`

	// the bidirectional network -- click to view / edit parameters for layers, paths, etc
	NetBidir *leabra.Network `new-window:"+" display:"no-inline"`

	// network parameter management
	Params emer.NetParams `display:"add-fields"`

	// contains looper control loops for running sim
	LoopsFF    *looper.Stacks `display:"-"`
	LoopsBidir *looper.Stacks `display:"-"`

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

	NetviewFF    *netview.NetView `display:"-"`
	NetviewBidir *netview.NetView `display:"-"`

	// a list of random seeds to use for each run
	RandSeeds randx.Seeds `display:"-"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Defaults()
	ss.NetFF = leabra.NewNetwork("InhibFF")
	ss.NetBidir = leabra.NewNetwork("InhibBidir")
	ss.Params.Config(ParamSets, "", "", ss.Net())
	ss.Stats.Init()
	ss.Patterns = &table.Table{}
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
}

func (ss *Sim) Defaults() {
	ss.TrainedWts = false
	ss.InputPct = 20
	ss.FFFBInhib = false
	ss.HiddenGbarI = 0.4
	ss.InhibGbarI = 0.75
	ss.FFinhibWtScale = 1
	ss.FBinhibWtScale = 1
	ss.HiddenGTau = 40
	ss.InhibGTau = 20
	ss.FmInhibWtScaleAbs = 1
}

//////////////////////////////////////////////////////////////////////////////
// 		Configs

// Net returns the current active network
func (ss *Sim) Net() *leabra.Network {
	if ss.BidirNet {
		return ss.NetBidir
	} else {
		return ss.NetFF
	}
}

// Loops returns the current active looper
func (ss *Sim) Loops() *looper.Stacks {
	if ss.BidirNet {
		return ss.LoopsBidir
	} else {
		return ss.LoopsFF
	}
}

// ConfigAll configures all the elements using the standard functions
func (ss *Sim) ConfigAll() {
	ss.ConfigPatterns()
	ss.ConfigEnv()
	ss.ConfigNetFF(ss.NetFF)
	ss.ConfigNetBidir(ss.NetBidir)
	ss.ConfigLogs()
	ss.LoopsFF = ss.ConfigLoops(ss.NetFF)
	ss.LoopsBidir = ss.ConfigLoops(ss.NetBidir)
}

func (ss *Sim) ConfigNetFF(net *leabra.Network) {
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	inp := net.AddLayer2D("Input", 10, 10, leabra.InputLayer)
	hid := net.AddLayer2D("Hidden", 10, 10, leabra.SuperLayer)
	inh := net.AddLayer2D("Inhib", 10, 2, leabra.SuperLayer)
	inh.AddClass("InhibLay")

	full := paths.NewFull()
	net.ConnectLayers(inp, hid, full, leabra.ForwardPath).AddClass("Excite")
	net.ConnectLayers(hid, inh, full, leabra.BackPath)
	net.ConnectLayers(inp, inh, full, leabra.ForwardPath)
	net.ConnectLayers(inh, hid, full, leabra.InhibPath)
	net.ConnectLayers(inh, inh, full, leabra.InhibPath)

	inh.PlaceRightOf(hid, 2)

	net.Build()
	net.Defaults()
	ss.ApplyParams(net)
	ss.InitWeights(net)
}

func (ss *Sim) ConfigNetBidir(net *leabra.Network) {
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	inp := net.AddLayer2D("Input", 10, 10, leabra.InputLayer)
	hid := net.AddLayer2D("Hidden", 10, 10, leabra.SuperLayer)
	inh := net.AddLayer2D("Inhib", 10, 2, leabra.SuperLayer)
	inh.AddClass("InhibLay")
	hid2 := net.AddLayer2D("Hidden2", 10, 10, leabra.SuperLayer)
	inh2 := net.AddLayer2D("Inhib2", 10, 2, leabra.SuperLayer)
	inh2.AddClass("InhibLay")

	full := paths.NewFull()
	net.ConnectLayers(inp, hid, full, leabra.ForwardPath).AddClass("Excite")
	net.ConnectLayers(hid, inh, full, leabra.BackPath)
	net.ConnectLayers(inp, inh, full, leabra.ForwardPath)
	net.ConnectLayers(hid2, inh, full, leabra.ForwardPath)
	net.ConnectLayers(inh, hid, full, leabra.InhibPath)
	net.ConnectLayers(inh, inh, full, leabra.InhibPath)

	net.ConnectLayers(hid, hid2, full, leabra.ForwardPath).AddClass("Excite")
	net.ConnectLayers(hid2, hid, full, leabra.BackPath).AddClass("Excite")
	net.ConnectLayers(hid, inh2, full, leabra.ForwardPath)
	net.ConnectLayers(hid2, inh2, full, leabra.BackPath)
	net.ConnectLayers(inh2, hid2, full, leabra.InhibPath)
	net.ConnectLayers(inh2, inh2, full, leabra.InhibPath)

	inh.PlaceRightOf(hid, 2)
	inh2.PlaceRightOf(hid2, 2)
	hid2.PlaceAbove(hid)

	net.Build()
	net.Defaults()
	ss.ApplyParams(net)
	ss.InitWeights(net)
}

// InitWeights initializes weights to digit 8
func (ss *Sim) InitWeights(net *leabra.Network) {
	net.InitWeights()
}

func (ss *Sim) ApplyParams(net *leabra.Network) {
	ss.Params.Network = net
	ss.Params.SetAll()
	if ss.TrainedWts {
		ss.Params.SetAllSheet("Trained")
	} else {
		ss.Params.SetAllSheet("Untrained")
	}
	ffinhsc := ss.FFinhibWtScale
	if net == ss.NetBidir {
		ffinhsc *= 0.5 // 2 inhib prjns so .5 ea
	}
	hid := net.LayerByName("Hidden")
	hid.Act.Gbar.I = ss.HiddenGbarI
	hid.Act.Dt.GTau = ss.HiddenGTau
	hid.Act.Update()
	inh := net.LayerByName("Inhib")
	inh.Act.Gbar.I = ss.InhibGbarI
	inh.Act.Dt.GTau = ss.InhibGTau
	inh.Act.Update()
	ff := errors.Log1(inh.RecvPathBySendName("Input")).(*leabra.Path)
	ff.WtScale.Rel = ffinhsc
	fb := errors.Log1(inh.RecvPathBySendName("Hidden")).(*leabra.Path)
	fb.WtScale.Rel = ss.FBinhibWtScale
	hid.Inhib.Layer.On = ss.FFFBInhib
	inh.Inhib.Layer.On = ss.FFFBInhib
	fi := errors.Log1(hid.RecvPathBySendName("Inhib")).(*leabra.Path)
	fi.WtScale.Abs = ss.FmInhibWtScaleAbs
	fi = errors.Log1(inh.RecvPathBySendName("Inhib")).(*leabra.Path)
	fi.WtScale.Abs = ss.FmInhibWtScaleAbs
	if net == ss.NetBidir {
		hid = net.LayerByName("Hidden2")
		hid.Act.Gbar.I = ss.HiddenGbarI
		hid.Act.Dt.GTau = ss.HiddenGTau
		hid.Act.Update()
		inh = net.LayerByName("Inhib2")
		inh.Act.Gbar.I = ss.InhibGbarI
		inh.Act.Dt.GTau = ss.InhibGTau
		inh.Act.Update()
		hid.Inhib.Layer.On = ss.FFFBInhib
		inh.Inhib.Layer.On = ss.FFFBInhib
		fi = errors.Log1(hid.RecvPathBySendName("Inhib2")).(*leabra.Path)
		fi.WtScale.Abs = ss.FmInhibWtScaleAbs
		fi = errors.Log1(inh.RecvPathBySendName("Inhib2")).(*leabra.Path)
		fi.WtScale.Abs = ss.FmInhibWtScaleAbs
		ff = errors.Log1(inh.RecvPathBySendName("Hidden")).(*leabra.Path)
		ff.WtScale.Rel = ffinhsc
		fb = errors.Log1(inh.RecvPathBySendName("Hidden2")).(*leabra.Path)
		fb.WtScale.Rel = ss.FBinhibWtScale
		inh = net.LayerByName("Inhib")
		ff = errors.Log1(inh.RecvPathBySendName("Hidden2")).(*leabra.Path)
		ff.WtScale.Rel = ffinhsc
	}
}

func (ss *Sim) ConfigPatterns() {
	dt := ss.Patterns
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")
	dt.AddStringColumn("Name")
	dt.AddFloat32TensorColumn("Input", []int{10, 10}, "Y", "X")
	dt.SetNumRows(1)

	patgen.PermutedBinaryRows(dt.Columns[1].(*tensor.Float32), int(ss.InputPct), 1, 0)
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

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	if ss.BidirNet {
		ss.ViewUpdate.View = ss.NetviewBidir
	} else {
		ss.ViewUpdate.View = ss.NetviewFF
	}
	ss.LoopsFF.ResetCounters()
	ss.LoopsBidir.ResetCounters()
	// ss.InitRandSeed(0)
	ss.GUI.StopNow = false
	ss.ApplyParams(ss.Net())
	ss.NewRun()
	ss.ViewUpdate.RecordSyns()
	ss.ViewUpdate.Update()
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	ss.RandSeeds.Set(run)
	ss.RandSeeds.Set(run, &ss.Net().Rand)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops(net *leabra.Network) *looper.Stacks {
	ls := looper.NewStacks()

	ntrls := 10
	cycles := 200
	ls.AddStack(etime.Test).
		AddTime(etime.Epoch, 1).
		AddTime(etime.Trial, ntrls).
		AddTime(etime.Cycle, cycles)

	leabra.LooperStdPhases(ls, &ss.Context, net, cycles-25, cycles-1)
	leabra.LooperSimCycleAndLearn(ls, net, &ss.Context, &ss.ViewUpdate) // std algo code
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

	leabra.LooperUpdateNetView(ls, &ss.ViewUpdate, net, ss.NetViewCounters)
	leabra.LooperUpdatePlots(ls, &ss.GUI)
	ls.Stacks[etime.Test].OnInit.Add("GUI-Init", func() { ss.GUI.UpdateWindow() })
	return ls
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	ctx := &ss.Context
	net := ss.Net()
	ss.ApplyParams(net) // also apply the params
	ev := ss.Envs.ByMode(ctx.Mode).(*env.FixedTable)
	ev.Step()
	lays := net.LayersByType(leabra.InputLayer, leabra.CompareLayer)
	net.InitExt()
	ss.Stats.SetString("TrialName", ev.TrialName.Cur)
	for _, lnm := range lays {
		ly := ss.Net().LayerByName(lnm)
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
	ss.InitWeights(ss.Net())
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
	ss.ViewUpdate.RecordSyns()
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
	ss.Loops().Stacks[mode].CountersToStats(&ss.Stats)
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
	li := ss.Logs.AddStatAggItem("HiddenActAvg", etime.Trial, etime.Cycle)
	li.SetFixMin(true).SetFixMax(true)
	li = ss.Logs.AddStatAggItem("InhibActAvg", etime.Trial, etime.Cycle)
	li.SetFixMin(true).SetFixMax(true)

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net())
	ss.Logs.PlotItems("HiddenActAvg", "InhibActAvg")
}

func (ss *Sim) CycleStats() {
	layers := []string{"Hidden", "Inhib"}
	for _, lnm := range layers {
		ly := ss.Net().LayerByName(lnm)
		ss.Stats.SetFloat32(lnm+"ActAvg", ly.Pools[0].Inhib.Act.Avg)
	}
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
	title := "Inhib"
	ss.GUI.MakeBody(ss, "inhib", title, `inhib: This simulation explores how inhibitory interneurons can dynamically control overall activity levels within the network, by providing both feedforward and feedback inhibition to excitatory pyramidal neurons. See <a href="https://github.com/CompCogNeuro/sims/blob/main/ch3/inhib/README.md">README.md on GitHub</a>.</p>`, readme)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("FF Net")
	ss.NetviewFF = nv
	nv.Options.MaxRecs = 300
	nv.Options.Raster.Max = 100
	nv.SetNet(ss.NetFF)
	nv.Options.PathWidth = 0.005
	ss.ViewUpdate.Config(nv, etime.Cycle, etime.Cycle)
	ss.GUI.ViewUpdate = &ss.ViewUpdate
	nv.Current()

	nv = ss.GUI.AddNetView("Bidir Net")
	ss.NetviewBidir = nv
	nv.Options.MaxRecs = 300
	nv.Options.Raster.Max = 100
	nv.SetNet(ss.NetBidir)
	nv.Options.PathWidth = 0.005
	nv.Current()

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.LoopsFF, "FF")
	ss.GUI.AddLooperCtrl(p, ss.LoopsBidir, "Bidir")

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
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "ConfigPats",
		Icon:    icons.Image,
		Tooltip: "config patterns",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.CallFunc(ss.GUI.Body, ss.ConfigPatterns)
		},
	})
	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "README",
		Icon:    icons.FileMarkdown,
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.TheApp.OpenURL("https://github.com/CompCogNeuro/sims/blob/main/ch3/inhib/README.md")
		},
	})
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}
