// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
v1rf illustrates how self-organizing learning in response to natural images
produces the oriented edge detector receptive field properties of neurons
in primary visual cortex (V1). This provides insight into why the visual
system encodes information in the way it does, while also providing an
important test of the biological relevance of our computational models.
*/
package main

//go:generate core generate -add-types

import (
	"embed"
	"fmt"
	"os"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/system"
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
	"github.com/emer/etensor/tensor/table"
	"github.com/emer/etensor/tensor/tensorcore"
	"github.com/emer/leabra/v2/leabra"
)

//go:embed v1rf_img1.jpg v1rf_img2.jpg v1rf_img3.jpg v1rf_img4.jpg v1rf_rec2.wts.gz v1rf_rec05.wts.gz probes.tsv
var content embed.FS

//go:embed *.png README.md
var readme embed.FS

func main() {
	sim := &Sim{}
	sim.New()
	sim.ConfigAll()
	sim.RunGUI()
}

// see params.go for params, config.go for Config

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	// excitatory lateral (recurrent) WtScale.Rel value
	ExcitLateralScale float32 `def:"0.2"`

	// inhibitory lateral (recurrent) WtScale.Abs value
	InhibLateralScale float32 `def:"0.2"`

	// do excitatory lateral (recurrent) connections learn?
	ExcitLateralLearn bool `def:"true"`

	// simulation configuration parameters -- set by .toml config file and / or args
	Config Config `new-window:"+"`

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *leabra.Network `new-window:"+" display:"no-inline"`

	// all parameter management
	Params emer.NetParams `display:"add-fields"`

	// testing probe input paterns
	Probes *table.Table `new-window:"+" display:"no-inline"`

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
	ss.Config.Defaults()
	ss.Defaults()
	econfig.Config(&ss.Config, "config.toml")
	ss.Net = leabra.NewNetwork("V1 RF")
	ss.Probes = table.NewTable()
	ss.Params.Config(ParamSets, ss.Config.Params.Sheet, ss.Config.Params.Tag, ss.Net)
	ss.Stats.Init()
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
}

func (ss *Sim) Defaults() {
	ss.ExcitLateralScale = 0.2
	ss.InhibLateralScale = 0.2
	ss.ExcitLateralLearn = true
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) ConfigAll() {
	ss.OpenPatterns()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
	if ss.Config.Params.SaveAll {
		ss.Config.Params.SaveAll = false
		ss.Net.SaveParamsSnapshot(&ss.Params.Params, &ss.Config, ss.Config.Params.Good)
		os.Exit(0)
	}
}

func (ss *Sim) OpenPatterns() {
	ss.Probes.SetMetaData("name", "Probes")
	ss.Probes.SetMetaData("desc", "Probes testing patterns")
	errors.Log(ss.Probes.OpenFS(content, "probes.tsv", table.Tab))
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var trn *ImgEnv
	var tst *env.FixedTable
	if len(ss.Envs) == 0 {
		trn = &ImgEnv{}
		tst = &env.FixedTable{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*ImgEnv)
		tst = ss.Envs.ByMode(etime.Test).(*env.FixedTable)
	}

	trn.Name = etime.Train.String()
	trn.Defaults()
	trn.Trial.Max = ss.Config.Run.NTrials
	trn.ImageFiles = []string{"v1rf_img1.jpg", "v1rf_img2.jpg", "v1rf_img3.jpg", "v1rf_img4.jpg"}
	trn.OpenImagesFS(content)
	if ss.Config.Env.Env != nil {
		params.ApplyMap(trn, ss.Config.Env.Env, ss.Config.Debug)
	}

	tst.Name = etime.Test.String()
	tst.Trial.Max = ss.Config.Run.NTrials
	tst.Table = table.NewIndexView(ss.Probes)

	trn.Init(0)
	tst.Init(0)

	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	lgnOn := net.AddLayer2D("LGNon", 12, 12, leabra.InputLayer)
	lgnOff := net.AddLayer2D("LGNoff", 12, 12, leabra.InputLayer)
	v1 := net.AddLayer2D("V1", 14, 14, leabra.SuperLayer)

	lgnOn.Doc = "LGN (lateral geniculate nucleus of the thalamus), On-center neurons"
	lgnOff.Doc = "LGN (lateral geniculate nucleus of the thalamus), Off-center neurons"
	v1.Doc = "V1 (primary visual cortex), excitatory neurons with lateral excitatory and inhibitory connections"

	full := paths.NewFull()
	net.ConnectLayers(lgnOn, v1, full, leabra.ForwardPath)
	net.ConnectLayers(lgnOff, v1, full, leabra.ForwardPath)

	circ := paths.NewCircle()
	circ.TopoWeights = true
	circ.Radius = 4
	circ.Sigma = .75

	rec := net.ConnectLayers(v1, v1, circ, leabra.LateralPath)
	rec.AddClass("ExciteLateral")

	inh := net.ConnectLayers(v1, v1, full, leabra.InhibPath)
	inh.AddClass("InhibLateral")

	lgnOff.PlaceRightOf(lgnOn, 2)
	v1.PlaceAbove(lgnOn)
	v1.Pos.XOffset = 5

	net.Build()
	net.Defaults()
	ss.ApplyParams()
	ss.InitWeights()
}

func (ss *Sim) ApplyParams() {
	ss.Params.SetAll() // first hard-coded defaults
	nt := ss.Net
	v1 := nt.LayerByName("V1")
	elat := v1.RecvPaths[2]
	elat.WtScale.Rel = ss.ExcitLateralScale
	elat.Learn.Learn = ss.ExcitLateralLearn
	ilat := v1.RecvPaths[3]
	ilat.WtScale.Abs = ss.InhibLateralScale

	if ss.Config.Params.Network != nil {
		ss.Params.SetNetworkMap(ss.Net, ss.Config.Params.Network)
	}
}

func (ss *Sim) InitWeights() {
	ss.Net.InitWeights()
	ss.Net.InitTopoScales()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	if ss.Config.GUI {
		ss.Stats.SetString("RunName", ss.Params.RunName(0)) // in case user interactively changes tag
	}
	ss.Loops.ResetCounters()
	ss.InitRandSeed(0)
	// ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
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

	trls := ss.Config.Run.NTrials

	ls.AddStack(etime.Train).
		AddTime(etime.Run, ss.Config.Run.NRuns).
		AddTime(etime.Epoch, ss.Config.Run.NEpochs).
		AddTime(etime.Trial, trls).
		AddTime(etime.Cycle, 100)

	ls.AddStack(etime.Test).
		AddTime(etime.Epoch, 1).
		AddTime(etime.Trial, ss.Probes.Rows).
		AddTime(etime.Cycle, 100)

	leabra.LooperStdPhases(ls, &ss.Context, ss.Net, 75, 99)                // plus phase timing
	leabra.LooperSimCycleAndLearn(ls, ss.Net, &ss.Context, &ss.ViewUpdate) // std algo code

	ls.Stacks[etime.Train].OnInit.Add("Init", func() { ss.Init() })

	for m := range ls.Stacks {
		stack := ls.Stacks[m]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
	}

	ls.Loop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	// Add Testing
	trainEpoch := ls.Loop(etime.Train, etime.Epoch)
	trainEpoch.OnStart.Add("TestAtInterval", func() {
		ss.V1RFs()
		if (ss.Config.Run.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.Config.Run.TestInterval == 0) {
			// Note the +1 so that it doesn't occur at the 0th timestep.
			ss.TestAll()
		}
	})

	/////////////////////////////////////////////
	// Logging

	ls.AddOnEndToAll("Log", func(mode, time enums.Enum) {
		ss.Log(mode.(etime.Modes), time.(etime.Times))
	})
	leabra.LooperResetLogBelow(ls, &ss.Logs)

	// Save weights to file, to look at later
	ls.Loop(etime.Train, etime.Run).OnEnd.Add("SaveWeights", func() {
		ctrString := ss.Stats.PrintValues([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
		leabra.SaveWeightsIfConfigSet(ss.Net, ss.Config.Log.SaveWeights, ctrString, ss.Stats.String("RunName"))
	})

	////////////////////////////////////////////
	// GUI

	if !ss.Config.GUI {
		if ss.Config.Log.NetData {
			ls.Loop(etime.Test, etime.Trial).OnEnd.Add("NetDataRecord", func() {
				ss.GUI.NetDataRecord(ss.ViewUpdate.Text)
			})
		}
	} else {
		ls.Loop(etime.Test, etime.Trial).OnEnd.Add("ActRFs", func() {
			ss.Stats.UpdateActRFs(ss.Net, "ActM", 0.01, 0)
		})
		ls.Loop(etime.Train, etime.Trial).OnStart.Add("UpdateImage", func() {
			if system.TheApp.Platform() == system.Web { // todo: hangs on web
				return
			}
			ss.GUI.Grid("Image").NeedsRender()
		})
		ls.Loop(etime.Test, etime.Trial).OnStart.Add("UpdateImage", func() {
			if system.TheApp.Platform() == system.Web {
				return
			}
			ss.GUI.Grid("Image").NeedsRender()
		})

		leabra.LooperUpdateNetView(ls, &ss.ViewUpdate, ss.Net, ss.NetViewCounters)
		leabra.LooperUpdatePlots(ls, &ss.GUI)
		ls.Stacks[etime.Train].OnInit.Add("GUI-Init", func() { ss.GUI.UpdateWindow() })
		ls.Stacks[etime.Test].OnInit.Add("GUI-Init", func() { ss.GUI.UpdateWindow() })

	}

	if ss.Config.Debug {
		fmt.Println(ls.DocString())
	}
	ss.Loops = ls
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	ctx := &ss.Context
	net := ss.Net
	net.InitExt()
	lays := net.LayersByType(leabra.InputLayer, leabra.TargetLayer)

	ev := ss.Envs.ByMode(ctx.Mode)
	ev.Step()
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm)
		pats := ev.State(ly.Name)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
	if str, ok := ev.(fmt.Stringer); ok {
		ss.Stats.SetString("TrialName", str.String())
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
	ss.InitWeights()
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.Envs.ByMode(etime.Test).Init(0)
	ss.Stats.ActRFs.Reset()
	ss.Loops.ResetAndRun(etime.Test)
	ss.Loops.Mode = etime.Train // Important to reset Mode back to Train because this is called from within the Train Run.
	ss.Stats.ActRFsAvgNorm()
	ss.GUI.ViewActRFs(&ss.Stats.ActRFs)

}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.Logs.ResetLog(etime.Test, etime.Epoch) // only show last row
	ss.GUI.StopNow = false
	ss.TestAll()
	ss.GUI.Stopped()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetString("TrialName", "0")
	onValues := ss.Stats.F32Tensor("V1onWts")
	offValues := ss.Stats.F32Tensor("V1offWts")
	netValues := ss.Stats.F32Tensor("V1Wts")
	ss.ConfigWtTensors(onValues)
	ss.ConfigWtTensors(offValues)
	ss.ConfigWtTensors(netValues)
}

func (ss *Sim) ConfigWtTensors(dt *tensor.Float32) {
	dt.SetShape([]int{14, 14, 12, 12})
	dt.SetMetaData("grid-fill", "1")
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
	ss.Stats.SetString("TrialName", ss.Stats.String("TrialName"))
}

func (ss *Sim) NetViewCounters(tm etime.Times) {
	if ss.ViewUpdate.View == nil {
		return
	}
	if tm == etime.Trial {
		ss.TrialStats() // get trial stats for current di
	}
	ss.StatCounters()
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "Cat", "TrialName", "Cycle"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
}

func (ss *Sim) V1RFs() {
	onValues := ss.Stats.F32Tensor("V1onWts")
	offValues := ss.Stats.F32Tensor("V1offWts")
	netValues := ss.Stats.F32Tensor("V1Wts")
	on := ss.Net.LayerByName("LGNon")
	off := ss.Net.LayerByName("LGNoff")
	isz := on.Shape.Len()
	v1 := ss.Net.LayerByName("V1")
	ysz := v1.Shape.DimSize(0)
	xsz := v1.Shape.DimSize(1)
	for y := 0; y < ysz; y++ {
		for x := 0; x < xsz; x++ {
			ui := (y*xsz + x)
			ust := ui * isz
			onvls := onValues.Values[ust : ust+isz]
			offvls := offValues.Values[ust : ust+isz]
			netvls := netValues.Values[ust : ust+isz]
			on.SendPathValues(&onvls, "Wt", v1, ui, "")
			off.SendPathValues(&offvls, "Wt", v1, ui, "")
			for ui := 0; ui < isz; ui++ {
				netvls[ui] = 1.5 * (onvls[ui] - offvls[ui])
			}
		}
	}
	if system.TheApp.Platform() != system.Web { // todo: hangs on web
		ss.GUI.Grid("V1RFs").NeedsRender()
	}
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
	ss.Logs.SetMeta(etime.Test, etime.Epoch, "Type", "Bar")
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	if mode.String() != "Analyze" {
		ss.Context.Mode = mode // Also set specifically in a Loop callback.
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
		ss.Logs.LogRow(mode, time, row)
		return // don't do reg below
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "V1RF"
	ss.GUI.MakeBody(ss, "v1rf", title, `This simulation illustrates how self-organizing learning in response to natural images produces the oriented edge detector receptive field properties of neurons in primary visual cortex (V1). This provides insight into why the visual system encodes information in the way it does, while also providing an important test of the biological relevance of our computational models. See <a href="https://github.com/CompCogNeuro/sims/blob/main/ch6/v1rf/README.md">README.md on GitHub</a>.</p>`, readme)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 300
	nv.Options.Raster.Max = 100
	nv.Options.LayerNameSize = 0.03
	nv.SetNet(ss.Net)
	ss.ViewUpdate.Config(nv, etime.Phase, etime.Phase)

	cam := &(nv.SceneXYZ().Camera)
	cam.Pose.Pos.Set(0.0, 1.733, 2.3)
	cam.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))

	ss.GUI.ViewUpdate = &ss.ViewUpdate

	ss.GUI.AddPlots(title, &ss.Logs)

	itb, _ := ss.GUI.Tabs.NewTab("V1 RFs")
	tg := tensorcore.NewTensorGrid(itb).
		SetTensor(ss.Stats.F32Tensor("V1Wts"))
	ss.GUI.SetGrid("V1RFs", tg)

	itb, _ = ss.GUI.Tabs.NewTab("Image")
	tg = tensorcore.NewTensorGrid(itb).
		SetTensor(&ss.Envs.ByMode(etime.Train).(*ImgEnv).Vis.ImgTsr)
	ss.GUI.SetGrid("Image", tg)

	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.Loops)

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Test All",
		Icon:    icons.PlayArrow,
		Tooltip: "Tests a large same of testing items and records ActRFs.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.UpdateWindow()
				go ss.RunTestAll()
			}
		},
	})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "V1 RFs", Icon: icons.Open,
		Tooltip: "update the V1 receptive fields display",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.V1RFs()
		},
	})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Open Rec=0.2 Wts", Icon: icons.Open,
		Tooltip: "Opened weights from the recurrent weights = 0.2 case",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Net.OpenWeightsFS(content, "v1rf_rec2.wts.gz")
			ss.ViewUpdate.RecordSyns()
			ss.ViewUpdate.Update()
		},
	})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Open Rec=0.05 Wts", Icon: icons.Open,
		Tooltip: "Opened weights from the recurrent weights = 0.05 case",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Net.OpenWeightsFS(content, "v1rf_rec05.wts.gz")
			ss.ViewUpdate.RecordSyns()
			ss.ViewUpdate.Update()
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
			core.TheApp.OpenURL("https://github.com/CompCogNeuro/sims/blob/main/ch6/v1rf/README.md")
		},
	})
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}
