// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// faces: This project explores how sensory inputs
// (in this case simple cartoon faces) can be categorized
// in multiple different ways, to extract the relevant information
// and collapse across the irrelevant.
// It allows you to explore both bottom-up processing from face image
// to categories, and top-down processing from category values to
// face images (imagery), including the ability to dynamically iterate
// both bottom-up and top-down to cleanup partial inputs
// (partially occluded face images).
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
	"github.com/emer/etensor/plot/plotcore"
	"github.com/emer/etensor/tensor"
	"github.com/emer/etensor/tensor/stats/clust"
	"github.com/emer/etensor/tensor/stats/metric"
	"github.com/emer/etensor/tensor/table"
	"github.com/emer/leabra/v2/leabra"
	"golang.org/x/exp/rand"
)

//go:embed faces.tsv partial_faces.tsv faces.wts
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
		{Sel: "Layer", Desc: "fix expected activity levels, reduce leak",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Init":  "0.15",
				"Layer.Inhib.ActAvg.Fixed": "true",
				"Layer.Act.Gbar.L":         "0.1", // needs lower leak
			}},
		{Sel: "#Input", Desc: "specific inhibition",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "2.0",
				"Layer.Act.Clamp.Hard": "false",
				"Layer.Act.Clamp.Gain": "0.2",
			}},
		{Sel: "#Identity", Desc: "specific inhbition",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "3.6",
			}},
		{Sel: "#Gender", Desc: "specific inhbition",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "1.6",
			}},
		{Sel: "#Emotion", Desc: "specific inhbition",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "1.3",
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

	// the partial patterns to use
	PartialPatterns *table.Table `new-window:"+" display:"no-inline"`

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
	ss.Net = leabra.NewNetwork("Faces")
	ss.Params.Config(ParamSets, "", "", ss.Net)
	ss.Stats.Init()
	ss.Patterns = &table.Table{}
	ss.PartialPatterns = &table.Table{}
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

	inp := net.AddLayer2D("Input", 16, 16, leabra.InputLayer)
	inp.Doc = "Input represents visual image inputs in a simple cartoon form."
	emo := net.AddLayer2D("Emotion", 1, 2, leabra.CompareLayer)
	emo.Doc = "Emotion has synaptic weights that detect the facial features in the eyes and mouth that specifically reflect emotions."
	gend := net.AddLayer2D("Gender", 1, 2, leabra.CompareLayer)
	gend.Doc = "Gender has synaptic weights that detect the hair and face features that discriminate male vs female gender in this simplified cartoon set of inputs."
	iden := net.AddLayer2D("Identity", 1, 10, leabra.CompareLayer)
	iden.Doc = "Identity detects individual faces."

	full := paths.NewFull()

	net.BidirConnectLayers(inp, emo, full)
	net.BidirConnectLayers(inp, gend, full)
	net.BidirConnectLayers(inp, iden, full)

	emo.PlaceAbove(inp)
	gend.PlaceAbove(inp)
	gend.Pos.XAlign = relpos.Right
	iden.PlaceBehind(emo, 4)
	iden.Pos.XOffset = 3

	net.Build()
	net.Defaults()
	ss.ApplyParams()
	ss.InitWeights(net)
}

// InitWeights initializes weights to digit 8
func (ss *Sim) InitWeights(net *leabra.Network) {
	net.InitWeights()
	net.OpenWeightsFS(content, "faces.wts")
}

func (ss *Sim) ApplyParams() {
	ss.Params.SetAll()
}

// SetInput sets whether the input to the network comes in bottom-up
// (Input layer) or top-down (Higher-level category layers)
func (ss *Sim) SetInput(topDown bool) { //types:add
	inp := ss.Net.LayerByName("Input")
	emo := ss.Net.LayerByName("Emotion")
	gend := ss.Net.LayerByName("Gender")
	iden := ss.Net.LayerByName("Identity")
	if topDown {
		inp.Type = leabra.CompareLayer
		emo.Type = leabra.InputLayer
		gend.Type = leabra.InputLayer
		iden.Type = leabra.InputLayer
	} else {
		inp.Type = leabra.InputLayer
		emo.Type = leabra.CompareLayer
		gend.Type = leabra.CompareLayer
		iden.Type = leabra.CompareLayer
	}
}

// SetPatterns selects which patterns to present: full or partial faces
func (ss *Sim) SetPatterns(partial bool) { //types:add
	ev := ss.Envs.ByMode(etime.Test).(*env.FixedTable)
	if partial {
		ev.Table = table.NewIndexView(ss.PartialPatterns)
		ev.Init(0)
	} else {
		ev.Table = table.NewIndexView(ss.Patterns)
		ev.Init(0)
	}
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

	ls.AddStack(etime.Test).
		AddTime(etime.Epoch, 1).
		AddTime(etime.Trial, ntrls).
		AddTime(etime.Cycle, 20)

	leabra.LooperStdPhases(ls, &ss.Context, ss.Net, 15, 19)
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
	dt.SetMetaData("name", "Faces")
	dt.SetMetaData("desc", "Face testing patterns")
	errors.Log(dt.OpenFS(content, "faces.tsv", table.Tab))
	dt = ss.PartialPatterns
	dt.SetMetaData("name", "FacesPartial")
	dt.SetMetaData("desc", "Patrial face testing patterns")
	errors.Log(dt.OpenFS(content, "partial_faces.tsv", table.Tab))
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
	if tm == etime.Trial {
		ss.TrialStats() // get trial stats for current di
	}
	ss.StatCounters()
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Trial", "TrialName", "Cycle"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Logs.AddCounterItems(etime.Trial, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.Test, etime.Trial, "TrialName")

	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "InputLayer", "CompareLayer")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	ss.Logs.NoPlot(etime.Test, etime.Cycle)
	ss.Logs.PlotItems("Emotion_Act", "Gender_Act", "Identity_Act")
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
}

// ClusterPlots computes all the cluster plots from the faces input data.
func (ss *Sim) ClusterPlots() {
	ptix := table.NewIndexView(ss.Patterns)
	estats.ClusterPlot(ss.GUI.PlotByName("ClustFaces"), ptix, "Input", "Name", clust.MinDist)
	estats.ClusterPlot(ss.GUI.PlotByName("ClustEmote"), ptix, "Emotion", "Name", clust.MinDist)
	estats.ClusterPlot(ss.GUI.PlotByName("ClustGend"), ptix, "Gender", "Name", clust.MinDist)
	estats.ClusterPlot(ss.GUI.PlotByName("ClustIdent"), ptix, "Identity", "Name", clust.MinDist)
	ss.ProjectionPlot()
}

func (ss *Sim) ProjectionPlot() {
	rvec0 := ss.Stats.F32Tensor("rvec0")
	rvec1 := ss.Stats.F32Tensor("rvec1")
	rvec0.SetShape([]int{256})
	rvec1.SetShape([]int{256})
	for i := range rvec1.Values {
		rvec0.Values[i] = .15 * (2*rand.Float32() - 1)
		rvec1.Values[i] = .15 * (2*rand.Float32() - 1)
	}

	tst := ss.Logs.Table(etime.Test, etime.Trial)
	nr := tst.Rows
	dt := ss.Logs.MiscTable("ProjectionTable")
	ss.ConfigProjectionTable(dt)

	for r := 0; r < nr; r++ {
		// single emotion dimension from sad to happy
		emote := 0.5*tst.TensorFloat1D("Emotion_Act", r, 0) + -0.5*tst.TensorFloat1D("Emotion_Act", r, 1)
		emote += .1 * (2*rand.Float64() - 1) // some jitter so labels are readable
		// single geneder dimension from male to femail
		gend := 0.5*tst.TensorFloat1D("Gender_Act", r, 0) + -0.5*tst.TensorFloat1D("Gender_Act", r, 1)
		gend += .1 * (2*rand.Float64() - 1) // some jitter so labels are readable
		input := tst.Tensor("Input_Act", r).(*tensor.Float32)
		rprjn0 := metric.InnerProduct32(rvec0.Values, input.Values)
		rprjn1 := metric.InnerProduct32(rvec1.Values, input.Values)
		dt.SetFloat("Trial", r, tst.Float("Trial", r))
		dt.SetString("TrialName", r, tst.StringValue("TrialName", r))
		dt.SetFloat("GendPrjn", r, gend)
		dt.SetFloat("EmotePrjn", r, emote)
		dt.SetFloat("RndPrjn0", r, float64(rprjn0))
		dt.SetFloat("RndPrjn1", r, float64(rprjn1))
	}

	plt := ss.GUI.PlotByName("ProjectionRandom")
	plt.Options.Title = "Face Random Projection Plot"
	plt.Options.XAxis = "RndPrjn0"
	plt.SetTable(dt)
	plt.Options.Lines = false
	plt.Options.Points = true
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColumnOptions("TrialName", plotcore.On, plotcore.FloatMin, 0, plotcore.FloatMax, 0)
	plt.SetColumnOptions("GendPrjn", plotcore.Off, plotcore.FixMin, -1, plotcore.FixMax, 1)
	plt.SetColumnOptions("RndPrjn0", plotcore.Off, plotcore.FloatMin, -1, plotcore.FloatMax, 1)
	plt.SetColumnOptions("RndPrjn1", plotcore.On, plotcore.FloatMin, -1, plotcore.FloatMax, 1)

	plt = ss.GUI.PlotByName("ProjectionEmoteGend")
	plt.Options.Title = "Face Emotion / Gender Projection Plot"
	plt.Options.XAxis = "GendPrjn"
	plt.SetTable(dt)
	plt.Options.Lines = false
	plt.Options.Points = true
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColumnOptions("TrialName", plotcore.On, plotcore.FloatMin, 0, plotcore.FloatMax, 0)
	plt.SetColumnOptions("GendPrjn", plotcore.Off, plotcore.FixMin, -1, plotcore.FixMax, 1)
	plt.SetColumnOptions("EmotePrjn", plotcore.On, plotcore.FixMin, -1, plotcore.FixMax, 1)
}

func (ss *Sim) ConfigProjectionTable(dt *table.Table) {
	dt.SetMetaData("name", "ProjectionTable")
	dt.SetMetaData("desc", "projection of data onto dimension")
	dt.SetMetaData("read-only", "true")

	if dt.NumColumns() == 0 {
		dt.AddIntColumn("Trial")
		dt.AddStringColumn("TrialName")
		dt.AddFloat64Column("GendPrjn")
		dt.AddFloat64Column("EmotePrjn")
		dt.AddFloat64Column("RndPrjn0")
		dt.AddFloat64Column("RndPrjn1")
	}
	ev := ss.Envs.ByMode(etime.Test).(*env.FixedTable)
	nt := ev.Table.Len() // number in indexview
	dt.SetNumRows(nt)
}

////////////////////////////////////////////////////////////////
// 		GUI

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	labs := []string{"happy sad", "female  male", "Albt Bett Lisa Mrk Wnd Zane"}
	nv.ConfigLabels(labs)
	emot := nv.LayerByName("Emotion")
	hs := nv.LabelByName(labs[0])
	hs.Pose = emot.Pose
	hs.Pose.Pos.Y += .1
	hs.Pose.Scale.SetMulScalar(0.5)
	hs.Pose.RotateOnAxis(0, 1, 0, 180)

	gend := nv.LayerByName("Gender")
	fm := nv.LabelByName(labs[1])
	fm.Pose = gend.Pose
	fm.Pose.Pos.X -= .05
	fm.Pose.Pos.Y += .1
	fm.Pose.Scale.SetMulScalar(0.5)
	fm.Pose.RotateOnAxis(0, 1, 0, 180)

	id := nv.LayerByName("Identity")
	nms := nv.LabelByName(labs[2])
	nms.Pose = id.Pose
	nms.Pose.Pos.Y += .1
	nms.Pose.Scale.SetMulScalar(0.5)
	nms.Pose.RotateOnAxis(0, 1, 0, 180)

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1.7, 2.37)
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "Faces"
	ss.GUI.MakeBody(ss, "faces", title, `This project explores how sensory inputs (in this case simple cartoon faces) can be categorized in multiple different ways, to extract the relevant information and collapse across the irrelevant. It allows you to explore both bottom-up processing from face image to categories, and top-down processing from category values to face images (imagery), including the ability to dynamically iterate both bottom-up and top-down to cleanup partial inputs (partially occluded face images). See <a href="https://github.com/CompCogNeuro/sims/blob/main/ch3/faces/README.md">README.md on GitHub</a>.</p>`, readme)
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

	ss.GUI.AddMiscPlotTab("ClustFaces")
	ss.GUI.AddMiscPlotTab("ClustEmote")
	ss.GUI.AddMiscPlotTab("ClustGend")
	ss.GUI.AddMiscPlotTab("ClustIdent")
	ss.GUI.AddMiscPlotTab("ProjectionRandom")
	ss.GUI.AddMiscPlotTab("ProjectionEmoteGend")

	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.Loops)

	////////////////////////////////////////////////
	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Set Input",
		Icon:    icons.Image,
		Tooltip: "set whether the input comes from the bottom-up (Input layer) or top-down (higher-level Category layers)",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.CallFunc(ss.GUI.Body, ss.SetInput)
		},
	})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Set Patterns",
		Icon:    icons.Image,
		Tooltip: "set which set of patterns to present: full or partial faces",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.CallFunc(ss.GUI.Body, ss.SetPatterns)
		},
	})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Cluster Plot",
		Icon:    icons.Image,
		Tooltip: "tests all the patterns and generates cluster plots and projections onto different dimensions",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.ClusterPlots()
		},
	})
	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "README",
		Icon:    icons.FileMarkdown,
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.TheApp.OpenURL("https://github.com/CompCogNeuro/sims/blob/main/ch3/faces/README.md")
		},
	})
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}
