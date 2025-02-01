// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// sem is trained using Hebbian learning on paragraphs from an early draft
// of the *Computational Explorations..* textbook, allowing it to learn about
// the overall statistics of when different words co-occur with other words,
// and thereby learning a surprisingly capable (though clearly imperfect)
// level of semantic knowledge about the topics covered in the textbook.
// This replicates the key results from the Latent Semantic Analysis
// research by Landauer and Dumais (1997).
package main

//go:generate core generate -add-types

import (
	"embed"
	"strings"

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
	"github.com/emer/leabra/v2/leabra"
)

//go:embed cecn_lg_f5.text cecn_lg_f5.words quiz.text trained_rec05.wts.gz
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
		{Sel: "Path", Desc: "no extra learning factors, hebbian learning",
			Params: params.Params{
				"Path.Learn.Norm.On":      "false",
				"Path.Learn.Momentum.On":  "false",
				"Path.Learn.WtBal.On":     "false",
				"Path.Learn.XCal.MLrn":    "0", // pure hebb
				"Path.Learn.XCal.SetLLrn": "true",
				"Path.Learn.XCal.LLrn":    "1",
				"Path.Learn.WtSig.Gain":   "1", // key: more graded weights
			}},
		{Sel: "Layer", Desc: "needs some special inhibition and learning params",
			Params: params.Params{
				"Layer.Act.Gbar.L": "0.1", // note: .2 in E1 but fine as .1
			}},
		{Sel: "#Input", Desc: "weak act",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Init":  "0.02",
				"Layer.Inhib.ActAvg.Fixed": "true",
			}},
		{Sel: ".ExciteLateral", Desc: "lateral excitatory connection",
			Params: params.Params{
				"Path.WtInit.Mean": ".5",
				"Path.WtInit.Var":  "0",
				"Path.WtInit.Sym":  "false",
				"Path.WtScale.Rel": "0.05",
			}},
		{Sel: ".InhibLateral", Desc: "lateral inhibitory connection",
			Params: params.Params{
				"Path.WtInit.Mean": "0",
				"Path.WtInit.Var":  "0",
				"Path.WtInit.Sym":  "false",
				"Path.WtScale.Abs": "0.05",
			}},
	},
}

// Config has config parameters related to running the sim
type Config struct {
	// total number of runs to do when running Train
	NRuns int `default:"1" min:"1"`

	// total number of epochs per run
	NEpochs int `default:"50"`
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	// space-separated words to test the network with
	Words1 string

	// space-separated words to test the network with
	Words2 string

	// excitatory lateral (recurrent) WtScale.Rel value
	ExcitLateralScale float32 `default:"0.05"`

	// inhibitory lateral (recurrent) WtScale.Abs value
	InhibLateralScale float32 `default:"0.05"`

	// do excitatory lateral (recurrent) connections learn?
	ExcitLateralLearn bool `default:"true"`

	// threshold for weight strength for including in WtWords
	WtWordsThr float32 `default:"0.75"`

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
	ss.Defaults()
	ss.Net = leabra.NewNetwork("Semantics")
	ss.Params.Config(ParamSets, "", "", ss.Net)
	ss.Stats.Init()
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
}

func (ss *Sim) Defaults() {
	ss.Words1 = "attention"
	ss.Words2 = "binding"
	ss.ExcitLateralScale = 0.05
	ss.InhibLateralScale = 0.05
	ss.ExcitLateralLearn = true
	ss.WtWordsThr = 0.75
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
	var trn, tst, quiz *SemEnv
	if len(ss.Envs) == 0 {
		trn = &SemEnv{}
		tst = &SemEnv{}
		quiz = &SemEnv{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*SemEnv)
		tst = ss.Envs.ByMode(etime.Test).(*SemEnv)
		quiz = ss.Envs.ByMode(etime.Validate).(*SemEnv)
	}

	// note: names must be standard here!
	trn.Name = etime.Train.String()
	trn.OpenTextsFS(content, "cecn_lg_f5.text")
	trn.OpenWordsFS(content, "cecn_lg_f5.words")

	tst.Name = etime.Test.String()
	tst.Sequential = true
	tst.OpenWordsFS(content, "cecn_lg_f5.words")
	tst.SetParas([]string{ss.Words1, ss.Words2})

	quiz.Name = etime.Validate.String()
	quiz.Sequential = true
	quiz.OpenTextsFS(content, "quiz.text")
	quiz.OpenWordsFS(content, "cecn_lg_f5.words")

	trn.Init(0)
	tst.Init(0)
	quiz.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn, tst, quiz)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	in := net.AddLayer2D("Input", 43, 45, leabra.InputLayer)
	hid := net.AddLayer2D("Hidden", 20, 20, leabra.SuperLayer)

	full := paths.NewFull()
	net.ConnectLayers(in, hid, full, leabra.ForwardPath)

	circ := paths.NewCircle()
	circ.TopoWeights = true
	circ.Radius = 4
	circ.Sigma = .75

	rec := net.ConnectLayers(hid, hid, circ, leabra.LateralPath)
	rec.AddClass("ExciteLateral")

	inh := net.ConnectLayers(hid, hid, full, leabra.InhibPath)
	inh.AddClass("InhibLateral")

	net.Build()
	net.Defaults()
	ss.ApplyParams()
	ss.InitWeights(net)
}

func (ss *Sim) InitWeights(net *leabra.Network) {
	net.InitTopoScales() // needed for gaussian topo Circle wts
	net.InitWeights()
}

func (ss *Sim) ApplyParams() {
	ss.Params.SetAll()

	net := ss.Net
	hid := net.LayerByName("Hidden")
	elat := hid.RecvPaths[1]
	elat.WtScale.Rel = ss.ExcitLateralScale
	elat.Learn.Learn = ss.ExcitLateralLearn
	ilat := hid.RecvPaths[2]
	ilat.WtScale.Abs = ss.InhibLateralScale

	if ss.Loops != nil {
		trn := ss.Loops.Stacks[etime.Train]
		trn.Loops[etime.Run].Counter.Max = ss.Config.NRuns
		trn.Loops[etime.Epoch].Counter.Max = ss.Config.NEpochs
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

	trls := 1533

	ls.AddStack(etime.Train).
		AddTime(etime.Run, ss.Config.NRuns).
		AddTime(etime.Epoch, ss.Config.NEpochs).
		AddTime(etime.Trial, trls).
		AddTime(etime.Cycle, 100)

	ls.AddStack(etime.Test).
		AddTime(etime.Epoch, 1).
		AddTime(etime.Trial, 2).
		AddTime(etime.Cycle, 100)

	ls.AddStack(etime.Validate).
		AddTime(etime.Epoch, 1).
		AddTime(etime.Trial, 40).
		AddTime(etime.Cycle, 100)

	leabra.LooperStdPhases(ls, &ss.Context, ss.Net, 75, 99)                // plus phase timing
	leabra.LooperSimCycleAndLearn(ls, ss.Net, &ss.Context, &ss.ViewUpdate) // std algo code
	ls.Stacks[etime.Train].OnInit.Add("Init", func() { ss.Init() })
	ls.Stacks[etime.Test].OnInit.Add("Init", func() { ss.TestInit() })

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
	ev := ss.Envs.ByMode(ctx.Mode).(*SemEnv)
	ev.Step()

	ss.Stats.SetString("TrialName", ev.String())
	lays := net.LayersByType(leabra.InputLayer, leabra.TargetLayer)
	net.InitExt()
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm)
		pats := ev.State(ly.Name)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}

	ss.SetInputActAvg(net)
}

// Sets Input layer Inhib.ActAvg.Init from ext input
func (ss *Sim) SetInputActAvg(net *leabra.Network) {
	nin := 0
	inp := net.LayerByName("Input")
	for ni := range inp.Neurons {
		nrn := &(inp.Neurons[ni])
		if nrn.Ext > 0 {
			nin++
		}
	}
	if nin > 0 {
		avg := float32(nin) / float32(len(inp.Neurons))
		inp.Inhib.ActAvg.Init = avg
		inp.UpdateActAvgEff()
		net.GScaleFromAvgAct()
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
	ss.InitWeights(ss.Net)
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
}

func (ss *Sim) TestInit() {
	ev := ss.Envs.ByMode(etime.Test).(*SemEnv)
	ev.Init(0)
	err := ev.SetParas([]string{ss.Words1, ss.Words2})
	if err != nil {
		core.ErrorSnackbar(ss.GUI.Body, err)
		return
	}
	ev.Init(0)
	ss.Net.InitActs()
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.Envs.ByMode(etime.Test).Init(0)
	ss.Loops.ResetAndRun(etime.Test)
	ss.Loops.Mode = etime.Train // Important to reset Mode back to Train because this is called from within the Train Run.
}

////////////////////////////////////////////////////////////////////////
// 		Stats

// QuizAll runs through the full set of testing items
func (ss *Sim) QuizAll() {
	ev := ss.Envs.ByMode(etime.Validate)
	ev.Init(0)
	ss.Loops.ResetAndRun(etime.Validate)

	ss.Loops.Mode = etime.Test

	trl := ss.Logs.Log(etime.Validate, etime.Trial)
	epc := ss.Logs.Log(etime.Validate, etime.Epoch)

	nper := 4 // number of paras per quiz question: Q, A, B, C
	nt := trl.Rows
	nq := nt / nper
	pctcor := 0.0
	epc.SetNumRows(nq + 1)
	cors := []float32{0, 0, 0}
	for qi := 0; qi < nq; qi++ {
		ri := nper * qi
		qv := trl.Tensor("Hidden_Act", ri).(*tensor.Float32)
		mxai := 0
		mxcor := float32(0.0)
		for ai := 0; ai < nper-1; ai++ {
			av := trl.Tensor("Hidden_Act", ri+ai+1).(*tensor.Float32)
			cor := metric.Correlation32(qv.Values, av.Values)
			cors[ai] = cor
			if cor > mxcor {
				mxai = ai
				mxcor = cor
			}
			// dt.SetCellTensorFloat1D("Correls", row, ai, cor)
		}
		ans := []string{"A", "B", "C"}[mxai]
		cr := 0.0
		if mxai == 0 { // A
			pctcor += 1
			cr = 1
		}
		epc.SetFloat("Question", qi, float64(qi))
		epc.SetString("Response", qi, ans)
		epc.SetFloat("Correct", qi, cr)
		epc.SetFloat("A", qi, float64(cors[0]))
		epc.SetFloat("B", qi, float64(cors[1]))
		epc.SetFloat("C", qi, float64(cors[2]))
	}
	pctcor /= float64(nq)
	epc.SetFloat("Question", nq, -1)
	epc.SetString("Response", nq, "Total")
	epc.SetFloat("Correct", nq, pctcor)
	ss.GUI.UpdateTableView(etime.Validate, etime.Epoch)
}

func (ss *Sim) WtWords() []string {
	nv := ss.GUI.ViewUpdate.View
	if nv.Data.PathLay != "Hidden" {
		core.MessageDialog(ss.GUI.Body, "WtWords: must select unit in Hidden layer in Network View")
		return nil
	}
	ly := ss.Net.LayerByName("Hidden")
	slay := ss.Net.LayerByName("Input")
	var pvals []float32
	slay.SendPathValues(&pvals, "Wt", ly, nv.Data.PathUnIndex, "")
	ww := make([]string, 0, 1000)
	ev := ss.Envs.ByMode(etime.Train).(*SemEnv)
	for i, wrd := range ev.Words {
		wv := pvals[i]
		if wv > ss.WtWordsThr {
			ww = append(ww, wrd)
		}
	}
	if len(ww) == 0 {
		core.MessageSnackbar(ss.GUI.Body, "No words")
		return ww
	}
	core.MessageDialog(ss.GUI.Body, strings.Join(ww, ", "))
	return ww
}

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetString("TrialName", "")
	ss.Stats.SetString("Words", "")
	ss.Stats.SetFloat("WordsCorrel", 0)
	ss.Stats.SetString("Response", "")
	ss.Stats.SetFloat("Question", 0)
	ss.Stats.SetFloat("Correct", 0)
	ss.Stats.SetFloat("A", 0)
	ss.Stats.SetFloat("B", 0)
	ss.Stats.SetFloat("C", 0)
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
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "TrialName", "Cycle"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
}

func (ss *Sim) WordsShort(wr string) string {
	wf := strings.Fields(wr)
	mx := min(len(wf), 2)
	ws := ""
	for i := 0; i < mx; i++ {
		w := wf[i]
		if len(w) > 4 {
			w = w[:4]
		}
		ws += w
		if i < mx-1 {
			ws += "-"
		}
	}
	return ws
}

func (ss *Sim) WordsLabel() string {
	return ss.WordsShort(ss.Words1) + " v " + ss.WordsShort(ss.Words2)
}

func (ss *Sim) TestStats() {
	trl := ss.Logs.Log(etime.Test, etime.Trial)
	wr1 := trl.Tensor("Hidden_Act", 0).(*tensor.Float32)
	wr2 := trl.Tensor("Hidden_Act", 1).(*tensor.Float32)

	ss.Stats.SetString("Words", ss.WordsLabel())
	ss.Stats.SetFloat32("WordsCorrel", metric.Correlation32(wr1.Values, wr2.Values))
}

//////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddStatStringItem(etime.Test, etime.Epoch, "Words")
	ss.Logs.AddStatFloatNoAggItem(etime.Test, etime.Epoch, "WordsCorrel")

	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "SuperLayer")
	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Validate, etime.Trial, "SuperLayer")

	ss.Logs.AddStatFloatNoAggItem(etime.Validate, etime.Epoch, "Question")
	ss.Logs.AddStatStringItem(etime.Validate, etime.Epoch, "Response")
	ss.Logs.AddStatFloatNoAggItem(etime.Validate, etime.Epoch, "Correct", "A", "B", "C")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Train, etime.Run)
	ss.Logs.NoPlot(etime.Test, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Trial)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")

	ss.Logs.SetMeta(etime.Test, etime.Epoch, "Type", "Bar")
	ss.Logs.SetMeta(etime.Test, etime.Epoch, "XAxis", "Words")
	ss.Logs.SetMeta(etime.Test, etime.Epoch, "XAxisRotation", "-45")
	ss.Logs.SetMeta(etime.Test, etime.Epoch, "WordsCorrel:On", "+")
	ss.Logs.SetMeta(etime.Test, etime.Epoch, "WordsCorrel:FixMax", "+")
	ss.Logs.SetMeta(etime.Test, etime.Epoch, "WordsCorrel:Max", "1")
}

// more "head on" than default which is more "top down"
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
	case mode == etime.Test && time == etime.Epoch:
		ss.TestStats()
	case mode == etime.Validate && time == etime.Epoch:
		return
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

//////////////////////////////////////////////////////////////////////
// 		GUI

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "Semantics"
	ss.GUI.MakeBody(ss, "sem", title, `sem is trained using Hebbian learning on paragraphs from an early draft of the *Computational Explorations..* textbook, allowing it to learn about the overall statistics of when different words co-occur with other words, and thereby learning a surprisingly capable (though clearly imperfect) level of semantic knowlege about the topics covered in the textbook.  This replicates the key results from the Latent Semantic Analysis research by Landauer and Dumais (1997). See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch10/sem/README.md">README.md on GitHub</a>.</p>`, readme)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 300
	nv.Options.Raster.Max = 100
	nv.SetNet(ss.Net)
	nv.Options.PathWidth = 0.003
	ss.ViewUpdate.Config(nv, etime.GammaCycle, etime.GammaCycle)
	ss.GUI.ViewUpdate = &ss.ViewUpdate
	nv.Current()

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1.73, 2.3)
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.AddTableView(&ss.Logs, etime.Validate, etime.Epoch)

	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.Loops)

	// ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Reset Test Plot",
	// 	Icon:    icons.Reset,
	// 	Tooltip: "resets the Test Trial Plot",
	// 	Active:  egui.ActiveAlways,
	// 	Func: func() {
	// 		ss.Logs.ResetLog(etime.Test, etime.Trial)
	// 		ss.GUI.UpdatePlot(etime.Test, etime.Trial)
	// 	},
	// })

	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Quiz all",
		Icon:    icons.RunCircle,
		Tooltip: "runs a quiz about knowledge learned from textbook",
		Active:  egui.ActiveAlways,
		Func: func() {
			go ss.QuizAll()
		},
	})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Open Trained Wts",
		Icon:    icons.Open,
		Tooltip: "Open trained weights",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.Net.OpenWeightsFS(content, "trained_rec05.wts.gz")
		},
	})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Wt Words",
		Icon:    icons.RunCircle,
		Tooltip: "reports the words associated with the strong weights shown in Hidden unit selected in the Network ",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.WtWords()
		},
	})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "README",
		Icon:    icons.FileMarkdown,
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.TheApp.OpenURL("https://github.com/CompCogNeuro/sims/blob/main/ch10/sem/README.md")
		},
	})
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}
