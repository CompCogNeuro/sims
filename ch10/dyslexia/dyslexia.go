// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// dyslexia simulates normal and disordered (dyslexic) reading performance in terms
// of a distributed representation of word-level knowledge across Orthography, Semantics,
// and Phonology. It is based on a model by Plaut and Shallice (1993).
// Note that this form of dyslexia is *acquired* (via brain lesions such as stroke)
// and not the more prevalent developmental variety.
package main

//go:generate core generate -add-types

import (
	"embed"
	"math/rand"
	"reflect"
	"strings"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32/minmax"
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
	"github.com/emer/etensor/tensor/stats/clust"
	"github.com/emer/etensor/tensor/stats/metric"
	"github.com/emer/etensor/tensor/stats/split"
	"github.com/emer/etensor/tensor/stats/stats"
	"github.com/emer/etensor/tensor/table"
	"github.com/emer/leabra/v2/leabra"
)

//go:embed train_pats.tsv semantics.tsv close_orthos.tsv close_sems.tsv trained.wts
var content embed.FS

//go:embed *.png README.md
var readme embed.FS

// LesionTypes is the type of lesion
type LesionTypes int32 //enums:enum

const (
	NoLesion LesionTypes = iota
	SemanticsFull
	DirectFull
	OShidden // partial
	SPhidden
	OPhidden
	OShidDirectFull
	SPhidDirectFull
	OPhidSemanticsFull
	AllPartial // do all above partial with partials .1..1
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
		{Sel: "Path", Desc: "no extra learning factors",
			Params: params.Params{
				"Path.Learn.Norm.On":     "true",
				"Path.Learn.Momentum.On": "true",
				"Path.Learn.WtBal.On":    "true",
				"Path.Learn.Lrate":       "0.04",
			}},
		{Sel: "Layer", Desc: "FB 0.5 apparently required",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi":     "2.1",
				"Layer.Inhib.Layer.FB":     "0.5",
				"Layer.Inhib.ActAvg.Init":  "0.2",
				"Layer.Inhib.ActAvg.Fixed": "true", // using fixed = fully reliable testing
			}},
		{Sel: "#Orthography", Desc: "higher inhib",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi":    "2.4",
				"Layer.Inhib.Layer.FB":    "0.5",
				"Layer.Inhib.ActAvg.Init": "0.08",
			}},
		{Sel: "#Semantics", Desc: "higher inhib",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi":    "2.2",
				"Layer.Inhib.Layer.FB":    "0.5",
				"Layer.Inhib.ActAvg.Init": "0.2",
			}},
		{Sel: "#Phonology", Desc: "pool-only inhib",
			Params: params.Params{
				"Layer.Inhib.Layer.On":    "false",
				"Layer.Inhib.Pool.On":     "true",
				"Layer.Inhib.Pool.Gi":     "2.8",
				"Layer.Inhib.Pool.FB":     "0.5",
				"Layer.Inhib.ActAvg.Init": "0.07",
			}},
		{Sel: ".BackPath", Desc: "there is no back / forward direction here..",
			Params: params.Params{
				"Path.WtScale.Rel": "1",
			}},
		{Sel: ".LateralPath", Desc: "self cons are weaker",
			Params: params.Params{
				"Path.WtScale.Rel": "0.3",
			}},
	},
}

// Config has config parameters related to running the sim
type Config struct {
	// total number of runs to do when running Train
	NRuns int `default:"1" min:"1"`

	// total number of epochs per run
	NEpochs int `default:"250"`

	// stop run after this number of perfect, zero-error epochs.
	NZero int `default:"-1"`

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
	// type of lesion -- use Lesion button to lesion
	Lesion LesionTypes `edit:"-"`

	// proportion of neurons lesioned -- use Lesion button to lesion
	LesionProp float32 `edit:"-"`

	// Config contains misc configuration parameters for running the sim
	Config Config `new-window:"+" display:"no-inline"`

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *leabra.Network `new-window:"+" display:"no-inline"`

	// network parameter management
	Params emer.NetParams `display:"add-fields"`

	// training patterns
	Train *table.Table `new-window:"+" display:"no-inline"`

	// properties of semnatic features
	Semantics *table.Table `new-window:"+" display:"no-inline"`

	// close orthography outputs
	CloseOrthos *table.Table `new-window:"+" display:"no-inline"`

	// close semantic outputs
	CloseSems *table.Table `new-window:"+" display:"no-inline"`

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
	ss.Net = leabra.NewNetwork("Dyslexia")
	ss.Params.Config(ParamSets, "", "", ss.Net)
	ss.Stats.Init()
	ss.Train = &table.Table{}
	ss.Semantics = &table.Table{}
	ss.CloseOrthos = &table.Table{}
	ss.CloseSems = &table.Table{}
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
}

func (ss *Sim) Defaults() {
	ss.Lesion = NoLesion
	ss.LesionProp = 0
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
	ss.OpenPatAsset(ss.Train, "train_pats.tsv", "Train", "Dyslexia Training Patterns")
	ss.OpenPatAsset(ss.Semantics, "semantics.tsv", "Semantics", "Dyslexia Semantics Patterns")
	ss.OpenPatAsset(ss.CloseOrthos, "close_orthos.tsv", "CloseOrthos", "Close Orthography Patterns")
	ss.OpenPatAsset(ss.CloseSems, "close_sems.tsv", "CloseSems", "Close Semantics Patterns")
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
	trn.Config(table.NewIndexView(ss.Train))
	trn.Validate()

	tst.Name = etime.Test.String()
	tst.Config(table.NewIndexView(ss.Train))
	tst.Sequential = true
	tst.Validate()

	trn.Init(0)
	tst.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	ort := net.AddLayer2D("Orthography", 6, 8, leabra.InputLayer)
	oph := net.AddLayer2D("OPhidden", 7, 7, leabra.SuperLayer)
	phn := net.AddLayer4D("Phonology", 1, 7, 7, 2, leabra.TargetLayer)
	osh := net.AddLayer2D("OShidden", 10, 7, leabra.SuperLayer)
	sph := net.AddLayer2D("SPhidden", 10, 7, leabra.SuperLayer)
	sem := net.AddLayer2D("Semantics", 10, 12, leabra.TargetLayer)

	full := paths.NewFull()
	net.BidirConnectLayers(ort, osh, full)
	net.BidirConnectLayers(osh, sem, full)
	net.BidirConnectLayers(sem, sph, full)
	net.BidirConnectLayers(sph, phn, full)
	net.BidirConnectLayers(ort, oph, full)
	net.BidirConnectLayers(oph, phn, full)

	// lateral cons
	net.LateralConnectLayer(ort, full)
	net.LateralConnectLayer(sem, full)
	net.LateralConnectLayer(phn, full)

	oph.PlaceRightOf(ort, 2)
	phn.PlaceRightOf(oph, 2)
	osh.PlaceAbove(ort)
	osh.Pos.XOffset = 4
	sph.PlaceAbove(phn)
	sph.Pos.XOffset = 2
	sem.PlaceAbove(osh)
	sem.Pos.XOffset = 4

	net.Build()
	net.Defaults()
	ss.ApplyParams()
	net.InitWeights()
}

func (ss *Sim) ApplyParams() {
	ss.Params.SetAll()
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

func (ss *Sim) TestInit() {
	ss.Envs.ByMode(etime.Test).Init(0)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	ls := looper.NewStacks()

	trls := ss.Train.Rows

	ls.AddStack(etime.Train).
		AddTime(etime.Run, ss.Config.NRuns).
		AddTime(etime.Epoch, ss.Config.NEpochs).
		AddTime(etime.Trial, trls).
		AddTime(etime.Cycle, 100)

	ls.AddStack(etime.Test).
		AddTime(etime.Epoch, 1).
		AddTime(etime.Trial, trls).
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

	// Train stop early condition
	ls.Loop(etime.Train, etime.Epoch).IsDone.AddBool("NZeroStop", func() bool {
		// This is calculated in TrialStats
		stopNz := ss.Config.NZero
		if stopNz <= 0 {
			stopNz = 2
		}
		curNZero := ss.Stats.Int("NZero")
		stop := curNZero >= stopNz
		return stop
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

	if ctx.Mode == etime.Train {
		ss.SetRndInputLayer()
	} else {
		ss.SetInputLayer(3)
	}

	lays := net.LayersByType(leabra.InputLayer, leabra.TargetLayer)
	net.InitExt()
	ss.Stats.SetString("TrialName", ev.TrialName.Cur)
	ss.Stats.SetString("Word", strings.Split(ev.TrialName.Cur, "_")[0])
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm)
		pats := ev.State(ly.Name)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// SetInputLayer determines which layer is the input -- others are targets
// 0 = Ortho, 1 = Sem, 2 = Phon, 3 = Ortho + compare for others
func (ss *Sim) SetInputLayer(layno int) {
	lays := []string{"Orthography", "Semantics", "Phonology"}
	test := false
	if layno > 2 {
		layno = 0
		test = true
	}
	for i, lnm := range lays {
		ly := ss.Net.LayerByName(lnm)
		if i == layno {
			ly.Type = leabra.InputLayer
		} else {
			if test {
				ly.Type = leabra.CompareLayer
			} else {
				ly.Type = leabra.TargetLayer
			}
		}
	}
}

// SetRndInputLayer sets one of 3 visible layers as input at random
func (ss *Sim) SetRndInputLayer() {
	ss.SetInputLayer(rand.Intn(3))
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

// LesionNet does lesion of network with given proportion of neurons damaged
// 0 < proportion < 1.
func (ss *Sim) LesionNet(les LesionTypes, proportion float32) { //types:add
	ss.Lesion = les
	ss.LesionProp = proportion
	net := ss.Net
	lesStep := float32(0.1)
	if les == AllPartial {
		for ls := OShidden; ls < AllPartial; ls++ {
			for prp := lesStep; prp < 1; prp += lesStep {
				ss.UnLesionNet(net)
				ss.LesionNetImpl(net, ls, prp)
				ss.TestAll()
			}
		}
	} else {
		ss.UnLesionNet(net)
		ss.LesionNetImpl(net, les, proportion)
	}
}

func (ss *Sim) UnLesionNet(net *leabra.Network) {
	net.LayersSetOff(false)
	net.UnLesionNeurons()
	net.InitActs()
}

func (ss *Sim) LesionNetImpl(net *leabra.Network, les LesionTypes, prop float32) {
	ss.Lesion = les
	ss.LesionProp = prop
	switch les {
	case NoLesion:
	case SemanticsFull:
		net.LayerByName("OShidden").Off = true
		net.LayerByName("Semantics").Off = true
		net.LayerByName("SPhidden").Off = true
	case DirectFull:
		net.LayerByName("OPhidden").Off = true
	case OShidden:
		net.LayerByName("OShidden").LesionNeurons(prop)
	case SPhidden:
		net.LayerByName("SPhidden").LesionNeurons(prop)
	case OPhidden:
		net.LayerByName("OPhidden").LesionNeurons(prop)
	case OShidDirectFull:
		net.LayerByName("OPhidden").Off = true
		net.LayerByName("OShidden").LesionNeurons(prop)
	case SPhidDirectFull:
		net.LayerByName("OPhidden").Off = true
		net.LayerByName("SPhidden").LesionNeurons(prop)
	case OPhidSemanticsFull:
		net.LayerByName("OShidden").Off = true
		net.LayerByName("Semantics").Off = true
		net.LayerByName("SPhidden").Off = true
		net.LayerByName("OPhidden").LesionNeurons(prop)
	}
}

////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetFloat("SSE", 0.0)
	ss.Stats.SetFloat("PhonSSE", 0.0)
	ss.Stats.SetFloat("ConAbs", 0.0)
	ss.Stats.SetFloat("Vis", 0.0)
	ss.Stats.SetFloat("Sem", 0.0)
	ss.Stats.SetFloat("VisSem", 0.0)
	ss.Stats.SetFloat("Blend", 0.0)
	ss.Stats.SetFloat("Other", 0.0)
	ss.Stats.SetString("TrialName", "")
	ss.Stats.SetString("Phon", "")
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
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "TrialName", "Cycle", "SSE", "TrlErr"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	var sse, avgsse float64
	ntrg := 0
	for _, ly := range ss.Net.Layers {
		if ly.Off || (ly.Type != leabra.TargetLayer && ly.Type != leabra.CompareLayer) {
			continue
		}
		lsse, lavgsse := ly.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
		sse += lsse
		avgsse += lavgsse
		ntrg++
	}
	if ntrg > 0 {
		sse /= float64(ntrg)
		avgsse /= float64(ntrg)
	}
	ss.Stats.SetFloat("SSE", sse)
	ss.Stats.SetFloat("AvgSSE", avgsse)
	if sse > 0 {
		ss.Stats.SetFloat("TrlErr", 1)
	} else {
		ss.Stats.SetFloat("TrlErr", 0)
	}

	trlnm := ss.Stats.String("TrialName")
	pidx := errors.Log1(ss.Train.RowsByString("Name", trlnm, table.Equals, table.UseCase))[0]
	if pidx < 20 {
		ss.Stats.SetFloat("ConAbs", 0)
	} else {
		ss.Stats.SetFloat("ConAbs", 1)
	}
	if ss.Context.Mode == etime.Test {
		ss.DyslexStats(ss.Net)
	}
}

// DyslexStats computes dyslexia pronunciation, semantics stats
func (ss *Sim) DyslexStats(net *leabra.Network) {
	ss.Stats.SetString("Lesion", ss.Lesion.String())
	ss.Stats.SetFloat32("LesionProp", ss.LesionProp)
	_, sse, cnm := ss.ClosestPat(net, "Phonology", "ActM", ss.Train, "Phonology", "Name")
	ss.Stats.SetString("Phon", cnm)
	ss.Stats.SetFloat32("PhonSSE", sse)
	ss.Stats.SetFloat("Vis", 0)
	ss.Stats.SetFloat("Sem", 0)
	ss.Stats.SetFloat("VisSem", 0)
	ss.Stats.SetFloat("Blend", 0)
	ss.Stats.SetFloat("Other", 0)
	trlnm := ss.Stats.String("TrialName")
	if sse > 3 { // 3 is the threshold for blend errors
		ss.Stats.SetFloat("Blend", 1)
	} else {
		if trlnm != cnm {
			vis := ss.ClosePat(trlnm, cnm, ss.CloseOrthos)
			sem := ss.ClosePat(trlnm, cnm, ss.CloseSems)
			ss.Stats.SetFloat("Vis", vis)
			ss.Stats.SetFloat("Sem", sem)
			if vis > 0 && sem > 0 {
				ss.Stats.SetFloat("VisSem", 1)
			}
			if vis == 0 && sem == 0 {
				ss.Stats.SetFloat("Other", 1)
			}
		}
	}
}

func (ss *Sim) ClosestPat(net *leabra.Network, layNm, unitVar string, pats *table.Table, colnm, namecol string) (int, float32, string) {
	tsr := ss.Stats.SetLayerTensor(net, layNm, unitVar, 0)
	col := errors.Log1(pats.ColumnByName(colnm))
	row, cor := metric.ClosestRow32(tsr, col.(*tensor.Float32), metric.SumSquaresBinTol32)
	nm := ""
	if namecol != "" {
		nm = pats.StringValue(namecol, row)
	}
	return row, cor, nm
}

// ClosePat looks up phon pattern name in given table of close names -- if found returns 1, else 0
func (ss *Sim) ClosePat(trlnm, phon string, clsdt *table.Table) float64 {
	rws, _ := clsdt.RowsByString(trlnm, phon, table.Equals, table.UseCase)
	return float64(len(rws))
}

// ClusterPlot generates a cluster plot of the
func (ss *Sim) ClusterPlot() {
	// get rid of _phon in names
	tpcp := ss.Train.Clone()
	for r := 0; r < tpcp.Rows; r++ {
		n := tpcp.StringValue("Name", r)
		n = strings.Split(n, "_")[0]
		tpcp.SetString("Name", r, n)
	}
	estats.ClusterPlot(ss.GUI.PlotByName("SemCluster"), table.NewIndexView(tpcp), "Semantics", "Name", clust.ContrastDist)
}

//////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName", "Word")
	ss.Logs.AddStatStringItem(etime.Test, etime.Trial, "Phon")
	ss.Logs.AddStatStringItem(etime.Test, etime.Epoch, "Lesion")
	ss.Logs.AddStatFloatNoAggItem(etime.Test, etime.Epoch, "LesionProp")

	ss.Logs.AddStatAggItem("SSE", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("AvgSSE", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddErrStatAggItems("TrlErr", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	ss.AddTestStatAggItem("ConAbs")
	ss.AddTestStatAggItem("PhonSSE")
	ss.AddTestStatAggItem("Vis")
	ss.AddTestStatAggItem("Sem")
	ss.AddTestStatAggItem("VisSem")
	ss.AddTestStatAggItem("Blend")
	ss.AddTestStatAggItem("Other")

	ss.AddTestEpochAggs()

	// ss.Logs.AddLayerTensorItems(ss.Net, "ActM", etime.Test, etime.Trial, "InputLayer", "SuperLayer", "TargetLayer")
	// ss.Logs.AddLayerTensorItems(ss.Net, "Targ", etime.Test, etime.Trial, "TargetLayer")

	ss.Logs.PlotItems("PctErr", "FirstZero", "LastZero")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Train, etime.Run)
	ss.Logs.NoPlot(etime.Test, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	ss.Logs.SetMeta(etime.Test, etime.Trial, "Type", "Bar")
	ss.Logs.SetMeta(etime.Test, etime.Trial, "XAxis", "Word")
	ss.Logs.SetMeta(etime.Test, etime.Trial, "XAxisRotation", "-45")
	ss.Logs.SetMeta(etime.Test, etime.Trial, "Vis:On", "+")
	ss.Logs.SetMeta(etime.Test, etime.Trial, "Sem:On", "+")
	ss.Logs.SetMeta(etime.Test, etime.Trial, "VisSem:On", "+")
	ss.Logs.SetMeta(etime.Test, etime.Trial, "Blend:On", "+")
	ss.Logs.SetMeta(etime.Test, etime.Trial, "Other:On", "+")

	ss.Logs.SetMeta(etime.Test, etime.Epoch, "Type", "Bar")
	ss.Logs.SetMeta(etime.Test, etime.Epoch, "XAxis", "Lesion")
	ss.Logs.SetMeta(etime.Test, etime.Epoch, "XAxisRotation", "-45")
	ss.Logs.SetMeta(etime.Test, etime.Epoch, "PctErr:On", "-")
	cols := []string{"Vis", "Sem", "VisSem", "Blend", "Other"}
	for _, cl := range cols {
		ss.Logs.SetMeta(etime.Test, etime.Epoch, "Con"+cl+":On", "+")
		ss.Logs.SetMeta(etime.Test, etime.Epoch, "Abs"+cl+":On", "+")
	}
}

func (ss *Sim) AddTestStatAggItem(statName string) {
	it := ss.Logs.AddItem(&elog.Item{
		Name:   statName,
		Type:   reflect.Float64,
		FixMax: true,
		Range:  minmax.F32{Max: 1},
		Write: elog.WriteMap{
			etime.Scope(etime.Test, etime.Trial): func(ctx *elog.Context) {
				ctx.SetFloat64(ss.Stats.Float(statName))
			}}})
	ss.Logs.AddStdAggs(it, etime.Test, etime.Epoch, etime.Trial)
}

func (ss *Sim) TestEpochStats() {
	dt := ss.Logs.Table(etime.Test, etime.Trial)
	if dt == nil {
		return
	}
	ix := table.NewIndexView(dt)
	spl := split.GroupBy(ix, "ConAbs")
	cols := []string{"Vis", "Sem", "VisSem", "Blend", "Other"}
	for _, cl := range cols {
		split.AggColumn(spl, cl, stats.Sum)
	}
	st := spl.AggsToTable(table.ColumnNameOnly)
	ss.Logs.MiscTables["EpochStats"] = st
}

func (ss *Sim) AddTestEpochAggs() {
	cols := []string{"Vis", "Sem", "VisSem", "Blend", "Other"}
	for _, cl := range cols {
		ss.Logs.AddItem(&elog.Item{
			Name:   "Con" + cl,
			Type:   reflect.Float64,
			FixMax: false,
			Range:  minmax.F32{Max: 10},
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Epoch): func(ctx *elog.Context) {
					st := ss.Logs.MiscTable("EpochStats")
					ctx.SetFloat64(st.Float(cl, 0))
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   "Abs" + cl,
			Type:   reflect.Float64,
			FixMax: false,
			Range:  minmax.F32{Max: 10},
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Epoch): func(ctx *elog.Context) {
					st := ss.Logs.MiscTable("EpochStats")
					ctx.SetFloat64(st.Float(cl, 1))
				}}})
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
		return
	case time == etime.Trial:
		ss.TrialStats()
		ss.StatCounters()
	case time == etime.Epoch && mode == etime.Test:
		ss.TestEpochStats()
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
	title := "Dyslexia"
	ss.GUI.MakeBody(ss, "dyslexia", title, `Simulates normal and disordered (dyslexic) reading performance in terms of a distributed representation of word-level knowledge across Orthography, Semantics, and Phonology. It is based on a model by Plaut and Shallice (1993). Note that this form of dyslexia is *aquired* (via brain lesions such as stroke) and not the more prevalent developmental variety.  See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch10/dyslexia/README.md">README.md on GitHub</a>.</p>`, readme)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 300
	nv.Options.Raster.Max = 100
	nv.SetNet(ss.Net)
	nv.Options.PathWidth = 0.003
	ss.ViewUpdate.Config(nv, etime.GammaCycle, etime.Cycle)
	ss.GUI.ViewUpdate = &ss.ViewUpdate
	nv.Current()

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.AddTableView(&ss.Logs, etime.Test, etime.Trial)

	ss.GUI.AddMiscPlotTab("SemCluster")

	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.Loops)

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Lesion",
		Icon:    icons.Delete,
		Tooltip: "lesion the network",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.CallFunc(ss.GUI.Body, ss.LesionNet)
		},
	})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Open Trained Wts",
		Icon:    icons.Open,
		Tooltip: "Open trained weights",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.Net.OpenWeightsFS(content, "trained.wts")
			ss.GUI.ViewUpdate.View.Current()
		},
	})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Cluster Plot",
		Icon:    icons.BarChart,
		Tooltip: "Generates a cluster plot of the semantic representations",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.ClusterPlot()
		},
	})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Reset Epoch Plot",
		Icon:    icons.Reset,
		Tooltip: "resets the Test Epoch Plot",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.Logs.ResetLog(etime.Test, etime.Epoch)
			ss.GUI.UpdatePlot(etime.Test, etime.Epoch)
		},
	})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "README",
		Icon:    icons.FileMarkdown,
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.TheApp.OpenURL("https://github.com/CompCogNeuro/sims/blob/main/ch10/dyslexia/README.md")
		},
	})
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}
