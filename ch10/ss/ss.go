// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// ss explores the way that regularities and exceptions are learned in the
// mapping between spelling (orthography) and sound (phonology), in the context
// of a "direct pathway" mapping between these two forms of word representations.
package main

//go:generate core generate -add-types

import (
	"embed"
	"strings"

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
	"github.com/emer/emergent/v2/relpos"
	"github.com/emer/etensor/tensor"
	"github.com/emer/etensor/tensor/stats/metric"
	"github.com/emer/etensor/tensor/stats/split"
	"github.com/emer/etensor/tensor/stats/stats"
	"github.com/emer/etensor/tensor/table"
	"github.com/emer/etensor/tensor/tensorcore"
	"github.com/emer/leabra/v2/leabra"
)

//go:embed train_pats.tsv probe.tsv besner.tsv glushko.tsv taraban.tsv phon_cons.tsv phon_vowel.tsv trained.wts.gz
var content embed.FS

//go:embed *.png README.md
var readme embed.FS

// EnvType is the type of test environment
type EnvType int32 //enums:enum

const (
	Probe EnvType = iota
	Besner
	Glushko
	Taraban
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
		{Sel: "Path", Desc: "all extra learning factors",
			Params: params.Params{
				"Path.Learn.Norm.On":     "true",
				"Path.Learn.Momentum.On": "true",
				"Path.Learn.WtBal.On":    "true",
				"Path.Learn.Lrate":       "0.04",
			}},
		{Sel: "Layer", Desc: "FB 0.5 apparently required",
			Params: params.Params{
				"Layer.Act.XX1.Gain":       "250", // this is only model where high gain really helps
				"Layer.Act.Dt.GTau":        "3",   // slower is better here
				"Layer.Inhib.Layer.Gi":     "1.8",
				"Layer.Inhib.ActAvg.Init":  "0.1",
				"Layer.Inhib.ActAvg.Fixed": "false", // NOT: using fixed = fully reliable testing
			}},
		{Sel: "#Ortho", Desc: "pool inhib",
			Params: params.Params{
				"Layer.Inhib.Pool.On":     "true",
				"Layer.Inhib.Pool.Gi":     "1.8",
				"Layer.Inhib.ActAvg.Init": "0.022",
			}},
		{Sel: "#OrthoCode", Desc: "pool inhib",
			Params: params.Params{
				"Layer.Inhib.Pool.On":     "true",
				"Layer.Inhib.Pool.Gi":     "1.8",
				"Layer.Inhib.ActAvg.Init": "0.07",
			}},
		{Sel: "#Phon", Desc: "pool-only inhib",
			Params: params.Params{
				"Layer.Inhib.Layer.On":    "false",
				"Layer.Inhib.Pool.On":     "true",
				"Layer.Inhib.Pool.Gi":     "1.8",
				"Layer.Inhib.ActAvg.Init": "0.14",
			}},
		{Sel: ".BackPath", Desc: "weaker top down as usual",
			Params: params.Params{
				"Path.WtScale.Rel": ".1",
			}},
		{Sel: "#HiddenToOrthoCode", Desc: "stronger from hidden",
			Params: params.Params{
				"Path.WtScale.Rel": ".2",
			}},
	},
}

// Config has config parameters related to running the sim
type Config struct {
	// total number of runs to do when running Train
	NRuns int `default:"1" min:"1"`

	// total number of epochs per run
	NEpochs int `default:"400"`

	// total number of trials for training
	NTrials int `default:"1000"`

	// stop run after this number of perfect, zero-error epochs.
	NZero int `default:"-1"`

	// how often to run through all the test patterns, in terms of training epochs.
	// can use 0 or -1 for no testing.
	TestInterval int `default:"-1"`

	// RTThreshold is the threshold for change in max activity level from once cycle to the next
	RTThreshold float32 `default:"0.000001"`
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	// the environment to use for testing -- only takes effect for TestAll.
	TestingEnv EnvType

	// simulation configuration parameters -- set by .toml config file and / or args
	Config Config `new-window:"+"`

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *leabra.Network `new-window:"+" display:"no-inline"`

	// all parameter management
	Params emer.NetParams `display:"add-fields"`

	// training patterns
	Train *table.Table `new-window:"+" display:"no-inline"`

	// probe patterns
	Probe *table.Table `new-window:"+" display:"no-inline"`

	// nonword testing patterns
	Besner *table.Table `new-window:"+" display:"no-inline"`

	// nonword testing patterns
	Glushko *table.Table `new-window:"+" display:"no-inline"`

	// nonword testing patterns
	Taraban *table.Table `new-window:"+" display:"no-inline"`

	// phonology consonant patterns
	PhonCons *table.Table `new-window:"+" display:"no-inline"`

	// phonology vowel patterns
	PhonVowel *table.Table `new-window:"+" display:"no-inline"`

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
	ss.Net = leabra.NewNetwork("SS")
	ss.Params.Config(ParamSets, "", "", ss.Net)
	ss.Stats.Init()
	ss.Train = &table.Table{}
	ss.Probe = &table.Table{}
	ss.Besner = &table.Table{}
	ss.Glushko = &table.Table{}
	ss.Taraban = &table.Table{}
	ss.PhonCons = &table.Table{}
	ss.PhonVowel = &table.Table{}
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
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
	ss.OpenPatAsset(ss.Train, "train_pats.tsv", "Train", "Training Patterns")
	ss.OpenPatAsset(ss.Probe, "probe.tsv", "Probe", "Probe Patterns")
	ss.OpenPatAsset(ss.Besner, "besner.tsv", "Besner", "Nonword Testing Patterns")
	ss.OpenPatAsset(ss.Glushko, "glushko.tsv", "Glushko", "Nonword Testing Patterns")
	ss.OpenPatAsset(ss.Taraban, "taraban.tsv", "Taraban", "Nonword Testing Patterns")
	ss.OpenPatAsset(ss.PhonCons, "phon_cons.tsv", "PhonCons", "Phonology patterns -- consonants")
	ss.OpenPatAsset(ss.PhonVowel, "phon_vowel.tsv", "PhonVowel", "Phonology patterns -- vowels")
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var trn *env.FreqTable
	var tst *env.FixedTable
	if len(ss.Envs) == 0 {
		trn = &env.FreqTable{}
		tst = &env.FixedTable{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*env.FreqTable)
		tst = ss.Envs.ByMode(etime.Test).(*env.FixedTable)
	}

	trn.Name = etime.Train.String()
	trn.Table = table.NewIndexView(ss.Train)
	trn.NSamples = 1
	trn.RandSamp = true

	tst.Name = etime.Test.String()
	tst.GroupCol = "Type"
	tst.Config(table.NewIndexView(ss.Probe))
	tst.Sequential = true

	trn.Init(0)
	tst.Init(0)

	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigTestEnv() {
	tst := ss.Envs.ByMode(etime.Test).(*env.FixedTable)
	switch ss.TestingEnv {
	case Probe:
		tst.Table = table.NewIndexView(ss.Probe)
	case Besner:
		tst.Table = table.NewIndexView(ss.Besner)
	case Glushko:
		tst.Table = table.NewIndexView(ss.Glushko)
	case Taraban:
		tst.Table = table.NewIndexView(ss.Taraban)
	}
	tst.Init(0)
	if ss.Loops != nil {
		tt := ss.Loops.Stacks[etime.Test]
		tt.Loops[etime.Trial].Counter.Max = tst.Table.Table.Rows
	}
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	ort := net.AddLayer4D("Ortho", 1, 7, 9, 3, leabra.InputLayer)
	ocd := net.AddLayer4D("OrthoCode", 1, 5, 14, 6, leabra.SuperLayer)
	hid := net.AddLayer2D("Hidden", 20, 30, leabra.SuperLayer)
	phn := net.AddLayer4D("Phon", 1, 7, 10, 2, leabra.TargetLayer)

	full := paths.NewFull()
	ocdPath := paths.NewPoolTile()
	ocdPath.Size.Set(3, 1)
	ocdPath.Skip.Set(1, 0)
	ocdPath.Start.Set(0, 0)

	net.ConnectLayers(ort, ocd, ocdPath, leabra.ForwardPath)
	net.BidirConnectLayers(ocd, hid, full)
	net.BidirConnectLayers(hid, phn, full)

	ocd.PlaceAbove(ort)
	ocd.Pos.XAlign = relpos.Middle
	hid.PlaceAbove(ocd)
	hid.Pos.XAlign = relpos.Middle
	phn.PlaceAbove(hid)
	phn.Pos.XAlign = relpos.Middle

	net.Build()
	net.Defaults()
	ss.ApplyParams()
	net.InitWeights()
}

func (ss *Sim) ApplyParams() {
	ss.Params.SetAll() // first hard-coded defaults
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

func (ss *Sim) TestInit() {
	ss.ConfigTestEnv()
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	ls := looper.NewStacks()

	trls := ss.Config.NTrials

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
		if (ss.Config.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.Config.TestInterval == 0) {
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
	net.InitExt()
	evi := ss.Envs.ByMode(ctx.Mode)
	evi.Step()
	if ctx.Mode == etime.Train {
		ss.Stats.SetString("TrialName", evi.(*env.FreqTable).TrialName.Cur)
	} else {
		ss.Stats.SetString("TrialName", evi.(*env.FixedTable).TrialName.Cur)
		ss.Stats.SetString("Type", evi.(*env.FixedTable).GroupName.Cur)
	}
	ss.Stats.SetFloat("RT", 100)
	ss.Stats.SetFloat("MaxAct", 0)

	lays := net.LayersByType(leabra.InputLayer, leabra.TargetLayer)
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
	ss.Stats.SetFloat("SSE", 0.0)
	ss.Stats.SetFloat("AvgSSE", 0.0)
	ss.Stats.SetFloat("RT", 0.0)
	ss.Stats.SetString("TrialName", "")
	ss.Stats.SetString("Type", "")
	ss.Stats.SetString("Phon", "")
	ss.Stats.SetFloat("PhonSSE", 0.0)
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
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "Type", "TrialName", "Phon", "Cycle", "SSE", "TrlErr"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	// ctx := &ss.Context
	out := ss.Net.LayerByName("Phon")
	sse, avgsse := out.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
	ss.Stats.SetFloat("SSE", sse)
	ss.Stats.SetFloat("AvgSSE", avgsse)
	trlnm := ss.Stats.String("TrialName")

	phon, psse := ss.Pronounce(ss.Net)
	ss.Stats.SetString("Phon", phon)
	ss.Stats.SetFloat("PhonSSE", psse)
	spnm := strings.Split(trlnm, "_")
	if phon == spnm[2] {
		ss.Stats.SetFloat("TrlErr", 0)
	} else {
		ss.Stats.SetFloat("TrlErr", 1)
	}
}

// Pronounce returns the pronunciation of the phonological output layer
func (ss *Sim) Pronounce(net emer.Network) (string, float64) {
	tsr := ss.Stats.SetLayerTensor(net, "Phon", "ActM", 0)
	ccol := errors.Log1(ss.PhonCons.ColumnByName("Phon")).(*tensor.Float32)
	vcol := errors.Log1(ss.PhonVowel.ColumnByName("Phon")).(*tensor.Float32)
	sseTol := float32(10.0)
	totSSE := float32(0.0)
	ph := ""
	for pi := 0; pi < 7; pi++ {
		cvt := tsr.SubSpace([]int{0, pi}).(*tensor.Float32)
		nm := ""
		sse := float32(0.0)
		row := 0
		if pi == 3 { // vowel
			row, sse = metric.ClosestRow32(cvt, vcol, metric.SumSquaresBinTol32)
			nm = ss.PhonVowel.StringValue("Name", row)
		} else {
			row, sse = metric.ClosestRow32(cvt, ccol, metric.SumSquaresBinTol32)
			nm = ss.PhonCons.StringValue("Name", row)
		}
		if sse > sseTol {
			nm = "X"
		}
		ph += nm
		totSSE += sse
	}
	return ph, float64(totSSE)
}

func (ss *Sim) TestEpochStats() {
	dt := ss.Logs.Table(etime.Test, etime.Trial)
	if dt == nil {
		return
	}
	tix := table.NewIndexView(dt)
	spl := split.GroupBy(tix, "TrialName")
	split.AggColumn(spl, "Err", stats.Min)
	split.AggColumn(spl, "RT", stats.Mean)
	minerr := spl.AggsToTableCopy(table.ColumnNameOnly)
	ss.Logs.MiscTables["MinErr"] = minerr

	allerr := table.NewIndexView(minerr)
	allerr.Filter(func(et *table.Table, row int) bool {
		return et.Float("Err", row) > 0
	})
	at := allerr.NewTable()
	ss.Logs.MiscTables["Errors"] = at
	tv := ss.GUI.TableViews[etime.ScopeKey("Errors")]
	tv.SetTable(at)
	tv.AsyncUpdateTable()

	rtspl := split.GroupBy(tix, "Type")
	split.AggColumn(rtspl, "RT", stats.Mean)
	split.AggColumn(rtspl, "RT", stats.Sem)
	rt := rtspl.AggsToTable(table.AddAggName)
	ss.Logs.MiscTables["RT"] = rt

	plt := ss.GUI.Plots[etime.ScopeKey("RT")]
	rt.SetMetaData("XAxis", "Type")
	rt.SetMetaData("Type", "Bar")
	rt.SetMetaData("RT:Mean:On", "+")
	rt.SetMetaData("RT:Mean:FixMin", "true")
	rt.SetMetaData("RT:Mean:Min", "0")
	rt.SetMetaData("RT:Mean:ErrColumn", "RT:Sem")

	plt.SetTable(rt)
	plt.GoUpdatePlot()
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "Type", "TrialName", "Phon")

	ss.Logs.AddStatAggItem("SSE", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("AvgSSE", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("PhonSSE", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddErrStatAggItems("TrlErr", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("RT", etime.Run, etime.Epoch, etime.Trial)

	leabra.LogInputLayer(&ss.Logs, ss.Net, etime.Train)

	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "TargetLayer")

	ss.Logs.PlotItems("RT", "PctErr")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Train, etime.Run)
	ss.Logs.NoPlot(etime.Test, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Epoch)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Test, etime.Trial, "Err:On", "+")
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	if mode.String() != "Analyze" {
		ss.Context.Mode = mode // Also set specifically in a Loop callback.
	}
	dt := ss.Logs.Table(mode, time)
	row := dt.Rows

	switch {
	case time == etime.Cycle:
		rtc := ss.Stats.Float("RT")
		if rtc == 100 {
			pmax := ss.Stats.Float32("MaxAct")
			phn := ss.Net.LayerByName("Phon")
			mxact := phn.Pools[0].Inhib.Act.Max
			da := math32.Abs(mxact - pmax)
			ss.Stats.SetFloat32("MaxAct", mxact)
			if mxact > 0.5 && da < ss.Config.RTThreshold {
				ss.Stats.SetFloat("RT", float64(ss.Context.Cycle))
			}
		}
		return
	case time == etime.Trial:
		ss.TrialStats()
		ss.StatCounters()
		ss.Logs.LogRow(mode, time, row)
		return // don't do reg below
	case time == etime.Epoch && mode == etime.Test:
		ss.TestEpochStats()
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		GUI

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "Spelling to Sound"
	ss.GUI.MakeBody(ss, "ss", title, `explores the way that regularities and exceptions are learned in the mapping between spelling (orthography) and sound (phonology), in the context of a "direct pathway" mapping between these two forms of word representations. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch10/ss/README.md">README.md on GitHub</a>.</p>`, readme)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 300
	nv.Options.Raster.Max = 100
	nv.Options.LayerNameSize = 0.03
	nv.SetNet(ss.Net)
	ss.ViewUpdate.Config(nv, etime.GammaCycle, etime.GammaCycle)
	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1.05, 2.75)
	nv.SceneXYZ().Camera.LookAt(math32.Vector3{0, 0, 0}, math32.Vector3{0, 1, 0})

	ss.GUI.ViewUpdate = &ss.ViewUpdate

	ss.GUI.AddPlots(title, &ss.Logs)

	gui := &ss.GUI
	if gui.TableViews == nil {
		gui.TableViews = make(map[etime.ScopeKey]*tensorcore.Table)
	}
	stnm := "Errors"
	dt := ss.Logs.MiscTable(stnm)
	key := etime.ScopeKey(stnm)
	tt, _ := gui.Tabs.NewTab(stnm)
	tv := tensorcore.NewTable(tt)
	gui.TableViews[key] = tv
	tv.SetReadOnly(true)
	tv.SetTable(dt)

	stnm = "RT"
	dt = ss.Logs.MiscTable(stnm)
	plt := ss.GUI.NewPlotTab(etime.ScopeKey(stnm), stnm+" Plot")
	plt.Options.Title = "Reaction Time by Type"
	plt.Options.XAxis = "Type"
	plt.SetTable(dt)

	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.Loops)

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Open Trained Wts", Icon: icons.Open,
		Tooltip: "Opened weights from the first phase of training, which excludes novel objects",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Net.OpenWeightsFS(content, "trained.wts.gz")
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
			core.TheApp.OpenURL("https://github.com/CompCogNeuro/sims/blob/main/ch10/ss/README.md")
		},
	})
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}
