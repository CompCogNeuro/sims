// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// hip runs a hippocampus model on the AB-AC paired associate learning task.
package main

//go:generate core generate -add-types

import (
	"fmt"
	"log"
	"math/rand"
	"reflect"

	"cogentcore.org/core/base/randx"
	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tensor/table"
	"cogentcore.org/core/tree"
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
	"github.com/emer/emergent/v2/patgen"
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/leabra/v2/leabra"
)

func main() {
	sim := &Sim{}
	sim.New()
	sim.ConfigAll()
	sim.RunGUI()
}

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	"Base": {
		{Sel: "Prjn", Desc: "keeping default params for generic prjns",
			Params: params.Params{
				"Prjn.Learn.Momentum.On": "true",
				"Prjn.Learn.Norm.On":     "true",
				"Prjn.Learn.WtBal.On":    "false",
			}},
		{Sel: ".EcCa1Prjn", Desc: "encoder projections -- no norm, moment",
			Params: params.Params{
				"Prjn.Learn.Lrate":        "0.04",
				"Prjn.Learn.Momentum.On":  "false",
				"Prjn.Learn.Norm.On":      "false",
				"Prjn.Learn.WtBal.On":     "true",
				"Prjn.Learn.XCal.SetLLrn": "false", // using bcm now, better
			}},
		{Sel: ".HippoCHL", Desc: "hippo CHL projections -- no norm, moment, but YES wtbal = sig better",
			Params: params.Params{
				"Prjn.CHL.Hebb":          "0.05",
				"Prjn.Learn.Lrate":       "0.2",
				"Prjn.Learn.Momentum.On": "false",
				"Prjn.Learn.Norm.On":     "false",
				"Prjn.Learn.WtBal.On":    "true",
			}},
		{Sel: ".PPath", Desc: "perforant path, new Dg error-driven EcCa1Prjn prjns",
			Params: params.Params{
				"Prjn.Learn.Momentum.On": "false",
				"Prjn.Learn.Norm.On":     "false",
				"Prjn.Learn.WtBal.On":    "true",
				"Prjn.Learn.Lrate":       "0.15", // err driven: .15 > .2 > .25 > .1
				// moss=4, delta=4, lr=0.2, test = 3 are best
			}},
		{Sel: "#CA1ToECout", Desc: "extra strong from CA1 to ECout",
			Params: params.Params{
				"Prjn.WtScale.Abs": "4.0",
			}},
		{Sel: "#InputToECin", Desc: "one-to-one input to EC",
			Params: params.Params{
				"Prjn.Learn.Learn": "false",
				"Prjn.WtInit.Mean": "0.8",
				"Prjn.WtInit.Var":  "0.0",
			}},
		{Sel: "#ECoutToECin", Desc: "one-to-one out to in",
			Params: params.Params{
				"Prjn.Learn.Learn": "false",
				"Prjn.WtInit.Mean": "0.9",
				"Prjn.WtInit.Var":  "0.01",
				"Prjn.WtScale.Rel": "0.5",
			}},
		{Sel: "#DGToCA3", Desc: "Mossy fibers: strong, non-learning",
			Params: params.Params{
				"Prjn.Learn.Learn": "false",
				"Prjn.WtInit.Mean": "0.9",
				"Prjn.WtInit.Var":  "0.01",
				"Prjn.WtScale.Rel": "4",
			}},
		{Sel: "#CA3ToCA3", Desc: "CA3 recurrent cons",
			Params: params.Params{
				"Prjn.WtScale.Rel": "0.1",
				"Prjn.Learn.Lrate": "0.1",
			}},
		{Sel: "#ECinToDG", Desc: "DG learning is surprisingly critical: maxed out fast, hebbian works best",
			Params: params.Params{
				"Prjn.Learn.Learn":       "true", // absolutely essential to have on!
				"Prjn.CHL.Hebb":          ".5",   // .5 > 1 overall
				"Prjn.CHL.SAvgCor":       "0.1",  // .1 > .2 > .3 > .4 ?
				"Prjn.CHL.MinusQ1":       "true", // dg self err?
				"Prjn.Learn.Lrate":       "0.4",  // .4 > .3 > .2
				"Prjn.Learn.Momentum.On": "false",
				"Prjn.Learn.Norm.On":     "false",
				"Prjn.Learn.WtBal.On":    "true",
			}},
		{Sel: "#CA3ToCA1", Desc: "Schaffer collaterals -- slower, less hebb",
			Params: params.Params{
				"Prjn.CHL.Hebb":          "0.01",
				"Prjn.CHL.SAvgCor":       "0.4",
				"Prjn.Learn.Lrate":       "0.1",
				"Prjn.Learn.Momentum.On": "false",
				"Prjn.Learn.Norm.On":     "false",
				"Prjn.Learn.WtBal.On":    "true",
			}},
		{Sel: ".EC", Desc: "all EC layers: only pools, no layer-level",
			Params: params.Params{
				"Layer.Act.Gbar.L":        ".1",
				"Layer.Inhib.ActAvg.Init": "0.2",
				"Layer.Inhib.Layer.On":    "false",
				"Layer.Inhib.Pool.Gi":     "2.0",
				"Layer.Inhib.Pool.On":     "true",
			}},
		{Sel: "#DG", Desc: "very sparse = high inibhition",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Init": "0.01",
				"Layer.Inhib.Layer.Gi":    "3.8",
			}},
		{Sel: "#CA3", Desc: "sparse = high inibhition",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Init": "0.02",
				"Layer.Inhib.Layer.Gi":    "2.8",
			}},
		{Sel: "#CA1", Desc: "CA1 only Pools",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Init": "0.1",
				"Layer.Inhib.Layer.On":    "false",
				"Layer.Inhib.Pool.Gi":     "2.4",
				"Layer.Inhib.Pool.On":     "true",
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

	// simulation configuration parameters -- set by .toml config file and / or args
	Config Config `new-window:"+"`

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *leabra.Network `new-window:"+" display:"no-inline"`

	// all parameter management
	Params emer.NetParams `display:"add-fields"`

	// contains looper control loops for running sim
	Loops *looper.Manager `new-window:"+" display:"no-inline"`

	// contains computed statistic values
	Stats estats.Stats `new-window:"+"`

	// Contains all the logs and information about the logs.'
	Logs elog.Logs `new-window:"+"`

	// if true, run in pretrain mode
	PretrainMode bool

	// pool patterns vocabulary
	PoolVocab patgen.Vocab `display:"no-inline"`

	// AB training patterns to use
	TrainAB *table.Table `display:"no-inline"`

	// AC training patterns to use
	TrainAC *table.Table `display:"no-inline"`

	// AB testing patterns to use
	TestAB *table.Table `display:"no-inline"`

	// AC testing patterns to use
	TestAC *table.Table `display:"no-inline"`

	// Lure pretrain patterns to use
	PreTrainLure *table.Table `display:"no-inline"`

	// Lure testing patterns to use
	TestLure *table.Table `display:"no-inline"`

	// all training patterns -- for pretrain
	TrainAll *table.Table `display:"no-inline"`

	// TestAB + TestAC
	TestABAC *table.Table `display:"no-inline"`

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
	// ss.Config.Defaults()
	econfig.Config(&ss.Config, "config.toml")
	// ss.Config.Hip.EC5Clamp = true      // must be true in hip.go to have a target layer
	// ss.Config.Hip.EC5ClampTest = false // key to be off for cmp stats on completion region

	ss.Net = leabra.NewNetwork("Hip")
	// ss.Params.Config(ParamSets, ss.Config.Params.Sheet, ss.Config.Params.Tag, ss.Net)
	ss.Stats.Init()

	ss.PoolVocab = patgen.Vocab{}
	ss.TrainAB = &table.Table{}
	ss.TrainAC = &table.Table{}
	ss.TestAB = &table.Table{}
	ss.TestAC = &table.Table{}
	ss.PreTrainLure = &table.Table{}
	ss.TestLure = &table.Table{}
	ss.TrainAll = &table.Table{}
	ss.TestABAC = &table.Table{}
	ss.PretrainMode = false

	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) ConfigAll() {
	ss.ConfigPats()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
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
	trn.Config(table.NewIndexView(ss.TrainAB))
	trn.Validate()

	tst.Name = etime.Test.String()
	tst.Config(table.NewIndexView(ss.TestABAC))
	tst.Sequential = true
	tst.Validate()

	trn.Init(0)
	tst.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	in := net.AddLayer4D("Input", 6, 2, 3, 4, leabra.InputLayer)
	ecin := net.AddLayer4D("ECin", 6, 2, 3, 4, leabra.SuperLayer)
	ecout := net.AddLayer4D("ECout", 6, 2, 3, 4, leabra.TargetLayer) // clamped in plus phase
	ca1 := net.AddLayer4D("CA1", 6, 2, 4, 10, leabra.SuperLayer)
	dg := net.AddLayer2D("DG", 25, 25, leabra.SuperLayer)
	ca3 := net.AddLayer2D("CA3", 30, 10, leabra.SuperLayer)

	ecin.AddClass("EC")
	ecout.AddClass("EC")

	onetoone := paths.NewOneToOne()
	pool1to1 := paths.NewPoolOneToOne()
	full := paths.NewFull()

	net.ConnectLayers(in, ecin, onetoone, leabra.ForwardPath)
	net.ConnectLayers(ecout, ecin, onetoone, leabra.BackPath)

	// EC <-> CA1 encoder pathways
	net.ConnectLayers(ecin, ca1, pool1to1, leabra.EcCa1Path)
	net.ConnectLayers(ca1, ecout, pool1to1, leabra.EcCa1Path)
	net.ConnectLayers(ecout, ca1, pool1to1, leabra.EcCa1Path)

	// Perforant pathway
	ppath := paths.NewUniformRand()
	ppath.PCon = 0.25

	net.ConnectLayers(ecin, dg, ppath, leabra.CHLPath).AddClass("HippoCHL")

	net.ConnectLayers(ecin, ca3, ppath, leabra.EcCa1Path).AddClass("PPath")
	net.ConnectLayers(ca3, ca3, full, leabra.EcCa1Path).AddClass("PPath")

	// Mossy fibers
	mossy := paths.NewUniformRand()
	mossy.PCon = 0.02
	net.ConnectLayers(dg, ca3, mossy, leabra.CHLPath).AddClass("HippoCHL")

	// Schafer collaterals
	net.ConnectLayers(ca3, ca1, full, leabra.CHLPath).AddClass("HippoCHL")

	net.Build()
	net.Defaults()
	ss.ApplyParams()
	net.InitWeights()
	net.InitTopoScales()
}

func (ss *Sim) ApplyParams() {
	ss.Params.Network = ss.Net
	ss.Params.SetAll()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // in case user interactively changes tag
	ss.Loops.ResetCounters()

	ss.GUI.StopNow = false
	ss.ApplyParams()
	ss.NewRun()
	ss.ViewUpdate.RecordSyns()
	ss.ViewUpdate.Update()
}

func (ss *Sim) TestInit() {
	ss.Loops.ResetCountersByMode(etime.Test)
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	rand.Seed(ss.RandSeeds[run])
	ss.RandSeeds.Set(run)
	ss.RandSeeds.Set(run, &ss.Net.Rand)
	patgen.NewRand(ss.RandSeeds[run])
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()

	trls := 10

	man.AddStack(etime.Train).AddTime(etime.Run, ss.Config.NRuns).AddTime(etime.Epoch, ss.Config.NEpochs).AddTime(etime.Trial, trls).AddTime(etime.Cycle, 200)

	man.AddStack(etime.Test).AddTime(etime.Epoch, 1).AddTime(etime.Trial, 2*trls).AddTime(etime.Cycle, 200)

	leabra.LooperStdPhases(man, &ss.Context, ss.Net, 150, 199)              // plus phase timing
	leabra.LooperSimCycleAndLearn(man, ss.Net, &ss.Context, &ss.ViewUpdate) // std algo code

	// ss.Net.ConfigLoopsHip(&ss.Context, man, &ss.Config.Hip, &ss.PretrainMode)

	for m, _ := range man.Stacks {
		stack := man.Stacks[m]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
	}

	man.GetLoop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	// Add Testing
	trainEpoch := man.GetLoop(etime.Train, etime.Epoch)
	trainEpoch.OnEnd.Add("TestAtInterval", func() {
		if (ss.Config.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.Config.TestInterval == 0) {
			// Note the +1 so that it doesn't occur at the 0th timestep.
			ss.TestAll()

			// switch to AC
			// trn := ss.Envs.ByMode(etime.Train).(*env.FixedTable)
			tstEpcLog := ss.Logs.Tables[etime.Scope(etime.Test, etime.Epoch)]
			epc := ss.Stats.Int("Epoch")
			abMem := float32(tstEpcLog.Table.Float("ABMem", epc))
			_ = abMem
			/*			if (trn.Table.Table.MetaData["name"] == "TrainAB") && (abMem >= ss.Config.StopMem || epc == ss.Config.Epochs/2) {
						ss.Stats.SetInt("FirstPerfect", epc)
						trn.Config(table.NewIndexView(ss.TrainAC))
						trn.Validate()
					} */
		}
	})

	// early stop
	man.GetLoop(etime.Train, etime.Epoch).IsDone["ACMemStop"] = func() bool {
		// This is calculated in TrialStats
		tstEpcLog := ss.Logs.Tables[etime.Scope(etime.Test, etime.Epoch)]
		acMem := float32(tstEpcLog.Table.Float("ACMem", ss.Stats.Int("Epoch")))
		_ = acMem
		// stop := acMem >= ss.Config.StopMem
		// return stop
		return false
	}

	/////////////////////////////////////////////
	// Logging

	man.GetLoop(etime.Test, etime.Epoch).OnEnd.Add("LogTestErrors", func() {
		leabra.LogTestErrors(&ss.Logs)
	})

	man.AddOnEndToAll("Log", ss.Log)
	leabra.LooperResetLogBelow(man, &ss.Logs)

	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("RunStats", func() {
		ss.Logs.RunStats("PctCor", "FirstZero", "LastZero")
	})

	leabra.LooperUpdateNetView(man, &ss.ViewUpdate, ss.Net, ss.NetViewCounters)
	leabra.LooperUpdatePlots(man, &ss.GUI)
	ss.Loops = man
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	ctx := &ss.Context
	net := ss.Net
	ev := ss.Envs.ByMode(ctx.Mode).(*env.FixedTable)
	lays := net.LayersByType(leabra.InputLayer, leabra.TargetLayer)
	net.InitExt()
	ev.Step()
	// note: must save env state for logging / stats due to data parallel re-use of same env
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
	ss.InitRandSeed(ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur)
	ss.ConfigPats()
	ss.ConfigEnv()
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

/////////////////////////////////////////////////////////////////////////
//   Pats

func (ss *Sim) ConfigPats() {
	// hp := &ss.Config.Hip
	ecY := 5               // hp.EC3NPool.Y
	ecX := 5               // hp.EC3NPool.X
	plY := 6               // hp.EC3NNrn.Y // good idea to get shorter vars when used frequently
	plX := 2               // hp.EC3NNrn.X // makes much more readable
	npats := 10            // ss.Config.NTrials
	pctAct := float32(.15) // ss.Config.Mod.ECPctAct
	minDiff := float32(6)  // ss.Config.Pat.MinDiffPct
	nOn := patgen.NFromPct(pctAct, plY*plX)
	ctxtFlipPct := float32(0.2)
	ctxtflip := patgen.NFromPct(ctxtFlipPct, nOn)
	patgen.AddVocabEmpty(ss.PoolVocab, "empty", npats, plY, plX)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "A", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "B", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "C", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lA", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lB", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "ctxt", 3, plY, plX, pctAct, minDiff) // totally diff

	for i := 0; i < (ecY-1)*ecX*3; i++ { // 12 contexts! 1: 1 row of stimuli pats; 3: 3 diff ctxt bases
		list := i / ((ecY - 1) * ecX)
		ctxtNm := fmt.Sprintf("ctxt%d", i+1)
		tsr, _ := patgen.AddVocabRepeat(ss.PoolVocab, ctxtNm, npats, "ctxt", list)
		patgen.FlipBitsRows(tsr, ctxtflip, ctxtflip, 1, 0)
		//todo: also support drifting
		//solution 2: drift based on last trial (will require sequential learning)
		//patgen.VocabDrift(ss.PoolVocab, ss.NFlipBits, "ctxt"+strconv.Itoa(i+1))
	}

	patgen.InitPats(ss.TrainAB, "TrainAB", "TrainAB Pats", "Input", "EC5", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TrainAB, ss.PoolVocab, "Input", []string{"A", "B", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})
	patgen.MixPats(ss.TrainAB, ss.PoolVocab, "EC5", []string{"A", "B", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})

	patgen.InitPats(ss.TestAB, "TestAB", "TestAB Pats", "Input", "EC5", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TestAB, ss.PoolVocab, "Input", []string{"A", "empty", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})
	patgen.MixPats(ss.TestAB, ss.PoolVocab, "EC5", []string{"A", "B", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})

	patgen.InitPats(ss.TrainAC, "TrainAC", "TrainAC Pats", "Input", "EC5", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TrainAC, ss.PoolVocab, "Input", []string{"A", "C", "ctxt5", "ctxt6", "ctxt7", "ctxt8"})
	patgen.MixPats(ss.TrainAC, ss.PoolVocab, "EC5", []string{"A", "C", "ctxt5", "ctxt6", "ctxt7", "ctxt8"})

	patgen.InitPats(ss.TestAC, "TestAC", "TestAC Pats", "Input", "EC5", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TestAC, ss.PoolVocab, "Input", []string{"A", "empty", "ctxt5", "ctxt6", "ctxt7", "ctxt8"})
	patgen.MixPats(ss.TestAC, ss.PoolVocab, "EC5", []string{"A", "C", "ctxt5", "ctxt6", "ctxt7", "ctxt8"})

	patgen.InitPats(ss.PreTrainLure, "PreTrainLure", "PreTrainLure Pats", "Input", "EC5", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.PreTrainLure, ss.PoolVocab, "Input", []string{"lA", "lB", "ctxt9", "ctxt10", "ctxt11", "ctxt12"}) // arbitrary ctxt here
	patgen.MixPats(ss.PreTrainLure, ss.PoolVocab, "EC5", []string{"lA", "lB", "ctxt9", "ctxt10", "ctxt11", "ctxt12"})   // arbitrary ctxt here

	patgen.InitPats(ss.TestLure, "TestLure", "TestLure Pats", "Input", "EC5", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TestLure, ss.PoolVocab, "Input", []string{"lA", "empty", "ctxt9", "ctxt10", "ctxt11", "ctxt12"}) // arbitrary ctxt here
	patgen.MixPats(ss.TestLure, ss.PoolVocab, "EC5", []string{"lA", "lB", "ctxt9", "ctxt10", "ctxt11", "ctxt12"})      // arbitrary ctxt here

	ss.TrainAll = ss.TrainAB.Clone()
	ss.TrainAll.AppendRows(ss.TrainAC)
	ss.TrainAll.AppendRows(ss.PreTrainLure)
	ss.TrainAll.MetaData["name"] = "TrainAll"
	ss.TrainAll.MetaData["desc"] = "All Training Patterns"

	ss.TestABAC = ss.TestAB.Clone()
	ss.TestABAC.AppendRows(ss.TestAC)
	ss.TestABAC.MetaData["name"] = "TestABAC"
	ss.TestABAC.MetaData["desc"] = "All Testing Patterns"
}

func (ss *Sim) OpenPats() {
	dt := ss.TrainAB
	dt.SetMetaData("name", "TrainAB")
	dt.SetMetaData("desc", "Training patterns")
	err := dt.OpenCSV("random_5x5_25.tsv", table.Tab)
	if err != nil {
		log.Println(err)
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetFloat("UnitErr", 0.0)
	ss.Stats.SetFloat("PhaseDiff", 0.0)
	ss.Stats.SetFloat("TrgOnWasOffAll", 0.0)
	ss.Stats.SetFloat("TrgOnWasOffCmp", 0.0)
	ss.Stats.SetFloat("TrgOffWasOn", 0.0)
	ss.Stats.SetFloat("ABMem", 0.0)
	ss.Stats.SetFloat("ACMem", 0.0)
	ss.Stats.SetFloat("Mem", 0.0)
	ss.Stats.SetInt("FirstPerfect", -1) // first epoch at when AB Mem is perfect
	ss.Stats.SetInt("RecallItem", -1)   // item recalled in EC5 completion pool
	ss.Stats.SetFloat("ABRecMem", 0.0)  // similar to ABMem but using correlation on completion pool
	ss.Stats.SetFloat("ACRecMem", 0.0)

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
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "Di", "TrialName", "Cycle", "UnitErr", "TrlErr", "PhaseDiff"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	out := ss.Net.LayerByName("EC5")
	_ = out

	// ss.Stats.SetFloat("PhaseDiff", float64(out.Values[di].PhaseDiff.Cor))
	// ss.Stats.SetFloat("UnitErr", out.PctUnitErr(&ss.Context)[di])
	ss.MemStats(ss.Loops.Mode)

	// if ss.Stats.Float("UnitErr") > ss.Config.Mod.MemThr {
	// 	ss.Stats.SetFloat("TrlErr", 1)
	// } else {
	// 	ss.Stats.SetFloat("TrlErr", 0)
	// }
}

// MemStats computes ActM vs. Target on ECout with binary counts
// must be called at end of 3rd quarter so that Target values are
// for the entire full pattern as opposed to the plus-phase target
// values clamped from ECin activations
func (ss *Sim) MemStats(mode etime.Modes) {
	/*
		memthr := 0.3 // ss.Config.Mod.MemThr
		ecout := ss.Net.LayerByName("EC5")
		inp := ss.Net.LayerByName("Input") // note: must be input b/c ECin can be active
		_ = inp
		nn := ecout.Shape.Len()
		actThr := float32(0.2)
		trgOnWasOffAll := 0.0 // all units
		trgOnWasOffCmp := 0.0 // only those that required completion, missing in ECin
		trgOffWasOn := 0.0    // should have been off
		cmpN := 0.0           // completion target
		trgOnN := 0.0
		trgOffN := 0.0
		actMi, _ := ecout.UnitVarIndex("ActM")
		targi, _ := ecout.UnitVarIndex("Target")

		ss.Stats.SetFloat("ABMem", math.NaN())
		ss.Stats.SetFloat("ACMem", math.NaN())
		ss.Stats.SetFloat("ABRecMem", math.NaN())
		ss.Stats.SetFloat("ACRecMem", math.NaN())

		trialnm := ss.Stats.String("TrialName")
		isAB := strings.Contains(trialnm, "AB")

		// for ni := 0; ni < nn; ni++ {
		// 	actm := ecout.UnitValue1D(actMi, ni, di)
		// 	trg := ecout.UnitValue1D(targi, ni, di) // full pattern target
		// 	inact := inp.UnitValue1D(actMi, ni, di)
		// 	if trg < actThr { // trgOff
		// 		trgOffN += 1
		// 		if actm > actThr {
		// 			trgOffWasOn += 1
		// 		}
		// 	} else { // trgOn
		// 		trgOnN += 1
		// 		if inact < actThr { // missing in ECin -- completion target
		// 			cmpN += 1
		// 			if actm < actThr {
		// 				trgOnWasOffAll += 1
		// 				trgOnWasOffCmp += 1
		// 			}
		// 		} else {
		// 			if actm < actThr {
		// 				trgOnWasOffAll += 1
		// 			}
		// 		}
		// 	}
		// }
		trgOnWasOffAll /= trgOnN
		trgOffWasOn /= trgOffN
		if mode == etime.Train { // no compare
			if trgOnWasOffAll < memthr && trgOffWasOn < memthr {
				ss.Stats.SetFloat("Mem", 1)
			} else {
				ss.Stats.SetFloat("Mem", 0)
			}
		} else { // test
			if cmpN > 0 { // should be
				trgOnWasOffCmp /= cmpN
				if trgOnWasOffCmp < memthr && trgOffWasOn < memthr {
					ss.Stats.SetFloat("Mem", 1)
					if isAB {
						ss.Stats.SetFloat("ABMem", 1)
					} else {
						ss.Stats.SetFloat("ACMem", 1)
					}
				} else {
					ss.Stats.SetFloat("Mem", 0)
					if isAB {
						ss.Stats.SetFloat("ABMem", 0)
					} else {
						ss.Stats.SetFloat("ACMem", 0)
					}
				}
			}
		}
		ss.Stats.SetFloat("TrgOnWasOffAll", trgOnWasOffAll)
		ss.Stats.SetFloat("TrgOnWasOffCmp", trgOnWasOffCmp)
		ss.Stats.SetFloat("TrgOffWasOn", trgOffWasOn)

		// take completion pool to do CosDiff
		var recallPat tensor.Float32
		ecout.UnitValuesTensor(&recallPat, "ActM", di)
		mostSimilar := -1
		highestCosDiff := float32(0)
		var cosDiff float32
		var patToComplete *tensor.Float32
		var correctIndex int
		if isAB {
			patToComplete, _ = ss.PoolVocab.ByName("B")
			correctIndex, _ = strconv.Atoi(strings.Split(trialnm, "AB")[1])
		} else {
			patToComplete, _ = ss.PoolVocab.ByName("C")
			correctIndex, _ = strconv.Atoi(strings.Split(trialnm, "AC")[0])
		}
		for i := 0; i < patToComplete.DimSize(0); i++ { // for each item in the list
			cosDiff = metric.Correlation32(recallPat.SubSpace([]int{0, 1}).(*tensor.Float32).Values, patToComplete.SubSpace([]int{i}).(*tensor.Float32).Values)
			if cosDiff > highestCosDiff {
				highestCosDiff = cosDiff
				mostSimilar = i
			}
		}

		ss.Stats.SetInt("RecallItem", mostSimilar)
		if isAB {
			ss.Stats.SetFloat("ABRecMem", num.FromBool[float64](mostSimilar == correctIndex))
		} else {
			ss.Stats.SetFloat("ACRecMem", num.FromBool[float64](mostSimilar == correctIndex))
		}
	*/
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) AddLogItems() {
	itemNames := []string{"PhaseDiff", "UnitErr", "PctCor", "PctErr", "TrgOnWasOffAll", "TrgOnWasOffCmp", "TrgOffWasOn", "Mem", "ABMem", "ACMem", "ABRecMem", "ACRecMem"}
	for _, st := range itemNames {
		stnm := st
		tonm := "Tst" + st
		ss.Logs.AddItem(&elog.Item{
			Name: tonm,
			Type: reflect.Float64,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
					ctx.SetFloat64(ctx.ItemFloat(etime.Test, etime.Epoch, stnm))
				},
				etime.Scope(etime.Train, etime.Run): func(ctx *elog.Context) {
					ctx.SetFloat64(ctx.ItemFloat(etime.Test, etime.Epoch, stnm)) // take the last epoch
					// ctx.SetAgg(ctx.Mode, etime.Epoch, stats.Max) // stats.Max for max over epochs
				}}})
	}
}

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.Trial, "Di")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")

	ss.Logs.AddStatAggItem("PhaseDiff", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("UnitErr", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("TrgOnWasOffAll", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("TrgOnWasOffCmp", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("TrgOffWasOn", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("ABMem", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("ACMem", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("Mem", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("ABRecMem", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("ACRecMem", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatIntNoAggItem(etime.Train, etime.Run, "FirstPerfect")
	ss.Logs.AddStatIntNoAggItem(etime.Train, etime.Trial, "RecallItem")
	ss.Logs.AddStatIntNoAggItem(etime.Test, etime.Trial, "RecallItem")
	ss.Logs.AddErrStatAggItems("TrlErr", etime.Run, etime.Epoch, etime.Trial)

	// ss.Logs.AddCopyFromFloatItems(etime.Train, etime.Epoch, etime.Test, etime.Epoch, "Tst", "PhaseDiff", "UnitErr", "PctCor", "PctErr", "TrgOnWasOffAll", "TrgOnWasOffCmp", "TrgOffWasOn", "Mem")
	ss.AddLogItems()

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	layers := ss.Net.LayersByType(leabra.SuperLayer, leabra.CTLayer, leabra.TargetLayer)
	leabra.LogAddDiagnosticItems(&ss.Logs, layers, etime.Train, etime.Epoch, etime.Trial)
	leabra.LogInputLayer(&ss.Logs, ss.Net, etime.Train)

	// leabra.LogAddPCAItems(&ss.Logs, ss.Net, etime.Train, etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddLayerTensorItems(ss.Net, "ActM", etime.Test, etime.Trial, "TargetLayer")
	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "TargetLayer")

	ss.Logs.PlotItems("TrgOnWasOffAll", "TrgOnWasOffCmp", "ABMem", "ACMem", "ABRecMem", "ACRecMem", "TstTrgOnWasOffAll", "TstTrgOnWasOffCmp", "TstMem", "TstABMem", "TstACMem", "TstABRecMem", "TstACRecMem")

	ss.Logs.CreateTables()
	ss.Logs.SetMeta(etime.Train, etime.Run, "TrgOnWasOffAll:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "TrgOnWasOffCmp:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "ABMem:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "ACMem:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "ABRecMem:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "ACRecMem:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "TstTrgOnWasOffAll:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "TstTrgOnWasOffCmp:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "TstMem:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "TstACMem:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "TstACRecMem:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "FirstPerfect:On", "+")
	ss.Logs.SetMeta(etime.Train, etime.Run, "Type", "Bar")
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	// note: Analyze not plotted by default
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
		ss.Logs.LogRow(mode, time, row)
		return // don't do reg below
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "Leabra Hippocampus"
	ss.GUI.MakeBody(ss, "hip", title, `Benchmarking`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 300
	nv.SetNet(ss.Net)
	ss.ViewUpdate.Config(nv, etime.Phase, etime.Phase)
	ss.GUI.ViewUpdate = &ss.ViewUpdate

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1, 2.75) // more "head on" than default which is more "top down"
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Init", Icon: icons.Update,
		Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Init()
			ss.GUI.UpdateWindow()
		},
	})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Test Init", Icon: icons.Update,
		Tooltip: "Call ResetCountersByMode with test mode and update GUI.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.TestInit()
			ss.GUI.UpdateWindow()
		},
	})

	ss.GUI.AddLooperCtrl(p, ss.Loops, []etime.Modes{etime.Train, etime.Test})

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
			core.TheApp.OpenURL("https://github.com/emer/leabra/blob/main/examples/hip/README.md")
		},
	})
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}
