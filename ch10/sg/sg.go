// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// sg is the sentence gestalt model, which learns to encode both
// syntax and semantics of sentences in an integrated "gestalt"
// hidden layer. The sentences have simple agent-verb-patient
// structure with optional prepositional or adverb modifier
// phrase at the end, and can be either in the active or passive
// form (80% active, 20% passive). There are ambiguous terms that
// need to be resolved via context, showing a key interaction
// between syntax and semantics.
package main

//go:generate core generate -add-types

import (
	"embed"
	"strings"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/randx"
	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tensor/stats/clust"
	"cogentcore.org/core/tensor/stats/metric"
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
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/leabra/v2/leabra"
	"gonum.org/v1/gonum/mat"
)

//go:embed trained.wts sg_rules.txt sg_tests.txt sg_probes.txt
var content embed.FS

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
		{Sel: "Path", Desc: "norm and momentum on is critical, wt bal not as much but fine",
			Params: params.Params{
				"Path.Learn.Norm.On":     "true",
				"Path.Learn.Momentum.On": "true",
				"Path.Learn.WtBal.On":    "true",
				"Path.Learn.Lrate":       "0.04", // critical for lrate sched
			}},
		{Sel: "Layer", Desc: "more inhibition is better",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi":  "2.4", // 2.4 > 2.6+ > 2.2-
				"Layer.Learn.AvgL.Gain": "2.5", // 2.5 > 3
				"Layer.Act.Gbar.L":      "0.1", // lower leak = better
				"Layer.Act.Init.Decay":  "0",
			}},
		{Sel: ".PulvinarLayer", Desc: "standard weight is .3 here for larger distributed reps. no learn",
			Params: params.Params{
				"Layer.Pulvinar.DriveScale":   "0.8", // using .8 for localist layer
				"Layer.Inhib.ActAvg.UseFirst": "false",
			}},
		{Sel: ".CTLayer", Desc: "don't use first as it is typically very low",
			Params: params.Params{
				"Layer.Inhib.ActAvg.UseFirst": "false",
			}},
		{Sel: ".Encode", Desc: "except encoder needs less",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "1.8", // 1.8 > 2.0 > 1.6 > 2.2
			}},
		{Sel: "#Encode", Desc: "except encoder needs less",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "2.0", // 1.8 > 2.0 > 1.6 > 2.2
			}},
		{Sel: "#Decode", Desc: "except decoder needs less",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "1.8", // 1.8 > 2.0+
			}},
		{Sel: ".Gestalt", Desc: "gestalt needs more inhib",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "2.4", // 2.4 > 2.2 > 2.6 -- very sensitive!
			}},
		{Sel: "#Filler", Desc: "higher inhib, 3.6 > 3.8 > 3.4 > 3.2 > 3.0 > 2.8 -- key for ambig!",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "3.6",
			}},
		{Sel: "#EncodeP", Desc: "higher inhib -- 2.4 == 2.2 > 2.6",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "2.4",
			}},
		{Sel: ".BackPath", Desc: "weaker back -- .2 > .3, .1",
			Params: params.Params{
				"Path.WtScale.Rel": "0.2",
			}},
		{Sel: ".CTCtxtPath", Desc: "yes weight balance",
			Params: params.Params{
				"Path.Learn.WtBal.On": "true", // true > false
				"Path.WtScale.Rel":    "1",    // 1 > 2
			}},
		{Sel: ".CTFromSuper", Desc: "from superficial layer",
			Params: params.Params{
				"Path.WtInit.Mean": "0.5",
			}},
		{Sel: ".GestSelfCtxt", Desc: "yes weight balance",
			Params: params.Params{
				"Path.WtScale.Rel": "3", // 3 > 2 > 4 -- not better to start smaller
			}},
		{Sel: ".EncSelfCtxt", Desc: "yes weight balance",
			Params: params.Params{
				"Path.WtScale.Rel": "5", // 5 > 4 > 3 > 6 -- not better to start smaller
			}},
		{Sel: ".CtxtBack", Desc: "gest CT - > encode CT basically",
			Params: params.Params{
				"Path.WtScale.Rel": "1",
			}},
		{Sel: ".FmInput", Desc: "from localist inputs -- 1 == .3",
			Params: params.Params{
				"Path.WtScale.Rel":      "1",
				"Path.Learn.WtSig.Gain": "6", // 1 == 6
			}},
		{Sel: ".EncodePToSuper", Desc: "teaching signal from input pulvinar, to super -- .05 > .2",
			Params: params.Params{
				"Path.WtScale.Rel": "0.05", // .05 == .02 > .2
			}},
		{Sel: ".EncodePToCT", Desc: "critical to make this small so deep context dominates -- .05",
			Params: params.Params{
				"Path.WtScale.Rel": "0.05", // .05 == .02
			}},
		{Sel: ".CtxtFmInput", Desc: "making this weaker than 1 causes encodeD to freeze, 1 == 1.5 > lower",
			Params: params.Params{
				"Path.WtScale.Rel": "1.0",
			}},
		{Sel: "#DecodeToGestaltCT", Desc: "this leaks current role into context directly",
			Params: params.Params{
				"Path.WtScale.Rel": "0.2", // .2 > .3 > .1 > .05(bad) > .02(vbad)
			}},
	},
}

// Config has config parameters related to running the sim
type Config struct {
	// total number of runs to do when running Train
	NRuns int `default:"1" min:"1"`

	// total number of epochs per run
	NEpochs int `default:"500"`

	// stop run after this number of perfect, zero-error epochs.
	NZero int `default:"5"`

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

	// Config contains misc configuration parameters for running the sim
	Config Config `new-window:"+" display:"no-inline"`

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *leabra.Network `new-window:"+" display:"no-inline"`

	// network parameter management
	Params emer.NetParams `display:"add-fields"`

	// contains looper control loops for running sim
	Loops *looper.Manager `new-window:"+" display:"no-inline"`

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
	ss.Net = leabra.NewNetwork("SG")
	ss.Params.Config(ParamSets, "", "", ss.Net)
	ss.Stats.Init()
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
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
	var trn, tst *SentGenEnv
	if len(ss.Envs) == 0 {
		trn = &SentGenEnv{}
		tst = &SentGenEnv{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*SentGenEnv)
		tst = ss.Envs.ByMode(etime.Test).(*SentGenEnv)
	}

	// note: names must be standard here!
	trn.Name = etime.Train.String()
	trn.Seq.Max = 100 // sequences per epoch training
	trn.OpenRulesFromAsset("sg_rules.txt")
	trn.PPassive = 0.2
	trn.Words = SGWords
	trn.Roles = SGRoles
	trn.Fillers = SGFillers
	trn.WordTrans = SGWordTrans
	trn.AmbigVerbs = SGAmbigVerbs
	trn.AmbigNouns = SGAmbigNouns
	trn.Validate()

	tst.Name = etime.Test.String()
	tst.Seq.Max = 14
	tst.OpenRulesFromAsset("sg_tests.txt")
	//	tst.Rules.OpenRules("sg_tests.txt")
	tst.PPassive = 0 // passive explicitly marked
	tst.Words = SGWords
	tst.Roles = SGRoles
	tst.Fillers = SGFillers
	tst.WordTrans = SGWordTrans
	tst.AmbigVerbs = SGAmbigVerbs
	tst.AmbigNouns = SGAmbigNouns
	tst.Validate()

	trn.Init(0)
	tst.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	// overall strategy:
	//
	// Encode does pure prediction of next word, which remains about 60% correct at best
	// Gestalt gets direct word input, does full error-driven fill-role learning
	// via decoder.
	//
	// Gestalt can be entirely independent of encode, or recv encode -- testing value.
	// GestaltD depends *critically* on getting direct error signal from Decode!
	//
	// For pure predictive encoder, EncodeP -> Gestalt is bad.  if we leak Decode
	// error signal back to Encode, then it is actually useful, as is GestaltD -> EncodeP
	//
	// run notes:
	// 54 = no enc <-> gestalt -- not much diff..  probably just get rid of enc then?
	// 48 = enc -> gestalt still, no inp -> gest
	// 44 = gestd -> encd, otherwise same as 48 -- improves inp pred due to leak via gestd, else fill same
	// 43 = best perf overall -- 44 + gestd -> inp  -- inp a bit better
	//

	in := net.AddLayer2D("Input", 10, 5, leabra.InputLayer)
	role := net.AddLayer2D("Role", 9, 1, leabra.InputLayer)
	fill := net.AddLayer2D("Filler", 11, 5, leabra.TargetLayer)
	enc, encct, encp := net.AddDeep2D("Encode", 12, 12) // 12x12 better..
	enc.AddClass("Encode")
	encct.AddClass("Encode")
	dec := net.AddLayer2D("Decode", 12, 12, leabra.SuperLayer)
	gest, gestct := net.AddDeepNoPulvinar2D("Gestalt", 12, 12) // 12x12 def better with full
	gest.AddClass("Gestalt")
	gestct.AddClass("Gestalt")

	encp.Shape.CopyShape(&in.Shape)
	encp.Drivers.Add("Input")

	encp.PlaceRightOf(in, 2)
	role.PlaceRightOf(encp, 4)
	fill.PlaceRightOf(role, 4)
	enc.PlaceAbove(in)
	encct.PlaceRightOf(enc, 2)
	dec.PlaceRightOf(encct, 2)
	gest.PlaceAbove(enc)
	gestct.PlaceRightOf(gest, 2)

	full := paths.NewFull()
	full.SelfCon = true

	pj := net.ConnectLayers(in, enc, full, leabra.ForwardPath)
	pj.AddClass("FmInput")

	pj = net.ConnectLayers(in, gest, full, leabra.ForwardPath) // this is key -- skip encoder
	pj.AddClass("FmInput")

	errors.Log1(encct.RecvPathBySendName("EncodeP")).(*leabra.Path).AddClass("EncodePToCT")
	errors.Log1(enc.RecvPathBySendName("EncodeP")).(*leabra.Path).AddClass("EncodePToSuper")

	// gestd gets error from Filler, this communicates Filler to encd -> corrupts prediction
	// net.ConnectLayers(gestd, encd, full, leabra.ForwardPath)

	// testing no use of enc at all
	net.BidirConnectLayers(enc, gest, full)

	net.ConnectLayers(gestct, enc, full, leabra.BackPath) // give enc the best of gestd
	// net.ConnectLayers(gestd, gest, full, leabra.BackPath) // not essential?  todo retest

	// this allows current role info to propagate back to input prediction
	// does not seem to be important
	// net.ConnectLayers(gestd, inp, full, leabra.ForwardPath) // must be weaker..

	// if gestd not driving inp, then this is bad -- .005 MIGHT be tiny bit beneficial but not worth it
	// net.ConnectLayers(inp, gestd, full, leabra.BackPath).AddClass("EncodePToGestalt")
	// net.ConnectLayers(inp, gest, full, leabra.BackPath).AddClass("EncodePToGestalt")

	net.BidirConnectLayers(gest, dec, full)
	net.BidirConnectLayers(gestct, dec, full) // bidir is essential here to get error signal
	// directly into context layer -- has rel of 0.2

	// net.BidirConnectLayers(enc, dec, full) // not beneficial

	net.BidirConnectLayers(dec, role, full)
	net.BidirConnectLayers(dec, fill, full)

	// add extra deep context
	net.ConnectCtxtToCT(encct, encct, full).AddClass("EncSelfCtxt") // one2one doesn't work
	net.ConnectCtxtToCT(in, encct, full).AddClass("CtxtFmInput")
	net.ConnectCtxtToCT(gestct, encct, full).AddClass("CtxtBack")

	// add extra deep context
	net.ConnectCtxtToCT(gestct, gestct, full).AddClass("GestSelfCtxt") // full > one2one
	// net.ConnectCtxtToCT(in, gestct, full).AddClass("CtxtFmInput")
	// net.ConnectLayers(encp, gestct, full, leabra.BackPath).AddClass("EncodePToCT") // actually bad
	net.ConnectCtxtToCT(enc, gestct, full).AddClass("CtxtFmInput") // better than direct from in

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

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()

	trls := 100

	man.AddStack(etime.Train).
		AddTime(etime.Run, ss.Config.NRuns).
		AddTime(etime.Epoch, ss.Config.NEpochs).
		AddTime(etime.Trial, trls).
		AddTime(etime.Cycle, 100)

	man.AddStack(etime.Test).
		AddTime(etime.Epoch, 1).
		AddTime(etime.Trial, trls).
		AddTime(etime.Cycle, 100)

	leabra.LooperStdPhases(man, &ss.Context, ss.Net, 75, 99)                // plus phase timing
	leabra.LooperSimCycleAndLearn(man, ss.Net, &ss.Context, &ss.ViewUpdate) // std algo code

	for m, _ := range man.Stacks {
		stack := man.Stacks[m]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
	}

	man.GetLoop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	// Train stop early condition
	man.GetLoop(etime.Train, etime.Epoch).IsDone["NZeroStop"] = func() bool {
		// This is calculated in TrialStats
		stopNz := ss.Config.NZero
		if stopNz <= 0 {
			stopNz = 2
		}
		curNZero := ss.Stats.Int("NZero")
		stop := curNZero >= stopNz
		return stop
	}

	// Add Testing
	trainEpoch := man.GetLoop(etime.Train, etime.Epoch)
	trainEpoch.OnStart.Add("TestAtInterval", func() {
		if (ss.Config.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.Config.TestInterval == 0) {
			// Note the +1 so that it doesn't occur at the 0th timestep.
			ss.TestAll()
		}
	})

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

	////////////////////////////////////////////
	// GUI

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
	ev := ss.Envs.ByMode(ctx.Mode).(*SentGenEnv)
	ev.Step()

	out := ss.Net.LayerByName("Filler")
	if ctx.Mode == etime.Test {
		out.Type = leabra.CompareLayer // don't clamp plus phase
	} else {
		out.Type = leabra.TargetLayer
	}

	lays := net.LayersByType(leabra.InputLayer, leabra.TargetLayer)
	net.InitExt()
	ss.Stats.SetString("TrialName", ev.String())
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

////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetFloat("SSE", 0.0)
	ss.Stats.SetString("TrialName", "")
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
	out := ss.Net.LayerByName("Filler")

	sse, avgsse := out.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
	ss.Stats.SetFloat("SSE", sse)
	ss.Stats.SetFloat("AvgSSE", avgsse)
	if sse > 0 {
		ss.Stats.SetFloat("TrlErr", 1)
	} else {
		ss.Stats.SetFloat("TrlErr", 0)
	}
}

// RepsAnalysis analyzes the representations as captured in the Test Trial Log
func (ss *Sim) RepsAnalysis() {
	trl := ss.Logs.Table(etime.Test, etime.Trial)
	names := make([]string, trl.Rows)
	nmtsr := errors.Log1(trl.ColumnByName("TrialName")).(*tensor.String)
	copy(names, nmtsr.Values) // save

	// replace name with just rel
	for i, nm := range nmtsr.Values {
		nnm := nm[strings.Index(nm, ".")+1:]
		nnm = nnm[:strings.Index(nnm, ".")]
		nmtsr.Values[i] = nnm
	}

	ss.Stats.SVD.Kind = mat.SVDFull // critical
	rels := table.NewIndexView(trl)
	rels.SortColumnName("TrialName", table.Ascending)
	ss.Stats.SimMat("HiddenRel").TableColumn(rels, "Hidden_ActM", "TrialName", true, metric.Correlation64)
	errors.Log(ss.Stats.SVD.TableColumn(rels, "Hidden_ActM", metric.Covariance64))
	svt := ss.Logs.MiscTable("HiddenRelPCA")
	ss.Stats.SVD.ProjectColumnToTable(svt, rels, "Hidden_ActM", "TrialName", []int{0, 1})
	estats.ConfigPCAPlot(ss.GUI.PlotByName("HiddenRelPCA"), svt, "Hidden Rel PCA")
	estats.ClusterPlot(ss.GUI.PlotByName("HiddenRelClust"), rels, "Hidden_ActM", "TrialName", clust.ContrastDist)

	// replace name with just agent
	for i, nm := range names {
		nnm := nm[:strings.Index(nm, ".")]
		nmtsr.Values[i] = nnm
	}
	ags := table.NewIndexView(trl)
	ags.SortColumnName("TrialName", table.Ascending)

	ss.Stats.SimMat("HiddenAgent").TableColumn(ags, "Hidden_ActM", "TrialName", true, metric.Correlation64)
	errors.Log(ss.Stats.SVD.TableColumn(ags, "Hidden_ActM", metric.Covariance64))
	svt = ss.Logs.MiscTable("HiddenAgentPCA")
	ss.Stats.SVD.ProjectColumnToTable(svt, ags, "Hidden_ActM", "TrialName", []int{2, 3})
	estats.ConfigPCAPlot(ss.GUI.PlotByName("HiddenAgentPCA"), svt, "Hidden Agent PCA")
	estats.ClusterPlot(ss.GUI.PlotByName("HiddenAgentClust"), ags, "Hidden_ActM", "TrialName", clust.ContrastDist)

	// ss.HiddenAgent.PCAPlot.SetColParams("Path3", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	// ss.HiddenAgent.PCAPlot.Params.XAxisCol = "Path2"

	ss.Stats.SimMat("AgentCode").TableColumn(ags, "AgentCode_ActM", "TrialName", true, metric.Correlation64)
	errors.Log(ss.Stats.SVD.TableColumn(ags, "AgentCode_ActM", metric.Covariance64))
	svt = ss.Logs.MiscTable("AgentCodePCA")
	ss.Stats.SVD.ProjectColumnToTable(svt, ags, "AgentCode_ActM", "TrialName", []int{0, 1})
	estats.ConfigPCAPlot(ss.GUI.PlotByName("AgentCodePCA"), svt, "AgentCode Agent PCA")
	estats.ClusterPlot(ss.GUI.PlotByName("AgentCodeClust"), ags, "AgentCode_ActM", "TrialName", clust.ContrastDist)

	copy(nmtsr.Values, names) // restore
}

//////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")

	ss.Logs.AddStatAggItem("SSE", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("AvgSSE", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddErrStatAggItems("TrlErr", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddCopyFromFloatItems(etime.Train, []etime.Times{etime.Epoch, etime.Run}, etime.Test, etime.Epoch, "Tst", "SSE", "AvgSSE")

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddLayerTensorItems(ss.Net, "ActM", etime.Test, etime.Trial, "InputLayer", "SuperLayer", "TargetLayer")
	ss.Logs.AddLayerTensorItems(ss.Net, "Targ", etime.Test, etime.Trial, "TargetLayer")

	ss.Logs.PlotItems("PctErr", "FirstZero", "LastZero")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Trial)
	ss.Logs.NoPlot(etime.Test, etime.Run)
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
	title := "Family Trees"
	ss.GUI.MakeBody(ss, "family_trees", title, `family_trees shows how learning can recode inputs that have no similarity structure into a hidden layer that captures the *functional* similarity structure of the items. See <a href="https://github.com/CompCogNeuro/sims/blob/main/ch4/family_trees/README.md">README.md on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 300
	nv.Options.Raster.Max = 100
	nv.SetNet(ss.Net)
	nv.Options.PathWidth = 0.003
	ss.ViewUpdate.Config(nv, etime.GammaCycle, etime.GammaCycle)
	ss.GUI.ViewUpdate = &ss.ViewUpdate
	nv.Current()

	// nv.SceneXYZ().Camera.Pose.Pos.Set(0.1, 1.5, 4) // more "head on" than default which is more "top down"
	// nv.SceneXYZ().Camera.LookAt(math32.Vec3(0.1, 0.1, 0), math32.Vec3(0, 1, 0))

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.AddTableView(&ss.Logs, etime.Test, etime.Trial)

	ss.GUI.AddMiscPlotTab("HiddenRelPCA")
	ss.GUI.AddMiscPlotTab("HiddenRelClust")
	ss.GUI.AddMiscPlotTab("HiddenAgentPCA")
	ss.GUI.AddMiscPlotTab("HiddenAgentClust")
	ss.GUI.AddMiscPlotTab("AgentCodePCA")
	ss.GUI.AddMiscPlotTab("AgentCodeClust")

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

	ss.GUI.AddLooperCtrl(p, ss.Loops, []etime.Modes{etime.Train, etime.Test})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Test Init", Icon: icons.Update,
		Tooltip: "Initialize testing to start over -- if Test Step doesn't work, then do this.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Loops.ResetCountersByMode(etime.Test)
		},
	})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Reps Analysis",
		Icon:    icons.Reset,
		Tooltip: "analyzes the current testing Hidden and AgentCode activations, producing PCA, Cluster and Similarity Matrix (SimMat) plots",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.RepsAnalysis()
		},
	})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Open Trained Wts",
		Icon:    icons.Open,
		Tooltip: "Open trained weights",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.Net.OpenWeightsFS(content, "trained.wts")
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
			core.TheApp.OpenURL("https://github.com/CompCogNeuro/sims/blob/main/ch4/family_trees/README.md")
		},
	})
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}
