// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
stroop illustrates how the PFC can produce top-down biasing for executive control, in the context of the widely-studied Stroop task.
*/
package main

import (
	"embed"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"cogentcore.org/core/errors"
	"cogentcore.org/core/gimain"
	"cogentcore.org/core/math32"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/netview"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/prjn"
	"github.com/emer/emergent/v2/relpos"
	"github.com/emer/etable/v2/agg"
	"github.com/emer/etable/v2/eplot"
	"github.com/emer/etable/v2/etable"
	"github.com/emer/etable/v2/etensor"
	"github.com/emer/etable/v2/split"
	"github.com/emer/leabra/v2/leabra"
	"github.com/goki/ki/kit"
)

// this is the stub main for gogi that calls our actual mainrun function, at end of file
func main() {
	gimain.Main(func() {
		mainrun()
	})
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

//go:embed stroop_train.tsv stroop_test.tsv stroop_soa.tsv
var content embed.FS

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "lower lrate, uniform init",
				Params: params.Params{
					"Prjn.Learn.Norm.On":     "false",
					"Prjn.Learn.Momentum.On": "false",
					"Prjn.Learn.WtBal.On":    "false",
					"Prjn.Learn.Lrate":       "0.04",
					"Prjn.WtInit.Mean":       "0.25",
					"Prjn.WtInit.Var":        "0",
				}},
			{Sel: "Layer", Desc: "high inhibition, layer act avg",
				Params: params.Params{
					"Layer.Act.XX1.Gain":       "40",
					"Layer.Learn.AvgL.Gain":    "1", // critical params here
					"Layer.Learn.AvgL.Init":    "0.2",
					"Layer.Learn.AvgL.Min":     "0.05",
					"Layer.Learn.AvgL.LrnMin":  "0.05",
					"Layer.Learn.AvgL.LrnMax":  "0.05",
					"Layer.Inhib.Layer.Gi":     "2.1",
					"Layer.Inhib.Layer.FB":     "0.5",
					"Layer.Inhib.ActAvg.Init":  "0.4",
					"Layer.Inhib.ActAvg.Fixed": "true",
				}},
			{Sel: "#Hidden", Desc: "higher inhibition",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":     "3",
					"Layer.Inhib.ActAvg.Init":  "0.5",
					"Layer.Inhib.ActAvg.Fixed": "true",
				}},
			{Sel: "#Colors", Desc: "layer act avg",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.5",
				}},
			{Sel: "#Words", Desc: "layer act avg",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.5",
				}},
			{Sel: "#PFCToHidden", Desc: "PFC top-down projection",
				Params: params.Params{
					"Prjn.WtScale.Rel":        "0.3",
					"Prjn.Learn.Lrate":        "0.01", // even slower
					"Prjn.Learn.XCal.SetLLrn": "true",
					"Prjn.Learn.XCal.LLrn":    "0.1",
				}},
			{Sel: "#OutputToHidden", Desc: "Output top-down projection",
				Params: params.Params{
					"Prjn.WtScale.Rel":        "0.2",
					"Prjn.Learn.Lrate":        "0.04",
					"Prjn.Learn.XCal.SetLLrn": "true",
					"Prjn.Learn.XCal.LLrn":    "0.1",
				}},
			{Sel: "#HiddenToOutput", Desc: "to output",
				Params: params.Params{
					"Prjn.Learn.Lrate":        "0.08",
					"Prjn.Learn.XCal.SetLLrn": "true",
					"Prjn.Learn.XCal.LLrn":    "0.1",
				}},
		},
	}},
	{Name: "Training", Desc: "training parameters", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "faster time constant",
				Params: params.Params{
					"Layer.Act.Init.Decay": "1",
					"Layer.Act.Dt.VmTau":   "3.3",
				}},
		},
	}},
	{Name: "Testing", Desc: "testing parameters", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "slower time constant",
				Params: params.Params{
					"Layer.Act.Init.Decay": "1",
					"Layer.Act.Dt.VmTau":   "30",
				}},
		},
	}},
	{Name: "SOATesting", Desc: "SOA testing parameters", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "no decay",
				Params: params.Params{
					"Layer.Act.Init.Decay": "0",
				}},
		},
	}},
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	// strength of projection from PFC to Hidden -- reduce to simulate PFC damage
	FmPFC float32 `def:"0.3" step:"0.01"`
	// time constant for updating the network
	DtVmTau float32 `def:"30" step:"5"`
	// the network -- click to view / edit parameters for layers, prjns, etc
	Net *leabra.Network `view:"no-inline"`
	// training patterns
	TrainPats *etable.Table `view:"no-inline"`
	// testing patterns
	TestPats *etable.Table `view:"no-inline"`
	// SOA testing patterns
	SOAPats *etable.Table `view:"no-inline"`
	// training epoch-level log data
	TrnEpcLog *etable.Table `view:"no-inline"`
	// testing epoch-level log data
	TstEpcLog *etable.Table `view:"no-inline"`
	// testing trial-level log data
	TstTrlLog *etable.Table `view:"no-inline"`
	// SOA testing trial-level log data
	SOATrlLog *etable.Table `view:"no-inline"`
	// summary log of each run
	RunLog *etable.Table `view:"no-inline"`
	// aggregate stats on all runs
	RunStats *etable.Table `view:"no-inline"`
	// full collection of param sets
	Params params.Sets `view:"no-inline"`
	// which set of *additional* parameters to use -- always applies Base and optionaly this next if set -- can use multiple names separated by spaces (don't put spaces in ParamSet names!)
	ParamSet string `view:"-"`
	// maximum number of model runs to perform
	MaxRuns int
	// maximum number of epochs to run per model run
	MaxEpcs int
	// Training environment -- contains everything about iterating over input / output patterns over training
	TrainEnv env.FreqTable
	// Testing environment for std strooop -- manages iterating over testing
	TestEnv env.FixedTable
	// Testing environment for SOA tests -- manages iterating over testing
	SOATestEnv env.FixedTable
	// leabra timing parameters and state
	Time leabra.Time
	// whether to update the network view while running
	ViewOn bool
	// at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model
	TrainUpdate leabra.TimeScales
	// at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model
	TestUpdate leabra.TimeScales
	// how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing
	TestInterval int
	// names of layers to record activations etc of during testing
	TstRecLays []string

	// 1 if trial was error, 0 if correct -- based on SSE = 0 (subject to .5 unit-wise tolerance)
	TrlErr float64 `inactive:"+"`
	// current trial's sum squared error
	TrlSSE float64 `inactive:"+"`
	// current trial's average sum squared error
	TrlAvgSSE float64 `inactive:"+"`
	// current trial's cosine difference
	TrlCosDiff float64 `inactive:"+"`
	// current SOA value
	SOA int `inactive:"+"`
	// current max cycles value for SOA
	SOAMaxCyc int `inactive:"+"`
	// current trial type for SOA
	SOATrlTyp int `inactive:"+"`
	// last epoch's total sum squared error
	EpcSSE float64 `inactive:"+"`
	// last epoch's average sum squared error (average over trials, and over units within layer)
	EpcAvgSSE float64 `inactive:"+"`
	// last epoch's percent of trials that had SSE > 0 (subject to .5 unit-wise tolerance)
	EpcPctErr float64 `inactive:"+"`
	// last epoch's percent of trials that had SSE == 0 (subject to .5 unit-wise tolerance)
	EpcPctCor float64 `inactive:"+"`
	// last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)
	EpcCosDiff float64 `inactive:"+"`

	// sum to increment as we go through epoch
	SumErr float64 `view:"-" inactive:"+"`
	// sum to increment as we go through epoch
	SumSSE float64 `view:"-" inactive:"+"`
	// sum to increment as we go through epoch
	SumAvgSSE float64 `view:"-" inactive:"+"`
	// sum to increment as we go through epoch
	SumCosDiff float64 `view:"-" inactive:"+"`
	// main GUI window
	Win *core.Window `view:"-"`
	// the network viewer
	NetView *netview.NetView `view:"-"`
	// the master toolbar
	ToolBar *core.ToolBar `view:"-"`
	// the training epoch plot
	TrnEpcPlot *eplot.Plot2D `view:"-"`
	// the testing epoch plot
	TstEpcPlot *eplot.Plot2D `view:"-"`
	// the test-trial plot
	TstTrlPlot *eplot.Plot2D `view:"-"`
	// the SOA test-trial plot
	SOATrlPlot *eplot.Plot2D `view:"-"`
	// the run plot
	RunPlot *eplot.Plot2D `view:"-"`
	// log file
	TrnEpcFile *os.File `view:"-"`
	// log file
	RunFile *os.File `view:"-"`
	// for holding layer values
	ValuesTsrs map[string]*etensor.Float32 `view:"-"`
	// true if sim is running
	IsRunning bool `view:"-"`
	// flag to stop running
	StopNow bool `view:"-"`
	// flag to initialize NewRun if last one finished
	NeedsNewRun bool `view:"-"`
	// the current random seed
	RndSeed int64 `view:"-"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Defaults()
	ss.Net = &leabra.Network{}
	ss.TrainPats = &etable.Table{}
	ss.TestPats = &etable.Table{}
	ss.SOAPats = &etable.Table{}
	ss.TrnEpcLog = &etable.Table{}
	ss.TstEpcLog = &etable.Table{}
	ss.TstTrlLog = &etable.Table{}
	ss.SOATrlLog = &etable.Table{}
	ss.RunLog = &etable.Table{}
	ss.RunStats = &etable.Table{}
	ss.Params = ParamSets
	ss.RndSeed = 1
	ss.ViewOn = true
	ss.TrainUpdate = leabra.Quarter
	ss.TestUpdate = leabra.Cycle
	ss.TestInterval = 5
	ss.TstRecLays = []string{"Colors", "Words", "PFC", "Hidden", "Output"}
}

func (ss *Sim) Defaults() {
	ss.FmPFC = 0.3
	ss.DtVmTau = 30
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.OpenPats()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigTrnEpcLog(ss.TrnEpcLog)
	ss.ConfigTstEpcLog(ss.TstEpcLog)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	ss.ConfigSOATrlLog(ss.SOATrlLog)
	ss.ConfigRunLog(ss.RunLog)
}

func (ss *Sim) ConfigEnv() {
	if ss.MaxRuns == 0 { // allow user override
		ss.MaxRuns = 1
	}
	if ss.MaxEpcs == 0 { // allow user override
		ss.MaxEpcs = 55
	}

	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	ss.TrainEnv.Table = etable.NewIndexView(ss.TrainPats)
	ss.TrainEnv.NSamples = 1
	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually

	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Dsc = "testing params and state"
	ss.TestEnv.Table = etable.NewIndexView(ss.TestPats)
	ss.TestEnv.Sequential = true
	ss.TestEnv.Validate()

	ss.SOATestEnv.Nm = "SOATestEnv"
	ss.SOATestEnv.Dsc = "test all params and state"
	ss.SOATestEnv.Table = etable.NewIndexView(ss.SOAPats)
	ss.SOATestEnv.Sequential = true
	ss.SOATestEnv.Validate()

	ss.TrainEnv.Init(0)
	ss.TestEnv.Init(0)
	ss.SOATestEnv.Init(0)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.InitName(net, "Stroop")
	clr := net.AddLayer2D("Colors", 1, 2, emer.Input)
	wrd := net.AddLayer2D("Words", 1, 2, emer.Input)
	hid := net.AddLayer4D("Hidden", 1, 2, 1, 2, emer.Hidden)
	pfc := net.AddLayer2D("PFC", 1, 2, emer.Input)
	out := net.AddLayer2D("Output", 1, 2, emer.Target)

	full := prjn.NewFull()
	clr2hid := prjn.NewOneToOne()
	wrd2hid := prjn.NewOneToOne()
	wrd2hid.RecvStart = 2

	pfc2hid := prjn.NewRect()
	pfc2hid.Scale.Set(0.5, 0.5)
	pfc2hid.Size.Set(1, 1)

	net.ConnectLayers(clr, hid, clr2hid, emer.Forward)
	net.ConnectLayers(wrd, hid, wrd2hid, emer.Forward)
	net.ConnectLayers(pfc, hid, pfc2hid, emer.Back)
	net.BidirConnectLayers(hid, out, full)

	wrd.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Colors", YAlign: relpos.Front, Space: 1})
	out.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Words", YAlign: relpos.Front, Space: 1})
	hid.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Colors", YAlign: relpos.Front, XAlign: relpos.Left, YOffset: 1})
	pfc.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Hidden", YAlign: relpos.Front, Space: 1})

	net.Defaults()
	ss.SetParams("Network", false) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.InitWts()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	rand.Seed(ss.RndSeed)
	ss.StopNow = false
	ss.SetParams("", false) // all sheets
	ss.NewRun()
	ss.UpdateView(true, -1)
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.RecordSyns()
	}
}

// NewRndSeed gets a new random seed based on current time -- otherwise uses
// the same random seed for every run
func (ss *Sim) NewRndSeed() {
	ss.RndSeed = time.Now().UnixNano()
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters(train bool) string {
	if train {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Trial.Cur, ss.Time.Cycle, ss.TrainEnv.TrialName.Cur)
	} else {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TestEnv.Trial.Cur, ss.Time.Cycle, ss.TestEnv.TrialName.Cur)
	}
}

func (ss *Sim) UpdateView(train bool, cyc int) {
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters(train), cyc)
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate() // note: using counters is significantly slower..
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// AlphaCyc runs one alpha-cycle (100 msec, 4 quarters)			 of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFmDWt calls are made.
// Handles netview updating within scope of AlphaCycle
func (ss *Sim) AlphaCyc(train bool) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdate := ss.TrainUpdate
	if !train {
		viewUpdate = ss.TestUpdate
	}

	ss.Net.AlphaCycInit(train)
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdate {
				case leabra.Cycle:
					if cyc != ss.Time.CycPerQtr-1 { // will be updated by quarter
						ss.UpdateView(train, ss.Time.Cycle)
					}
				case leabra.FastSpike:
					if (cyc+1)%10 == 0 {
						ss.UpdateView(train, -1)
					}
				}
			}
		}
		ss.Net.QuarterFinal(&ss.Time)
		ss.Time.QuarterInc()
		if ss.ViewOn {
			switch {
			case viewUpdate == leabra.Cycle:
				ss.UpdateView(train, ss.Time.Cycle)
			case viewUpdate <= leabra.Quarter:
				ss.UpdateView(train, -1)
			case viewUpdate == leabra.Phase:
				if qtr >= 2 {
					ss.UpdateView(train, -1)
				}
			}
		}
	}

	if train {
		ss.Net.DWt()
		if ss.NetView != nil && ss.NetView.IsVisible() {
			ss.NetView.RecordSyns()
		}
		ss.Net.WtFmDWt()
	}
	if ss.ViewOn && viewUpdate == leabra.AlphaCycle {
		ss.UpdateView(train, -1)
	}
}

// AlphaCycTest is for testing -- uses threshold stopping and longer quarters
func (ss *Sim) AlphaCycTest() {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdate := ss.TestUpdate
	train := false

	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()

	ss.Net.AlphaCycInit(train)
	ss.Time.AlphaCycStart()
	overThresh := false
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < 75; cyc++ { // note: fixed 75 per quarter = 200 total
			ss.Net.Cycle(&ss.Time)
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdate {
				case leabra.Cycle:
					ss.UpdateView(train, ss.Time.Cycle)
				case leabra.FastSpike:
					if (cyc+1)%10 == 0 {
						ss.UpdateView(train, -1)
					}
				}
			}
			outact := out.Pools[0].Inhib.Act.Max
			if outact > 0.51 {
				overThresh = true
				break
			}
		}
		ss.Net.QuarterFinal(&ss.Time)
		ss.Time.QuarterInc()
		if ss.ViewOn {
			switch {
			case viewUpdate == leabra.Cycle:
				ss.UpdateView(train, ss.Time.Cycle)
			case viewUpdate <= leabra.Quarter:
				ss.UpdateView(train, -1)
			case viewUpdate == leabra.Phase:
				if qtr >= 2 {
					ss.UpdateView(train, -1)
				}
			}
		}
		if overThresh {
			break
		}
	}

	ss.UpdateView(false, -1)
}

// AlphaCycTestCyc test with specified number of cycles
func (ss *Sim) AlphaCycTestCyc(cycs int) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdate := ss.TestUpdate
	train := false

	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()

	ss.Net.AlphaCycInit(train)
	ss.Time.AlphaCycStart()
	for cyc := 0; cyc < cycs; cyc++ { // just fixed cycles, no quarters
		ss.Net.Cycle(&ss.Time)
		ss.Time.CycleInc()
		if ss.ViewOn {
			switch viewUpdate {
			case leabra.Cycle:
				ss.UpdateView(train, ss.Time.Cycle)
			case leabra.FastSpike:
				if (cyc+1)%10 == 0 {
					ss.UpdateView(train, -1)
				}
			}
		}
		outact := out.Pools[0].Inhib.Act.Max
		if cycs > 100 && outact > 0.51 { // only for long trials
			break
		}
	}
	ss.Net.QuarterFinal(&ss.Time)
	ss.Time.QuarterInc()
	if ss.ViewOn {
		switch {
		case viewUpdate == leabra.Cycle:
			ss.UpdateView(train, ss.Time.Cycle)
		case viewUpdate <= leabra.Quarter:
			ss.UpdateView(train, -1)
		case viewUpdate == leabra.Phase:
			ss.UpdateView(train, -1)
		}
	}

	ss.UpdateView(false, -1)
}

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"Colors", "Words", "Output", "PFC"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := en.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// TrainTrial runs one trial of training using TrainEnv
func (ss *Sim) TrainTrial() {
	if ss.NeedsNewRun {
		ss.NewRun()
	}

	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.LogTrnEpc(ss.TrnEpcLog)
		if ss.ViewOn && ss.TrainUpdate > leabra.AlphaCycle {
			ss.UpdateView(true, -1)
		}
		if ss.TestInterval > 0 && epc%ss.TestInterval == 0 { // note: epc is *next* so won't trigger first time
			ss.TestAll()
		}
		if epc >= ss.MaxEpcs {
			// done with training..
			ss.RunEnd()
			if ss.TrainEnv.Run.Incr() { // we are done!
				ss.StopNow = true
				return
			} else {
				ss.NeedsNewRun = true
				return
			}
		}
	}

	ss.SetParamsSet("Training", "Network", false)
	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()
	out.SetType(emer.Target)

	ss.ApplyInputs(&ss.TrainEnv)
	ss.AlphaCyc(true)   // train
	ss.TrialStats(true) // accumulate
}

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
	ss.LogRun(ss.RunLog)
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	run := ss.TrainEnv.Run.Cur
	ss.TrainEnv.Init(run)
	ss.TestEnv.Init(run)
	ss.SOATestEnv.Init(run)
	ss.Time.Reset()
	ss.Net.InitWts()
	ss.InitStats()
	ss.TrnEpcLog.SetNumRows(0)
	ss.TstEpcLog.SetNumRows(0)
	ss.NeedsNewRun = false
}

// InitStats initializes all the statistics, especially important for the
// cumulative epoch stats -- called at start of new run
func (ss *Sim) InitStats() {
	// accumulators
	ss.SumErr = 0
	ss.SumSSE = 0
	ss.SumAvgSSE = 0
	ss.SumCosDiff = 0
	// clear rest just to make Sim look initialized
	ss.TrlErr = 0
	ss.TrlSSE = 0
	ss.TrlAvgSSE = 0
	ss.EpcSSE = 0
	ss.EpcAvgSSE = 0
	ss.EpcPctErr = 0
	ss.EpcCosDiff = 0
}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func (ss *Sim) TrialStats(accum bool) (sse, avgsse, cosdiff float64) {
	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()
	ss.TrlCosDiff = float64(out.CosDiff.Cos)
	ss.TrlSSE, ss.TrlAvgSSE = out.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
	if ss.TrlSSE > 0 {
		ss.TrlErr = 1
	} else {
		ss.TrlErr = 0
	}
	if accum {
		ss.SumErr += ss.TrlErr
		ss.SumSSE += ss.TrlSSE
		ss.SumAvgSSE += ss.TrlAvgSSE
		ss.SumCosDiff += ss.TrlCosDiff
	}
	return
}

// TrainEpoch runs training trials for remainder of this epoch
func (ss *Sim) TrainEpoch() {
	ss.StopNow = false
	curEpc := ss.TrainEnv.Epoch.Cur
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Epoch.Cur != curEpc {
			break
		}
	}
	ss.Stopped()
}

// TrainRun runs training trials for remainder of run
func (ss *Sim) TrainRun() {
	ss.StopNow = false
	curRun := ss.TrainEnv.Run.Cur
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Run.Cur != curRun {
			break
		}
	}
	ss.Stopped()
}

// Train runs the full training from this point onward
func (ss *Sim) Train() {
	ss.StopNow = false
	for {
		ss.TrainTrial()
		if ss.StopNow {
			break
		}
	}
	ss.Stopped()
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.StopNow = true
}

// Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
func (ss *Sim) Stopped() {
	ss.IsRunning = false
	if ss.Win != nil {
		vp := ss.Win.WinViewport2D()
		if ss.ToolBar != nil {
			ss.ToolBar.UpdateActions()
		}
		vp.SetNeedsFullRender()
	}
}

// SaveWeights saves the network weights -- when called with views.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWeights(filename core.FileName) {
	ss.Net.SaveWtsJSON(filename)
}

////////////////////////////////////////////////////////////////////////////////////////////
// Generalization Testing

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial(returnOnChg bool) {
	ss.TestEnv.Step()

	// Query counters FIRST
	_, _, chg := ss.TestEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewOn && ss.TestUpdate > leabra.AlphaCycle {
			ss.UpdateView(false, -1)
		}
		ss.LogTstEpc(ss.TstEpcLog)
		if returnOnChg {
			return
		}
	}

	ss.SetParamsSet("Testing", "Network", false)
	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()
	out.SetType(emer.Compare)

	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCycTest()
	ss.TrialStats(false) // !accumulate
	ss.LogTstTrl(ss.TstTrlLog, ss.TestEnv.Trial.Cur, ss.TestEnv.TrialName.Cur)
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	// note: this has no learning calls
	ss.SetParamsSet("Testing", "Network", false)
	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
	for {
		ss.TestTrial(true) // return on chg, don't present
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}
}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.StopNow = false
	ss.TestAll()
	ss.Stopped()
}

////////////////////////////////////////////////////////////////////////////////////////////
// SOATest

// SOATestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) SOATestTrial(returnOnChg bool) {
	ss.SOATestEnv.Step()

	// Query counters FIRST
	_, _, chg := ss.SOATestEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewOn && ss.TestUpdate > leabra.AlphaCycle {
			ss.UpdateView(false, -1)
		}
		if returnOnChg {
			return
		}
	}

	trl := ss.SOATestEnv.Trial.Cur
	ss.SOA = int(ss.SOAPats.CellFloat("SOA", trl))
	ss.SOAMaxCyc = int(ss.SOAPats.CellFloat("MaxCycles", trl))
	ss.SOATrlTyp = int(ss.SOAPats.CellFloat("TrialType", trl))

	ss.SetParamsSet("Testing", "Network", false)
	ss.SetParamsSet("SOATesting", "Network", false)
	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()
	out.SetType(emer.Compare)

	islate := strings.Contains(ss.SOATestEnv.TrialName.Cur, "latestim")
	if !islate || ss.SOA == 0 {
		ss.Net.InitActs()
	}
	ss.ApplyInputs(&ss.SOATestEnv)
	ss.AlphaCycTestCyc(ss.SOAMaxCyc)
	if strings.Contains(ss.SOATestEnv.TrialName.Cur, "latestim") {
		ss.TrialStats(false) // !accumulate
		ss.LogSOATrl(ss.SOATrlLog, ss.SOATestEnv.Trial.Cur)
	}
}

// SOATestAll runs through the full set of testing items
func (ss *Sim) SOATestAll() {
	ss.SOATestEnv.Init(ss.TrainEnv.Run.Cur)
	ss.SOATrlLog.SetNumRows(0)
	for {
		ss.SOATestTrial(true) // return on chg, don't present
		_, _, chg := ss.SOATestEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}
}

// RunSOATestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunSOATestAll() {
	ss.StopNow = false
	ss.SOATestAll()
	ss.Stopped()
}

/////////////////////////////////////////////////////////////////////////
//   Params setting

// SetParams sets the params for "Base" and then current ParamSet.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParams(sheet string, setMsg bool) error {
	if sheet == "" {
		// this is important for catching typos and ensuring that all sheets can be used
		ss.Params.ValidateSheets([]string{"Network", "Sim"})
	}
	err := ss.SetParamsSet("Base", sheet, setMsg)
	if ss.ParamSet != "" && ss.ParamSet != "Base" {
		sps := strings.Fields(ss.ParamSet)
		for _, ps := range sps {
			err = ss.SetParamsSet(ps, sheet, setMsg)
		}
	}

	return err
}

// SetParamsSet sets the params for given params.Set name.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParamsSet(setNm string, sheet string, setMsg bool) error {
	pset, err := ss.Params.SetByNameTry(setNm)
	if err != nil {
		return err
	}
	if sheet == "" || sheet == "Network" {

		spo := ss.Params.SetByName("Testing").SheetByName("Network").SelByName("Layer")
		spo.Params.SetByName("Layer.Act.Dt.VmTau", fmt.Sprintf("%g", ss.DtVmTau))

		netp, ok := pset.Sheets["Network"]
		if ok {
			ss.Net.ApplyParams(netp, setMsg)
		}
		hid := ss.Net.LayerByName("Hidden").(leabra.LeabraLayer).AsLeabra()
		fmpfc := hid.SendName("PFC").(leabra.LeabraPrjn).AsLeabra()
		fmpfc.WtScale.Rel = ss.FmPFC
	}

	if sheet == "" || sheet == "Sim" {
		simp, ok := pset.Sheets["Sim"]
		if ok {
			simp.Apply(ss, setMsg)
		}
	}
	// note: if you have more complex environments with parameters, definitely add
	// sheets for them, e.g., "TrainEnv", "TestEnv" etc
	return err
}

// OpenPatAsset opens pattern file from embedded assets
func (ss *Sim) OpenPatAsset(dt *etable.Table, fnm, name, desc string) error {
	dt.SetMetaData("name", name)
	dt.SetMetaData("desc", desc)
	err := dt.OpenFS(content, fnm, etable.Tab)
	if errors.Log(err) == nil {
		for i := 1; i < len(dt.Cols); i++ {
			dt.Cols[i].SetMetaData("grid-fill", "0.9")
		}
	}
	return err
}

func (ss *Sim) OpenPats() {
	// patgen.ReshapeCppFile(ss.TrainPats, "stroop_train.dat", "stroop_train.tsv") // one-time reshape
	// patgen.ReshapeCppFile(ss.TestPats, "stroop_test.dat", "stroop_test.tsv")    // one-time reshape
	// patgen.ReshapeCppFile(ss.SOAPats, "stroop_soa.dat", "stroop_soa.tsv")       // one-time reshape
	ss.OpenPatAsset(ss.TrainPats, "stroop_train.tsv", "Stroop Train", "Stroop Training patterns")
	ss.OpenPatAsset(ss.TestPats, "stroop_test.tsv", "Stroop Test", "Stroop Testing patterns")
	ss.OpenPatAsset(ss.SOAPats, "stroop_soa.tsv", "Stroop SOA", "Stroop SOA Testing patterns")
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

// ValuesTsr gets value tensor of given name, creating if not yet made
func (ss *Sim) ValuesTsr(name string) *etensor.Float32 {
	if ss.ValuesTsrs == nil {
		ss.ValuesTsrs = make(map[string]*etensor.Float32)
	}
	tsr, ok := ss.ValuesTsrs[name]
	if !ok {
		tsr = &etensor.Float32{}
		ss.ValuesTsrs[name] = tsr
	}
	return tsr
}

//////////////////////////////////////////////
//  TrnEpcLog

// LogTrnEpc adds data from current epoch to the TrnEpcLog table.
// computes epoch averages prior to logging.
func (ss *Sim) LogTrnEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epc := ss.TrainEnv.Epoch.Prv          // this is triggered by increment so use previous value
	nt := float64(len(ss.TrainEnv.Order)) // number of trials in view

	ss.EpcSSE = ss.SumSSE / nt
	ss.SumSSE = 0
	ss.EpcAvgSSE = ss.SumAvgSSE / nt
	ss.SumAvgSSE = 0
	ss.EpcPctErr = float64(ss.SumErr) / nt
	ss.SumErr = 0
	ss.EpcPctCor = 1 - ss.EpcPctErr
	ss.EpcCosDiff = ss.SumCosDiff / nt
	ss.SumCosDiff = 0

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("SSE", row, ss.EpcSSE)
	dt.SetCellFloat("AvgSSE", row, ss.EpcAvgSSE)
	dt.SetCellFloat("PctErr", row, ss.EpcPctErr)
	dt.SetCellFloat("PctCor", row, ss.EpcPctCor)
	dt.SetCellFloat("CosDiff", row, ss.EpcCosDiff)

	// note: essential to use Go version of update when called from another goroutine
	ss.TrnEpcPlot.GoUpdate()
	if ss.TrnEpcFile != nil {
		if ss.TrainEnv.Run.Cur == 0 && epc == 0 {
			dt.WriteCSVHeaders(ss.TrnEpcFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.TrnEpcFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigTrnEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnEpcLog")
	dt.SetMetaData("desc", "Record of performance over epochs of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Stroop Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0) // default plot
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	return plt
}

//////////////////////////////////////////////
//  TstTrlLog

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *etable.Table, trl int, trlnm string) {
	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value

	row := dt.Rows
	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Trial", row, float64(trl%3))
	dt.SetCellString("TrialName", row, trlnm)
	dt.SetCellFloat("Cycle", row, float64(ss.Time.Cycle))
	dt.SetCellFloat("Err", row, ss.TrlErr)
	dt.SetCellFloat("SSE", row, ss.TrlSSE)
	dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	for _, lnm := range ss.TstRecLays {
		tsr := ss.ValuesTsr(lnm)
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		ly.UnitValuesTensor(tsr, "ActM") // get minus phase act
		dt.SetCellTensor(lnm, row, tsr)
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TstTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTstTrlLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TestEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"Cycle", etensor.INT64, nil, nil},
		{"Err", etensor.FLOAT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.TstRecLays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		sch = append(sch, etable.Column{lnm, etensor.FLOAT64, ly.Shp.Shp, nil})
	}
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Stroop Test Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	plt.Params.Points = true
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Cycle", eplot.On, eplot.FixMin, 0, eplot.FixMax, 250) // default plot
	plt.SetColParams("Err", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	for _, lnm := range ss.TstRecLays {
		cp := plt.SetColParams(lnm, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
		cp.TensorIndex = -1 // plot all
	}
	return plt
}

//////////////////////////////////////////////
//  SOATrlLog

// LogSOATrl adds data from current trial to the SOATrlLog table.
// log always contains number of testing items
func (ss *Sim) LogSOATrl(dt *etable.Table, trl int) {
	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value

	row := dt.Rows
	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	conds := []string{"Color_Conf", "Color_Cong", "Word_Conf", "Word_Cong"}

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Trial", row, float64(ss.SOATrlTyp))
	dt.SetCellFloat("SOA", row, float64(ss.SOA))
	dt.SetCellString("TrialName", row, conds[ss.SOATrlTyp])
	dt.SetCellFloat("Cycle", row, float64(ss.Time.Cycle))
	dt.SetCellFloat("Err", row, ss.TrlErr)
	dt.SetCellFloat("SSE", row, ss.TrlSSE)
	dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	for _, lnm := range ss.TstRecLays {
		tsr := ss.ValuesTsr(lnm)
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		ly.UnitValuesTensor(tsr, "ActM") // get minus phase act
		dt.SetCellTensor(lnm, row, tsr)
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.SOATrlPlot.GoUpdate()
}

func (ss *Sim) ConfigSOATrlLog(dt *etable.Table) {
	dt.SetMetaData("name", "SOATrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.SOATestEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"SOA", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"Cycle", etensor.INT64, nil, nil},
		{"Err", etensor.FLOAT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.TstRecLays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		sch = append(sch, etable.Column{lnm, etensor.FLOAT64, ly.Shp.Shp, nil})
	}
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigSOATrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Stroop SOA Test Trial Plot"
	plt.Params.XAxisCol = "SOA"
	plt.Params.LegendCol = "TrialName"
	plt.SetTable(dt)
	plt.Params.Points = true
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SOA", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Cycle", eplot.On, eplot.FixMin, 0, eplot.FixMax, 220) // default plot
	plt.SetColParams("Err", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	for _, lnm := range ss.TstRecLays {
		cp := plt.SetColParams(lnm, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
		cp.TensorIndex = -1 // plot all
	}
	return plt
}

//////////////////////////////////////////////
//  TstEpcLog

func (ss *Sim) LogTstEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	trl := ss.TstTrlLog
	tix := etable.NewIndexView(trl)
	epc := ss.TrainEnv.Epoch.Prv // ?

	// note: this shows how to use agg methods to compute summary data from another
	// data table, instead of incrementing on the Sim
	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("SSE", row, agg.Sum(tix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, agg.Mean(tix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, agg.Mean(tix, "Err")[0])
	dt.SetCellFloat("PctCor", row, 1-agg.Mean(tix, "Err")[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(tix, "CosDiff")[0])

	// note: essential to use Go version of update when called from another goroutine
	ss.TstEpcPlot.GoUpdate()
}

func (ss *Sim) ConfigTstEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstEpcLog")
	dt.SetMetaData("desc", "Summary stats for testing trials")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Stroop Testing Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SSE", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0) // default plot
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	return plt
}

//////////////////////////////////////////////
//  RunLog

// LogRun adds data from current run to the RunLog table.
func (ss *Sim) LogRun(dt *etable.Table) {
	run := ss.TrainEnv.Run.Cur // this is NOT triggered by increment yet -- use Cur
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epclog := ss.TrnEpcLog
	epcix := etable.NewIndexView(epclog)
	// compute mean over last N epochs for run level
	nlast := 5
	if nlast > epcix.Len()-1 {
		nlast = epcix.Len() - 1
	}
	epcix.Indexes = epcix.Indexes[epcix.Len()-nlast:]

	params := ""

	dt.SetCellFloat("Run", row, float64(run))
	dt.SetCellString("Params", row, params)
	dt.SetCellFloat("SSE", row, agg.Mean(epcix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, agg.Mean(epcix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, agg.Mean(epcix, "PctErr")[0])
	dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])

	runix := etable.NewIndexView(dt)
	spl := split.GroupBy(runix, []string{"Params"})
	split.Desc(spl, "PctCor")
	ss.RunStats = spl.AggsToTable(etable.AddAggName)

	// note: essential to use Go version of update when called from another goroutine
	ss.RunPlot.GoUpdate()
	if ss.RunFile != nil {
		if row == 0 {
			dt.WriteCSVHeaders(ss.RunFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.RunFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigRunLog(dt *etable.Table) {
	dt.SetMetaData("name", "RunLog")
	dt.SetMetaData("desc", "Record of performance at end of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Params", etensor.STRING, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigRunPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Stroop Run Plot"
	plt.Params.XAxisCol = "Run"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SSE", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.ViewDefaults()
	nv.Scene().Camera.Pose.Pos.Set(0.1, 1.8, 3.5)
	nv.Scene().Camera.LookAt(math32.Vec3{0.1, 0.15, 0}, math32.Vec3{0, 1, 0})

	labs := []string{"     g      r", "   G       R", "  gr      rd", "     g      r         G      R", "  cn     wr"}
	nv.ConfigLabels(labs)

	lays := []string{"Colors", "Words", "Output", "Hidden", "PFC"}

	for li, lnm := range lays {
		ly := nv.LayerByName(lnm)
		lbl := nv.LabelByName(labs[li])
		lbl.Pose = ly.Pose
		lbl.Pose.Pos.Y += .2
		lbl.Pose.Pos.Z += .02
		lbl.Pose.Scale.SetMul(math32.Vec3{0.4, 0.06, 0.5})
	}
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *core.Window {
	width := 1600
	height := 1200

	core.SetAppName("stroop")
	core.SetAppAbout(`illustrates how the PFC can produce top-down biasing for executive control, in the context of the widely-studied Stroop task. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch10/stroop/README.md">README.md on GitHub</a>.</p>`)

	win := core.NewMainWindow("stroop", "Stroop", width, height)
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := core.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := core.AddNewSplitView(mfr, "split")
	split.Dim = math32.X
	split.SetStretchMax()

	sv := views.AddNewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := core.AddNewTabView(split, "tv")

	nv := tv.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	nv.Var = "Act"
	nv.SetNet(ss.Net)
	ss.NetView = nv
	ss.ConfigNetView(nv)

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TrnEpcPlot").(*eplot.Plot2D)
	ss.TrnEpcPlot = ss.ConfigTrnEpcPlot(plt, ss.TrnEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstTrlPlot").(*eplot.Plot2D)
	ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "SOATrlPlot").(*eplot.Plot2D)
	ss.SOATrlPlot = ss.ConfigSOATrlPlot(plt, ss.SOATrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstEpcPlot").(*eplot.Plot2D)
	ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.TstEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "RunPlot").(*eplot.Plot2D)
	ss.RunPlot = ss.ConfigRunPlot(plt, ss.RunLog)

	split.SetSplits(.2, .8)

	tbar.AddAction(core.ActOpts{Label: "Init", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(!ss.IsRunning)
	}}, win.This(), func(recv, send tree.Ki, sig int64, data interface{}) {
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(core.ActOpts{Label: "Train", Icon: "run", Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.",
		UpdateFunc: func(act *core.Action) {
			act.SetActiveStateUpdate(!ss.IsRunning)
		}}, win.This(), func(recv, send tree.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			// ss.Train()
			go ss.Train()
		}
	})

	tbar.AddAction(core.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(ss.IsRunning)
	}}, win.This(), func(recv, send tree.Ki, sig int64, data interface{}) {
		ss.Stop()
	})

	tbar.AddAction(core.ActOpts{Label: "Step Trial", Icon: "step-fwd", Tooltip: "Advances one training trial at a time.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(!ss.IsRunning)
	}}, win.This(), func(recv, send tree.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TrainTrial()
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(core.ActOpts{Label: "Step Epoch", Icon: "fast-fwd", Tooltip: "Advances one epoch (complete set of training patterns) at a time.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(!ss.IsRunning)
	}}, win.This(), func(recv, send tree.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainEpoch()
		}
	})

	tbar.AddAction(core.ActOpts{Label: "Step Run", Icon: "fast-fwd", Tooltip: "Advances one full training Run at a time.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(!ss.IsRunning)
	}}, win.This(), func(recv, send tree.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainRun()
		}
	})

	tbar.AddSeparator("test")

	tbar.AddAction(core.ActOpts{Label: "Test Trial", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(!ss.IsRunning)
	}}, win.This(), func(recv, send tree.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TestTrial(false) // don't break on chg
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(core.ActOpts{Label: "Test All", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(!ss.IsRunning)
	}}, win.This(), func(recv, send tree.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestAll()
		}
	})

	tbar.AddSeparator("soa")

	tbar.AddAction(core.ActOpts{Label: "SOA Test Trial", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(!ss.IsRunning)
	}}, win.This(), func(recv, send tree.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.SOATestTrial(false) // don't break on chg
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(core.ActOpts{Label: "SOA Test All", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(!ss.IsRunning)
	}}, win.This(), func(recv, send tree.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunSOATestAll()
		}
	})

	tbar.AddSeparator("misc")

	tbar.AddAction(core.ActOpts{Label: "Reset TstTrlLog", Icon: "reset", Tooltip: "Reset the test trial log -- otherwise it accumulates to compare across parameters etc."}, win.This(),
		func(recv, send tree.Ki, sig int64, data interface{}) {
			ss.TstTrlLog.SetNumRows(0)
			ss.TstTrlPlot.Update()
		})

	tbar.AddAction(core.ActOpts{Label: "New Seed", Icon: "new", Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."}, win.This(),
		func(recv, send tree.Ki, sig int64, data interface{}) {
			ss.NewRndSeed()
		})

	tbar.AddAction(core.ActOpts{Label: "Defaults", Icon: "update", Tooltip: "Restore initial default parameters.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(!ss.IsRunning)
	}}, win.This(), func(recv, send tree.Ki, sig int64, data interface{}) {
		ss.Defaults()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(core.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send tree.Ki, sig int64, data interface{}) {
			core.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch10/stroop/README.md")
		})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := core.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*core.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*core.Action)
	emen.Menu.AddCopyCutPaste(win)

	// note: Command in shortcuts is automatically translated into Control for
	// Linux, Windows or Meta for MacOS
	// fmen := win.MainMenu.ChildByName("File", 0).(*core.Action)
	// fmen.Menu.AddAction(core.ActOpts{Label: "Open", Shortcut: "Command+O"},
	// 	win.This(), func(recv, send tree.Ki, sig int64, data interface{}) {
	// 		FileViewOpenSVG(vp)
	// 	})
	// fmen.Menu.AddSeparator("csep")
	// fmen.Menu.AddAction(core.ActOpts{Label: "Close Window", Shortcut: "Command+W"},
	// 	win.This(), func(recv, send tree.Ki, sig int64, data interface{}) {
	// 		win.Close()
	// 	})

	inQuitPrompt := false
	core.SetQuitReqFunc(func() {
		if inQuitPrompt {
			return
		}
		inQuitPrompt = true
		core.PromptDialog(vp, core.DlgOpts{Title: "Really Quit?",
			Prompt: "Are you <i>sure</i> you want to quit and lose any unsaved params, weights, logs, etc?"}, core.AddOk, core.AddCancel,
			win.This(), func(recv, send tree.Ki, sig int64, data interface{}) {
				if sig == int64(core.DialogAccepted) {
					core.Quit()
				} else {
					inQuitPrompt = false
				}
			})
	})

	// core.SetQuitCleanFunc(func() {
	// 	fmt.Printf("Doing final Quit cleanup here..\n")
	// })

	inClosePrompt := false
	win.SetCloseReqFunc(func(w *core.Window) {
		if inClosePrompt {
			return
		}
		inClosePrompt = true
		core.PromptDialog(vp, core.DlgOpts{Title: "Really Close Window?",
			Prompt: "Are you <i>sure</i> you want to close the window?  This will Quit the App as well, losing all unsaved params, weights, logs, etc"}, core.AddOk, core.AddCancel,
			win.This(), func(recv, send tree.Ki, sig int64, data interface{}) {
				if sig == int64(core.DialogAccepted) {
					core.Quit()
				} else {
					inClosePrompt = false
				}
			})
	})

	win.SetCloseCleanFunc(func(w *core.Window) {
		go core.Quit() // once main window is closed, quit
	})

	win.MainMenuUpdated()
	return win
}

// These props register Save methods so they can be used
var SimProps = tree.Props{
	"CallMethods": tree.PropSlice{
		{"SaveWeights", tree.Props{
			"desc": "save network weights to file",
			"icon": "file-save",
			"Args": tree.PropSlice{
				{"File Name", tree.Props{
					"ext": ".wts,.wts.gz",
				}},
			},
		}},
	},
}

func mainrun() {
	TheSim.New()
	TheSim.Config()

	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}
