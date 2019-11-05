// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
abac explores the classic paired associates learning task in a
cortical-like network, which exhibits catastrophic levels of
interference.
*/
package main

import (
	"bytes"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/clust"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/metric"
	"github.com/emer/etable/pca"
	"github.com/emer/etable/simat"
	"github.com/emer/etable/split"
	"github.com/emer/leabra/leabra"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/gi/mat32"
	"github.com/goki/ki/ki"
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

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "fixed LLrn",
				Params: params.Params{
					"Prjn.Learn.Norm.On":      "false",
					"Prjn.Learn.Momentum.On":  "false",
					"Prjn.Learn.WtBal.On":     "true",
					"Prjn.WtInit.Var":         "0.25",
					"Prjn.Learn.XCal.SetLLrn": "true",
					"Prjn.Learn.XCal.LLrn":    "0.0003",
					"Prjn.Learn.Lrate":        "0.04",
				}},
			{Sel: "Layer", Desc: "Default learning, inhib params",
				Params: params.Params{
					"Layer.Learn.AvgL.Gain": "2.5",
					"Layer.Inhib.Layer.Gi":  "1.6", // output layer does better with this!
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.3", // note: this was missing in C++ version! makes this work much better
				}},
		},
	}},
}

// Reps contains standard analysis of representations
type Reps struct {
	SimMat    *simat.SimMat `view:"no-inline" desc:"similarity matrix"`
	PCAPlot   *eplot.Plot2D `view:"no-inline" desc:"plot of pca data"`
	ClustPlot *eplot.Plot2D `view:"no-inline" desc:"cluster plot"`
	PCA       *pca.PCA      `view:"-" desc:"pca results"`
	PCAPrjn   *etable.Table `view:"-" desc:"pca projections onto eigenvectors"`
}

func (rp *Reps) Init() {
	rp.SimMat = &simat.SimMat{}
	rp.SimMat.Init()
	rp.PCA = &pca.PCA{}
	rp.PCA.Init()
	rp.PCAPrjn = &etable.Table{}
	rp.PCAPlot = &eplot.Plot2D{}
	rp.PCAPlot.InitName(rp.PCAPlot, "PCAPlot") // any Ki obj needs this
	rp.ClustPlot = &eplot.Plot2D{}
	rp.ClustPlot.InitName(rp.ClustPlot, "ClustPlot") // any Ki obj needs this
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	HiddenInhibGi float32           `def:"1.8" desc:"hidden layer inhibition -- increase to make sparser"`
	WtInitVar     float32           `def:"0.25" desc:"random initial weight variance -- increase to make more random"`
	FmContext     float32           `def:"1" desc:"relative WtScale.Rel from Context layer"`
	XCalLLrn      float32           `def:"0.0003" desc:"amount of Hebbian BCM learning based on AvgL long-term average activity -- increase to increase amount of hebbian"`
	Lrate         float32           `def:"0.04" desc:"learning rate"`
	Net           *leabra.Network   `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	ABPats        *etable.Table     `view:"no-inline" desc:"AB paired associate training patterns"`
	ACPats        *etable.Table     `view:"no-inline" desc:"AC paired associate training patterns"`
	TrnEpcLog     *etable.Table     `view:"no-inline" desc:"training epoch-level log data"`
	TstEpcLog     *etable.Table     `view:"no-inline" desc:"testing epoch-level log data"`
	TstTrlLog     *etable.Table     `view:"no-inline" desc:"testing trial-level log data"`
	RunLog        *etable.Table     `view:"no-inline" desc:"summary log of each run"`
	RunStats      *etable.Table     `view:"no-inline" desc:"aggregate stats on all runs"`
	TstStats      *etable.Table     `view:"no-inline" desc:"testing stats"`
	Params        params.Sets       `view:"no-inline" desc:"full collection of param sets"`
	MaxRuns       int               `desc:"maximum number of model runs to perform"`
	MaxEpcs       int               `desc:"maximum number of epochs to run per model run"`
	NZeroStop     int               `desc:"if a positive number, training will stop after this many epochs with zero SSE"`
	TrainEnv      env.FixedTable    `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	TestEnv       env.FixedTable    `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	Time          leabra.Time       `desc:"leabra timing parameters and state"`
	ViewOn        bool              `desc:"whether to update the network view while running"`
	TrainUpdt     leabra.TimeScales `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt      leabra.TimeScales `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	TestInterval  int               `desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`
	TstRecLays    []string          `desc:"names of layers to record activations etc of during testing"`
	HiddenReps    Reps              `view:"inline" desc:"representational analysis of Hidden layer, sorted by relationship"`

	// statistics: note use float64 as that is best for etable.Table
	TestNm     string  `inactive:"+" desc:"what set of patterns are we currently testing"`
	TrlErr     float64 `inactive:"+" desc:"1 if trial was error, 0 if correct -- based on SSE = 0 (subject to .5 unit-wise tolerance)"`
	TrlSSE     float64 `inactive:"+" desc:"current trial's sum squared error"`
	TrlAvgSSE  float64 `inactive:"+" desc:"current trial's average sum squared error"`
	TrlCosDiff float64 `inactive:"+" desc:"current trial's cosine difference"`
	EpcSSE     float64 `inactive:"+" desc:"last epoch's total sum squared error"`
	EpcAvgSSE  float64 `inactive:"+" desc:"last epoch's average sum squared error (average over trials, and over units within layer)"`
	EpcPctErr  float64 `inactive:"+" desc:"last epoch's average TrlErr"`
	EpcPctCor  float64 `inactive:"+" desc:"1 - last epoch's average TrlErr"`
	EpcCosDiff float64 `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	FirstZero  int     `inactive:"+" desc:"epoch at when SSE first went to zero"`
	NZero      int     `inactive:"+" desc:"number of epochs in a row with zero SSE"`

	// internal state - view:"-"
	SumErr      float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumSSE      float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumAvgSSE   float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumCosDiff  float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	Win         *gi.Window                  `view:"-" desc:"main GUI window"`
	NetView     *netview.NetView            `view:"-" desc:"the network viewer"`
	ToolBar     *gi.ToolBar                 `view:"-" desc:"the master toolbar"`
	TrnEpcPlot  *eplot.Plot2D               `view:"-" desc:"the training epoch plot"`
	TstEpcPlot  *eplot.Plot2D               `view:"-" desc:"the testing epoch plot"`
	TstTrlPlot  *eplot.Plot2D               `view:"-" desc:"the test-trial plot"`
	RunPlot     *eplot.Plot2D               `view:"-" desc:"the run plot"`
	TrnEpcFile  *os.File                    `view:"-" desc:"log file"`
	RunFile     *os.File                    `view:"-" desc:"log file"`
	TstNms      []string                    `view:"-" desc:"names of test tables"`
	TstStatNms  []string                    `view:"-" desc:"names of test stats"`
	ValsTsrs    map[string]*etensor.Float32 `view:"-" desc:"for holding layer values"`
	IsRunning   bool                        `view:"-" desc:"true if sim is running"`
	StopNow     bool                        `view:"-" desc:"flag to stop running"`
	NeedsNewRun bool                        `view:"-" desc:"flag to initialize NewRun if last one finished"`
	RndSeed     int64                       `view:"-" desc:"the current random seed"`
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
	ss.ABPats = &etable.Table{}
	ss.ACPats = &etable.Table{}
	ss.TrnEpcLog = &etable.Table{}
	ss.TstEpcLog = &etable.Table{}
	ss.TstTrlLog = &etable.Table{}
	ss.RunLog = &etable.Table{}
	ss.RunStats = &etable.Table{}
	ss.Params = ParamSets
	ss.RndSeed = 1
	ss.ViewOn = true
	ss.TrainUpdt = leabra.Quarter
	ss.TestUpdt = leabra.Quarter
	ss.TestInterval = 1
	ss.TstRecLays = []string{"Hidden"}
	ss.TstNms = []string{"AB", "AC"}
	ss.TstStatNms = []string{"Err"}
	ss.HiddenReps.Init()
}

func (ss *Sim) Defaults() {
	ss.HiddenInhibGi = 1.8
	ss.WtInitVar = 0.25
	ss.FmContext = 1
	ss.XCalLLrn = 0.0003
	ss.Lrate = 0.04
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
	ss.ConfigRunLog(ss.RunLog)
}

func (ss *Sim) ConfigEnv() {
	if ss.MaxRuns == 0 { // allow user override
		ss.MaxRuns = 10
	}
	if ss.MaxEpcs == 0 { // allow user override
		ss.MaxEpcs = 100
		ss.NZeroStop = 1
	}

	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	ss.TrainEnv.Table = etable.NewIdxView(ss.ABPats)
	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually

	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Dsc = "testing params and state"
	ss.TestEnv.Table = etable.NewIdxView(ss.ABPats)
	ss.TestEnv.Validate()
	ss.TestEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually

	ss.TrainEnv.Init(0)
	ss.TestEnv.Init(0)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.InitName(net, "ABAC")
	inp := net.AddLayer2D("Input", 5, 5, emer.Input)
	ctx := net.AddLayer2D("Context", 5, 5, emer.Input)
	hid := net.AddLayer2D("Hidden", 5, 10, emer.Hidden)
	out := net.AddLayer2D("Output", 5, 5, emer.Target)

	full := prjn.NewFull()
	net.ConnectLayers(inp, hid, full, emer.Forward)
	net.ConnectLayers(ctx, hid, full, emer.Forward)
	net.BidirConnectLayers(hid, out, full)

	ctx.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Input", YAlign: relpos.Front, Space: 2})
	hid.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Input", YAlign: relpos.Front, XAlign: relpos.Left})
	out.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Hidden", YAlign: relpos.Front, XAlign: relpos.Middle})

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
	ss.TrainEnv.Init(0) // reset run
	ss.StopNow = false
	ss.SetParams("", false) // all sheets
	ss.NewRun()
	ss.UpdateView(true)
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
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%v\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Trial.Cur, ss.Time.Cycle, ss.TrainEnv.TrialName)
	} else {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%v\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TestEnv.Trial.Cur, ss.Time.Cycle, ss.TestEnv.TrialName)
	}
}

func (ss *Sim) UpdateView(train bool) {
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters(train))
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
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}

	// update prior weight changes at start, so any DWt values remain visible at end
	// you might want to do this less frequently to achieve a mini-batch update
	// in which case, move it out to the TrainTrial method where the relevant
	// counters are being dealt with.
	if train {
		ss.Net.WtFmDWt()
	}

	ss.Net.AlphaCycInit()
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdt {
				case leabra.Cycle:
					ss.UpdateView(train)
				case leabra.FastSpike:
					if (cyc+1)%10 == 0 {
						ss.UpdateView(train)
					}
				}
			}
		}
		ss.Net.QuarterFinal(&ss.Time)
		ss.Time.QuarterInc()
		if ss.ViewOn {
			switch {
			case viewUpdt <= leabra.Quarter:
				ss.UpdateView(train)
			case viewUpdt == leabra.Phase:
				if qtr >= 2 {
					ss.UpdateView(train)
				}
			}
		}
	}

	if train {
		ss.Net.DWt()
	}
	if ss.ViewOn && viewUpdt == leabra.AlphaCycle {
		ss.UpdateView(train)
	}
}

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"Input", "Context", "Output"}
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
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
			ss.UpdateView(true)
		}
		if ss.TestInterval > 0 && epc%ss.TestInterval == 0 { // note: epc is *next* so won't trigger first time
			ss.TestAll()
		}
		learned := (ss.NZeroStop > 0 && ss.NZero >= ss.NZeroStop)
		if ss.TrainEnv.Table.Table == ss.ABPats && (learned || epc == 50) {
			ss.TrainEnv.Table = etable.NewIdxView(ss.ACPats)
			learned = false
		}
		if learned || epc >= ss.MaxEpcs {
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
	ss.TrainEnv.Table = etable.NewIdxView(ss.ABPats)
	ss.TrainEnv.Init(run)
	ss.TestEnv.Init(run)
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
	ss.SumSSE = 0
	ss.SumAvgSSE = 0
	ss.SumCosDiff = 0
	ss.SumErr = 0
	ss.FirstZero = -1
	ss.NZero = 0
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
		vp.BlockUpdates()
		if ss.ToolBar != nil {
			ss.ToolBar.UpdateActions()
		}
		vp.UnblockUpdates()
		vp.SetNeedsFullRender()
	}
}

// SaveWeights saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWeights(filename gi.FileName) {
	ss.Net.SaveWtsJSON(filename)
}

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial(returnOnChg bool) {
	ss.TestEnv.Step()

	// Query counters FIRST
	_, _, chg := ss.TestEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewOn && ss.TestUpdt > leabra.AlphaCycle {
			ss.UpdateView(false)
		}
		if returnOnChg {
			return
		}
	}

	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc(false)   // !train
	ss.TrialStats(false) // !accumulate
	ss.LogTstTrl(ss.TstTrlLog)
}

// TestItem tests given item which is at given index in test item list
func (ss *Sim) TestItem(idx int) {
	cur := ss.TestEnv.Trial.Cur
	ss.TestEnv.Trial.Cur = idx
	ss.TestEnv.SetTrialName()
	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc(false)   // !train
	ss.TrialStats(false) // !accumulate
	ss.TestEnv.Trial.Cur = cur
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.TestNm = "AB"
	ss.TestEnv.Table = etable.NewIdxView(ss.ABPats)
	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
	for {
		ss.TestTrial(true) // return on chg
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}
	if !ss.StopNow {
		ss.TestNm = "AC"
		ss.TestEnv.Table = etable.NewIdxView(ss.ACPats)
		ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
		for {
			ss.TestTrial(true)
			_, _, chg := ss.TestEnv.Counter(env.Epoch)
			if chg || ss.StopNow {
				break
			}
		}
		// log only at very end
		ss.LogTstEpc(ss.TstEpcLog)
	}
}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.StopNow = false
	ss.TestAll()
	ss.Stopped()
}

// RepsAnalysis does a full test and then runs tests of representations
func (ss *Sim) RepsAnalysis() {
	ss.RunTestAll()

	rels := etable.NewIdxView(ss.TstTrlLog)
	rels.SortCol(ss.TstTrlLog.ColIdx("TrialName"), true)
	ss.HiddenReps.SimMat.TableCol(rels, "Hidden", "TrialName", true, metric.Correlation64)
	ss.HiddenReps.PCA.TableCol(rels, "Hidden", metric.Covariance64)
	ss.HiddenReps.PCA.ProjectColToTable(ss.HiddenReps.PCAPrjn, rels, "Hidden", "TrialName", []int{0, 1})
	ss.ConfigPCAPlot(ss.HiddenReps.PCAPlot, ss.HiddenReps.PCAPrjn)
	ss.ClustPlot(ss.HiddenReps.ClustPlot, rels, "Hidden")

	ss.Stopped()
}

func (ss *Sim) ConfigPCAPlot(plt *eplot.Plot2D, dt *etable.Table) {
	nm, _ := dt.MetaData["name"]
	plt.Params.Title = "AB-AC PCA Plot: " + nm
	plt.Params.XAxisCol = "Prjn0"
	plt.SetTable(dt)
	plt.Params.Lines = false
	plt.Params.Points = true
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("TrialName", true, true, 0, false, 0)
	plt.SetColParams("Prjn0", false, true, 0, false, 0)
	plt.SetColParams("Prjn1", true, true, 0, false, 0)
}

// ClustPlot does one cluster plot on given table column
func (ss *Sim) ClustPlot(plt *eplot.Plot2D, ix *etable.IdxView, colNm string) {
	nm, _ := ix.Table.MetaData["name"]
	smat := &simat.SimMat{}
	smat.TableCol(ix, colNm, "TrialName", false, metric.Euclidean64)
	pt := &etable.Table{}
	clust.Plot(pt, clust.Glom(smat, clust.ContrastDist), smat)
	plt.InitName(plt, colNm)
	plt.Params.Title = "Cluster Plot of: " + nm + " " + colNm
	plt.Params.XAxisCol = "X"
	plt.SetTable(pt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("X", false, true, 0, false, 0)
	plt.SetColParams("Y", true, true, 0, false, 0)
	plt.SetColParams("Label", true, false, 0, false, 0)
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

	spo := ss.Params.SetByName("Base").SheetByName("Network").SelByName("Prjn")
	spo.Params.SetParamByName("Prjn.WtInit.Var", fmt.Sprintf("%g", ss.WtInitVar))
	spo.Params.SetParamByName("Prjn.Learn.XCal.LLrn", fmt.Sprintf("%g", ss.XCalLLrn))
	spo.Params.SetParamByName("Prjn.Learn.Lrate", fmt.Sprintf("%g", ss.Lrate))

	err := ss.SetParamsSet("Base", sheet, setMsg)

	hid := ss.Net.LayerByName("Hidden").(leabra.LeabraLayer).AsLeabra()
	hid.Inhib.Layer.Gi = ss.HiddenInhibGi

	fmc := hid.RcvPrjns.SendName("Context").(leabra.LeabraPrjn).AsLeabra()
	fmc.WtScale.Rel = ss.FmContext

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
		netp, ok := pset.Sheets["Network"]
		if ok {
			ss.Net.ApplyParams(netp, setMsg)
		}
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
	ab, err := Asset(fnm)
	if err != nil {
		log.Println(err)
		return err
	}
	err = dt.ReadCSV(bytes.NewBuffer(ab), etable.Tab)
	if err != nil {
		log.Println(err)
	} else {
		for i := 1; i < len(dt.Cols); i++ {
			dt.Cols[i].SetMetaData("grid-fill", "0.9")
		}
	}
	return err
}

func (ss *Sim) OpenPats() {
	// patgen.ReshapeCppFile(ss.ABPats, "ABInputs.dat", "ab_pats.tsv") // one-time reshape
	// patgen.ReshapeCppFile(ss.ACPats, "ACInputs.dat", "ac_pats.tsv") // one-time reshape
	ss.OpenPatAsset(ss.ABPats, "ab_pats.tsv", "AB", "AB Training patterns")
	// err := dt.OpenCSV("ab_pats.tsv", etable.Tab)
	ss.OpenPatAsset(ss.ACPats, "ac_pats.tsv", "AC", "AC Training patterns")
	// err := dt.OpenCSV("ac_pats.tsv", etable.Tab)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

// ValsTsr gets value tensor of given name, creating if not yet made
func (ss *Sim) ValsTsr(name string) *etensor.Float32 {
	if ss.ValsTsrs == nil {
		ss.ValsTsrs = make(map[string]*etensor.Float32)
	}
	tsr, ok := ss.ValsTsrs[name]
	if !ok {
		tsr = &etensor.Float32{}
		ss.ValsTsrs[name] = tsr
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

	epc := ss.TrainEnv.Epoch.Prv           // this is triggered by increment so use previous value
	nt := float64(ss.TrainEnv.Table.Len()) // number of trials in view

	ss.EpcSSE = ss.SumSSE / nt
	ss.SumSSE = 0
	ss.EpcAvgSSE = ss.SumAvgSSE / nt
	ss.SumAvgSSE = 0
	ss.EpcPctErr = float64(ss.SumErr) / nt
	ss.SumErr = 0
	ss.EpcPctCor = 1 - ss.EpcPctErr
	ss.EpcCosDiff = ss.SumCosDiff / nt
	ss.SumCosDiff = 0
	if ss.FirstZero < 0 && ss.EpcPctErr == 0 {
		ss.FirstZero = epc
	}
	if ss.EpcPctErr == 0 {
		ss.NZero++
	} else {
		ss.NZero = 0
	}

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
		dt.WriteCSVRow(ss.TrnEpcFile, row, etable.Tab, true)
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
	plt.Params.Title = "AB-AC Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("SSE", false, true, 0, false, 0) // default plot
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("PctErr", true, true, 0, true, 1)
	plt.SetColParams("PctCor", false, true, 0, true, 1)
	plt.SetColParams("CosDiff", false, true, 0, true, 1)

	return plt
}

//////////////////////////////////////////////
//  TstTrlLog

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *etable.Table) {
	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value
	trl := ss.TestEnv.Trial.Cur

	row := dt.Rows
	if ss.TestNm == "AB" && trl == 0 { // reset at start
		row = 0
		dt.SetNumRows(row)
	}
	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellString("TestNm", row, ss.TestNm)
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("TrialName", row, ss.TestEnv.TrialName)
	dt.SetCellFloat("Err", row, ss.TrlErr)
	dt.SetCellFloat("SSE", row, ss.TrlSSE)
	dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	for _, lnm := range ss.TstRecLays {
		tsr := ss.ValsTsr(lnm)
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		ly.UnitValsTensor(tsr, "ActM") // get minus phase act
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

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"TestNm", etensor.STRING, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"Err", etensor.FLOAT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.TstRecLays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		sch = append(sch, etable.Column{lnm, etensor.FLOAT64, ly.Shp.Shp, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "AB-AC Test Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("TestNm", false, true, 0, false, 0)
	plt.SetColParams("Trial", false, true, 0, false, 0)
	plt.SetColParams("TrialName", false, true, 0, false, 0)
	plt.SetColParams("Err", false, true, 0, false, 0)
	plt.SetColParams("SSE", true, true, 0, false, 0) // default plot
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("CosDiff", false, true, 0, true, 1)

	for _, lnm := range ss.TstRecLays {
		plt.SetColParams(lnm, false, true, 0, true, 1)
	}
	return plt
}

//////////////////////////////////////////////
//  TstEpcLog

func (ss *Sim) LogTstEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	trl := ss.TstTrlLog
	tix := etable.NewIdxView(trl)
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

	trix := etable.NewIdxView(trl)
	spl := split.GroupBy(trix, []string{"TestNm"})
	for _, ts := range ss.TstStatNms {
		split.Agg(spl, ts, agg.AggMean)
	}
	ss.TstStats = spl.AggsToTable(true) // no stat name

	for ri := 0; ri < ss.TstStats.Rows; ri++ {
		tst := ss.TstStats.CellString("TestNm", ri)
		for _, ts := range ss.TstStatNms {
			dt.SetCellFloat(tst+" "+ts, row, ss.TstStats.CellFloat(ts, ri))
		}
	}

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
	for _, tn := range ss.TstNms {
		for _, ts := range ss.TstStatNms {
			sch = append(sch, etable.Column{tn + " " + ts, etensor.FLOAT64, nil, nil})
		}
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "AB-AC Testing Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("SSE", false, true, 0, false, 0) // default plot
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("PctErr", false, true, 0, true, 1)
	plt.SetColParams("PctCor", false, true, 0, true, 1)
	plt.SetColParams("CosDiff", false, true, 0, true, 1)
	for _, tn := range ss.TstNms {
		for _, ts := range ss.TstStatNms {
			plt.SetColParams(tn+" "+ts, true, true, 0, true, 1) // default plot
		}
	}
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
	epcix := etable.NewIdxView(epclog)
	// compute mean over last N epochs for run level
	nlast := 1
	if nlast > epcix.Len()-1 {
		nlast = epcix.Len() - 1
	}
	epcix.Idxs = epcix.Idxs[epcix.Len()-nlast:]

	tstlog := ss.TstEpcLog
	tstix := etable.NewIdxView(tstlog)
	// compute mean over last N epochs for run level
	if nlast > tstix.Len()-1 {
		nlast = tstix.Len() - 1
	}
	tstix.Idxs = tstix.Idxs[tstix.Len()-nlast:]

	params := fmt.Sprintf("hid_gi: %g, wt_var: %g, fm_ctxt: %g, lrate: %g", ss.HiddenInhibGi, ss.WtInitVar, ss.FmContext, ss.Lrate)

	dt.SetCellFloat("Run", row, float64(run))
	dt.SetCellString("Params", row, params)
	dt.SetCellFloat("FirstZero", row, float64(ss.FirstZero))
	dt.SetCellFloat("SSE", row, agg.Mean(epcix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, agg.Mean(epcix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, agg.Mean(epcix, "PctErr")[0])
	dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])
	dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])

	for _, tn := range ss.TstNms {
		for _, ts := range ss.TstStatNms {
			nm := tn + " " + ts
			dt.SetCellFloat(nm, row, agg.Mean(tstix, nm)[0])
		}
	}

	runix := etable.NewIdxView(dt)
	spl := split.GroupBy(runix, []string{"Params"})
	for _, tn := range ss.TstNms {
		nm := tn + " " + "Err"
		split.Desc(spl, nm)
	}
	split.Desc(spl, "FirstZero")
	ss.RunStats = spl.AggsToTable(false)

	// note: essential to use Go version of update when called from another goroutine
	ss.RunPlot.GoUpdate()
	if ss.RunFile != nil {
		if row == 0 {
			dt.WriteCSVHeaders(ss.RunFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.RunFile, row, etable.Tab, true)
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
		{"FirstZero", etensor.FLOAT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	for _, tn := range ss.TstNms {
		for _, ts := range ss.TstStatNms {
			sch = append(sch, etable.Column{tn + " " + ts, etensor.FLOAT64, nil, nil})
		}
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigRunPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "AB-AC Run Plot"
	plt.Params.XAxisCol = "Run"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("FirstZero", false, true, 0, false, 0)
	plt.SetColParams("SSE", false, true, 0, false, 0)
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("PctErr", false, true, 0, true, 1)
	plt.SetColParams("PctCor", false, true, 0, true, 1)
	plt.SetColParams("CosDiff", false, true, 0, true, 1)
	for _, tn := range ss.TstNms {
		for _, ts := range ss.TstStatNms {
			plt.SetColParams(tn+" "+ts, true, true, 0, true, 1) // default plot
		}
	}
	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.ViewDefaults()
	nv.Scene().Camera.Pose.Pos.Set(0, .95, 3.2) // more "head on" than default which is more "top down"
	nv.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("abac")
	gi.SetAppAbout(`explores the classic paired associates learning task in a cortical-like network, which exhibits catastrophic levels of interference. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch8/abac/README.md">README.md on GitHub</a>.</p>`)

	win := gi.NewWindow2D("abac", "AB-AC Interference", width, height, true)
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = gi.X
	split.SetStretchMax()

	sv := giv.AddNewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := gi.AddNewTabView(split, "tv")

	nv := tv.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	nv.Var = "Act"
	nv.SetNet(ss.Net)
	ss.NetView = nv
	ss.ConfigNetView(nv)

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TrnEpcPlot").(*eplot.Plot2D)
	ss.TrnEpcPlot = ss.ConfigTrnEpcPlot(plt, ss.TrnEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstTrlPlot").(*eplot.Plot2D)
	ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstEpcPlot").(*eplot.Plot2D)
	ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.TstEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "RunPlot").(*eplot.Plot2D)
	ss.RunPlot = ss.ConfigRunPlot(plt, ss.RunLog)

	split.SetSplits(.3, .7)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Train", Icon: "run", Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.",
		UpdateFunc: func(act *gi.Action) {
			act.SetActiveStateUpdt(!ss.IsRunning)
		}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			// ss.Train()
			go ss.Train()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Stop()
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Trial", Icon: "step-fwd", Tooltip: "Advances one training trial at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TrainTrial()
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Epoch", Icon: "fast-fwd", Tooltip: "Advances one epoch (complete set of training patterns) at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainEpoch()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Run", Icon: "fast-fwd", Tooltip: "Advances one full training Run at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainRun()
		}
	})

	tbar.AddSeparator("test")

	tbar.AddAction(gi.ActOpts{Label: "Test Trial", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TestTrial(false) // don't break on chg
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Test All", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestAll()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Reps Analysis", Icon: "fast-fwd", Tooltip: "Does an All Test All and analyzes the resulting Hidden and AgentCode activations.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RepsAnalysis()
		}
	})

	tbar.AddSeparator("log")

	tbar.AddAction(gi.ActOpts{Label: "Reset RunLog", Icon: "update", Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.RunLog.SetNumRows(0)
			ss.RunPlot.Update()
		})

	tbar.AddSeparator("misc")

	tbar.AddAction(gi.ActOpts{Label: "Build Net", Icon: "new", Tooltip: "Build network -- do this if any sizes have changed"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.Net.Build()
			ss.NetView.Config()
			ss.NetView.Update()
			vp.SetNeedsFullRender()
		})

	tbar.AddAction(gi.ActOpts{Label: "New Seed", Icon: "new", Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.NewRndSeed()
		})

	tbar.AddAction(gi.ActOpts{Label: "Defaults", Icon: "update", Tooltip: "Restore initial default parameters.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Defaults()
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch8/abac/README.md")
		})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(win)

	// note: Command in shortcuts is automatically translated into Control for
	// Linux, Windows or Meta for MacOS
	// fmen := win.MainMenu.ChildByName("File", 0).(*gi.Action)
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Open", Shortcut: "Command+O"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		FileViewOpenSVG(vp)
	// 	})
	// fmen.Menu.AddSeparator("csep")
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Close Window", Shortcut: "Command+W"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		win.Close()
	// 	})

	inQuitPrompt := false
	gi.SetQuitReqFunc(func() {
		if inQuitPrompt {
			return
		}
		inQuitPrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Quit?",
			Prompt: "Are you <i>sure</i> you want to quit and lose any unsaved params, weights, logs, etc?"}, true, true,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inQuitPrompt = false
				}
			})
	})

	// gi.SetQuitCleanFunc(func() {
	// 	fmt.Printf("Doing final Quit cleanup here..\n")
	// })

	inClosePrompt := false
	win.SetCloseReqFunc(func(w *gi.Window) {
		if inClosePrompt {
			return
		}
		inClosePrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Close Window?",
			Prompt: "Are you <i>sure</i> you want to close the window?  This will Quit the App as well, losing all unsaved params, weights, logs, etc"}, true, true,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inClosePrompt = false
				}
			})
	})

	win.SetCloseCleanFunc(func(w *gi.Window) {
		go gi.Quit() // once main window is closed, quit
	})

	win.MainMenuUpdated()
	return win
}

// These props register Save methods so they can be used
var SimProps = ki.Props{
	"CallMethods": ki.PropSlice{
		{"SaveWeights", ki.Props{
			"desc": "save network weights to file",
			"icon": "file-save",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
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
