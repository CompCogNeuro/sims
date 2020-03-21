// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
ss explores the way that regularities and exceptions are learned in the mapping between spelling (orthography) and sound (phonology), in the context of a "direct pathway" mapping between these two forms of word representations.
*/
package main

import (
	"bytes"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/metric"
	"github.com/emer/etable/split"
	"github.com/emer/leabra/leabra"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

func main() {
	TheSim.New()
	TheSim.Config()
	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			guirun()
		})
	}
}

func guirun() {
	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// EnvType is the type of test environment
type EnvType int32

//go:generate stringer -type=EnvType

var KiT_EnvType = kit.Enums.AddEnum(EnvTypeN, kit.NotBitFlag, nil)

func (ev EnvType) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *EnvType) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

const (
	Probe EnvType = iota
	Besner
	Glushko
	Taraban

	EnvTypeN
)

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "all extra learning factors",
				Params: params.Params{
					"Prjn.Learn.Norm.On":     "true",
					"Prjn.Learn.Momentum.On": "true",
					"Prjn.Learn.WtBal.On":    "true",
					"Prjn.Learn.Lrate":       "0.04",
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
			{Sel: ".Back", Desc: "weaker top down as usual",
				Params: params.Params{
					"Prjn.WtScale.Rel": ".1",
				}},
			{Sel: "#HiddenToOrthoCode", Desc: "stronger from hidden",
				Params: params.Params{
					"Prjn.WtScale.Rel": ".2",
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
	TestingEnv    EnvType           `desc:"the environment to use for testing -- only takes effect for TestAll"`
	Net           *leabra.Network   `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	TrainPats     *etable.Table     `view:"no-inline" desc:"training patterns"`
	ProbePats     *etable.Table     `view:"no-inline" desc:"testing patterns based on training data"`
	BesnerPats    *etable.Table     `view:"no-inline" desc:"nonword testing patterns"`
	GlushkoPats   *etable.Table     `view:"no-inline" desc:"nonword testing patterns"`
	TarabanPats   *etable.Table     `view:"no-inline" desc:"nonword testing patterns"`
	PhonConsPats  *etable.Table     `view:"no-inline" desc:"phonology consonant patterns"`
	PhonVowelPats *etable.Table     `view:"no-inline" desc:"phonology vowel patterns"`
	TrnEpcLog     *etable.Table     `view:"no-inline" desc:"training epoch-level log data"`
	TstTrlLog     *etable.Table     `view:"no-inline" desc:"testing trial-level log data"`
	TstEpcLog     *etable.Table     `view:"no-inline" desc:"summary testing results"`
	TstErrLog     *etable.Table     `view:"no-inline" desc:"testing trials with errors (aggregating over multiple locations)"`
	TstRTStats    *etable.Table     `view:"no-inline" desc:"average RT as a function of type"`
	RunLog        *etable.Table     `view:"no-inline" desc:"summary log of each run"`
	RunStats      *etable.Table     `view:"no-inline" desc:"aggregate stats on all runs"`
	Params        params.Sets       `view:"no-inline" desc:"full collection of param sets"`
	ParamSet      string            `desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set"`
	Tag           string            `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"`
	MaxRuns       int               `desc:"maximum number of model runs to perform"`
	MaxEpcs       int               `desc:"maximum number of epochs to run per model run"`
	NZeroStop     int               `desc:"if a positive number, training will stop after this many epochs with zero SSE"`
	TrainEnv      env.FreqTable     `desc:"Training environment (frequency-based) -- contains everything about iterating over input / output patterns over training"`
	TestEnv       env.FixedTable    `desc:"Testing environment -- manages iterating over testing"`
	Time          leabra.Time       `desc:"leabra timing parameters and state"`
	ViewOn        bool              `desc:"whether to update the network view while running"`
	TrainUpdt     leabra.TimeScales `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt      leabra.TimeScales `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	TestInterval  int               `desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`
	LayStatNms    []string          `desc:"names of layers to collect more detailed stats on (avg act, etc)"`

	// statistics: note use float64 as that is best for etable.Table
	TrlName       string  `inactive:"+" desc:"name of current input pattern"`
	TrlPhon       string  `inactive:"+" desc:"pronunciation from phonology pattern"`
	TrlPhonSSE    float64 `inactive:"+" desc:"SSE for closest phonology pattern -- > 3 = blend"`
	TrlErr        float64 `inactive:"+" desc:"1 if trial was error, 0 if correct -- based on *closest pattern* stat, not SSE"`
	TrlNameErr    float64 `inactive:"+" desc:"1 if trial was error according to name match, 0 if correct -- based on phonology decoding, not SSE"`
	TrlSSE        float64 `inactive:"+" desc:"current trial's sum squared error"`
	TrlAvgSSE     float64 `inactive:"+" desc:"current trial's average sum squared error"`
	TrlCosDiff    float64 `inactive:"+" desc:"current trial's cosine difference"`
	TrlRTCycles   float64 `inactive:"+" desc:"current trial's number of cycles for phon activity > 0.8"`
	EpcSSE        float64 `inactive:"+" desc:"last epoch's total sum squared error"`
	EpcAvgSSE     float64 `inactive:"+" desc:"last epoch's average sum squared error (average over trials, and over units within layer)"`
	EpcPctErr     float64 `inactive:"+" desc:"last epoch's average TrlErr"`
	EpcPctCor     float64 `inactive:"+" desc:"1 - last epoch's average TrlErr"`
	EpcPctNameErr float64 `inactive:"+" desc:"last epoch's average TrlNameErr"`
	EpcCosDiff    float64 `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	EpcPerTrlMSec float64 `inactive:"+" desc:"how long did the epoch take per trial in wall-clock milliseconds"`
	FirstZero     int     `inactive:"+" desc:"epoch at when SSE first went to zero"`
	NZero         int     `inactive:"+" desc:"number of epochs in a row with zero SSE"`

	// internal state - view:"-"
	SumErr       float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumNameErr   float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumSSE       float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumAvgSSE    float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumCosDiff   float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	Win          *gi.Window                  `view:"-" desc:"main GUI window"`
	NetView      *netview.NetView            `view:"-" desc:"the network viewer"`
	ToolBar      *gi.ToolBar                 `view:"-" desc:"the master toolbar"`
	TrnEpcPlot   *eplot.Plot2D               `view:"-" desc:"the training epoch plot"`
	TstEpcPlot   *eplot.Plot2D               `view:"-" desc:"the testing epoch plot"`
	TstTrlPlot   *eplot.Plot2D               `view:"-" desc:"the test-trial plot"`
	TstRTPlot    *eplot.Plot2D               `view:"-" desc:"the test-trial plot"`
	RunPlot      *eplot.Plot2D               `view:"-" desc:"the run plot"`
	TrnEpcFile   *os.File                    `view:"-" desc:"log file"`
	RunFile      *os.File                    `view:"-" desc:"log file"`
	ValsTsrs     map[string]*etensor.Float32 `view:"-" desc:"for holding layer values"`
	SaveWts      bool                        `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	NoGui        bool                        `view:"-" desc:"if true, runing in no GUI mode"`
	LogSetParams bool                        `view:"-" desc:"if true, print message for all params that are set"`
	IsRunning    bool                        `view:"-" desc:"true if sim is running"`
	StopNow      bool                        `view:"-" desc:"flag to stop running"`
	NeedsNewRun  bool                        `view:"-" desc:"flag to initialize NewRun if last one finished"`
	RndSeed      int64                       `inactive:"+" desc:"the current random seed"`
	LastEpcTime  time.Time                   `view:"-" desc:"timer for last epoch"`
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
	ss.ProbePats = &etable.Table{}
	ss.BesnerPats = &etable.Table{}
	ss.GlushkoPats = &etable.Table{}
	ss.TarabanPats = &etable.Table{}
	ss.PhonConsPats = &etable.Table{}
	ss.PhonVowelPats = &etable.Table{}
	ss.TrnEpcLog = &etable.Table{}
	ss.TstEpcLog = &etable.Table{}
	ss.TstTrlLog = &etable.Table{}
	ss.RunLog = &etable.Table{}
	ss.RunStats = &etable.Table{}
	ss.Params = ParamSets
	ss.RndSeed = 10 // default 1 was particularly bad for direct full lesions..
	ss.ViewOn = true
	ss.TrainUpdt = leabra.Quarter
	ss.TestUpdt = leabra.Cycle
	ss.TestInterval = -1
	ss.LayStatNms = []string{"OrthoCode", "Hidden"}
}

func (ss *Sim) Defaults() {
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
		ss.MaxRuns = 1
	}
	if ss.MaxEpcs == 0 { // allow user override
		ss.MaxEpcs = 400
		ss.NZeroStop = -1
	}

	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	ss.TrainEnv.Table = etable.NewIdxView(ss.TrainPats)
	ss.TrainEnv.NSamples = 1
	ss.TrainEnv.RndSamp = true
	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually

	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Dsc = "testing params and state"
	ss.TestEnv.GroupCol = "Type"
	ss.TestEnv.Table = etable.NewIdxView(ss.ProbePats)
	ss.TestEnv.Sequential = true
	ss.TestEnv.Validate()

	ss.TrainEnv.Init(0)
	ss.TestEnv.Init(0)
}

func (ss *Sim) ConfigTestEnv() {
	switch ss.TestingEnv {
	case Probe:
		ss.TestEnv.Table = etable.NewIdxView(ss.ProbePats)
	case Besner:
		ss.TestEnv.Table = etable.NewIdxView(ss.BesnerPats)
	case Glushko:
		ss.TestEnv.Table = etable.NewIdxView(ss.GlushkoPats)
	case Taraban:
		ss.TestEnv.Table = etable.NewIdxView(ss.TarabanPats)
	}
	ss.TestEnv.Init(0)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.InitName(net, "SpellSound")
	ort := net.AddLayer4D("Ortho", 1, 7, 9, 3, emer.Input)
	ocd := net.AddLayer4D("OrthoCode", 1, 5, 14, 6, emer.Hidden)
	hid := net.AddLayer2D("Hidden", 20, 30, emer.Hidden)
	phn := net.AddLayer4D("Phon", 1, 7, 10, 2, emer.Target)

	full := prjn.NewFull()
	ocdPrjn := prjn.NewPoolTile()
	ocdPrjn.Size.Set(3, 1)
	ocdPrjn.Skip.Set(1, 0)
	ocdPrjn.Start.Set(0, 0)

	net.ConnectLayers(ort, ocd, ocdPrjn, emer.Forward)
	net.BidirConnectLayers(ocd, hid, full)
	net.BidirConnectLayers(hid, phn, full)

	ocd.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Ortho", YAlign: relpos.Front, XAlign: relpos.Middle})
	hid.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "OrthoCode", YAlign: relpos.Front, XAlign: relpos.Middle})
	phn.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Hidden", YAlign: relpos.Front, XAlign: relpos.Middle})

	// sig faster on blanca with no threads, slightly slower on mac
	// ocd.SetThread(1)
	// hid.SetThread(2) // all on 1,2 threads is also slower
	// 	phn.SetThread(3) // slower with 4 on mac

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
	ss.ConfigEnv()
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
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%v\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Trial.Cur, ss.Time.Cycle, ss.TrainEnv.TrialName.Cur)
	} else {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%v\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TestEnv.Trial.Cur, ss.Time.Cycle, ss.TestEnv.GroupName.Cur+": "+ss.TestEnv.TrialName.Cur)
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

	phn := ss.Net.LayerByName("Phon").(leabra.LeabraLayer).AsLeabra()
	rtcyc := -1

	ss.Net.AlphaCycInit()
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			ss.Time.CycleInc()
			if rtcyc < 0 {
				mxact := phn.Pools[0].Inhib.Act.Max
				if mxact > 0.5 {
					rtcyc = ss.Time.Cycle
				}
			}
			if ss.ViewOn {
				switch viewUpdt {
				case leabra.Cycle:
					if cyc != ss.Time.CycPerQtr-1 { // will be updated by quarter
						ss.UpdateView(train)
					}
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

	ss.TrlRTCycles = float64(rtcyc)

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

	lays := []string{"Ortho", "Phon"}
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
		ss.LrateSched(epc)
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
			ss.UpdateView(true)
		}
		if ss.TestInterval > 0 && epc%ss.TestInterval == 0 { // note: epc is *next* so won't trigger first time
			ss.TestAll()
		}
		if epc >= ss.MaxEpcs || (ss.NZeroStop > 0 && ss.NZero >= ss.NZeroStop) {
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

	phn := ss.Net.LayerByName("Phon").(leabra.LeabraLayer).AsLeabra()
	phn.SetType(emer.Target)
	ss.ApplyInputs(&ss.TrainEnv)
	ss.AlphaCyc(true)                              // train
	ss.TrialStats(true, ss.TrainEnv.TrialName.Cur) // accumulate
}

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
	ss.LogRun(ss.RunLog)
	if ss.SaveWts {
		fnm := ss.WeightsFileName()
		fmt.Printf("Saving Weights to: %v\n", fnm)
		ss.Net.SaveWtsJSON(gi.FileName(fnm))
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	run := ss.TrainEnv.Run.Cur
	ss.TrainEnv.Init(run)
	ss.TestEnv.Init(run)
	ss.Time.Reset()
	ss.Net.InitWts()
	ss.Net.LrateMult(1) // restore initial learning rate value
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
	ss.FirstZero = -1
	ss.NZero = 0
	// clear rest just to make Sim look initialized
	ss.TrlPhonSSE = 0
	ss.TrlErr = 0
	ss.TrlSSE = 0
	ss.TrlAvgSSE = 0
	ss.TrlCosDiff = 0
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
func (ss *Sim) TrialStats(accum bool, trlnm string) (sse, avgsse, cosdiff float64) {
	out := ss.Net.LayerByName("Phon").(leabra.LeabraLayer).AsLeabra()
	ss.TrlCosDiff = float64(out.CosDiff.Cos)
	ss.TrlSSE, ss.TrlAvgSSE = out.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
	if ss.TrlSSE == 0 {
		ss.TrlErr = 0
	} else {
		ss.TrlErr = 1
	}

	ss.TrlName = trlnm
	ss.TrlPhon, ss.TrlPhonSSE = ss.Pronounce(ss.Net)
	spnm := strings.Split(trlnm, "_")
	if ss.TrlPhon == spnm[2] {
		ss.TrlNameErr = 0
	} else {
		ss.TrlNameErr = 1
	}

	if accum {
		ss.SumErr += ss.TrlErr
		ss.SumNameErr += ss.TrlNameErr
		ss.SumSSE += ss.TrlSSE
		ss.SumAvgSSE += ss.TrlAvgSSE
		ss.SumCosDiff += ss.TrlCosDiff
	}
	return
}

// Pronounce returns the pronunciation of the phonological output layer
func (ss *Sim) Pronounce(net emer.Network) (string, float64) {
	vt := ss.ValsTsr("Phon")
	ly := net.LayerByName("Phon").(leabra.LeabraLayer).AsLeabra()
	ly.UnitValsTensor(vt, "ActM")

	ccol := ss.PhonConsPats.ColByName("Phon").(*etensor.Float32)
	vcol := ss.PhonVowelPats.ColByName("Phon").(*etensor.Float32)
	sseTol := float32(10.0)
	totSSE := float32(0.0)
	ph := ""
	for pi := 0; pi < 7; pi++ {
		cvt := vt.SubSpace([]int{0, pi}).(*etensor.Float32)
		nm := ""
		sse := float32(0.0)
		row := 0
		if pi == 3 { // vowel
			row, sse = metric.ClosestRow32(cvt, vcol, metric.SumSquaresBinTol32)
			nm = ss.PhonVowelPats.CellString("Name", row)
		} else {
			row, sse = metric.ClosestRow32(cvt, ccol, metric.SumSquaresBinTol32)
			nm = ss.PhonConsPats.CellString("Name", row)
		}
		if sse > sseTol {
			nm = "X"
		}
		ph += nm
		totSSE += sse
	}
	return ph, float64(totSSE)
}

// TrainEpoch runs training trials for remainder of this epoch
func (ss *Sim) TrainEpoch() {
	ss.SetParams("Network", false)
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

// LrateSched implements the learning rate schedule
func (ss *Sim) LrateSched(epc int) {
	doLr := false
	lrm := float32(1)
	switch epc {
	case 100:
		doLr = true
		lrm = 0.5
	case 200:
		doLr = true
		lrm = 0.2
	case 300:
		doLr = true
		lrm = 0.1
	case 400:
		doLr = true
		lrm = 0.05
	case 500:
		doLr = true
		lrm = 0.02
	case 600:
		doLr = true
		lrm = 0.01
	}
	if doLr {
		ss.Net.LrateMult(lrm)
		fmt.Printf("dropped lrate %g at epoch: %d\n", lrm, epc)
	}
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

// SaveWeights saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWeights(filename gi.FileName) {
	ss.Net.SaveWtsJSON(filename)
}

// OpenTrainedWts opens trained weights
func (ss *Sim) OpenTrainedWts() {
	ab, err := Asset("trained.wts") // embedded in executable
	if err != nil {
		log.Println(err)
	}
	ss.Net.ReadWtsJSON(bytes.NewBuffer(ab))
	// ss.Net.OpenWtsJSON("trained.wts")
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
		ss.LogTstEpc(ss.TstEpcLog)
		if returnOnChg {
			return
		}
	}

	// note: type must be in place before apply inputs
	phn := ss.Net.LayerByName("Phon").(leabra.LeabraLayer).AsLeabra()
	phn.SetType(emer.Compare)
	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc(false)                             // !train
	ss.TrialStats(false, ss.TestEnv.TrialName.Cur) // !accumulate
	ss.LogTstTrl(ss.TstTrlLog)
}

// TestItem tests given item which is at given index in test item list
func (ss *Sim) TestItem(idx int) {
	cur := ss.TestEnv.Trial.Cur
	ss.TestEnv.Trial.Cur = idx
	ss.TestEnv.SetTrialName()
	// note: type must be in place before apply inputs
	phn := ss.Net.LayerByName("Phon").(leabra.LeabraLayer).AsLeabra()
	phn.SetType(emer.Compare)
	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc(false)                             // !train
	ss.TrialStats(false, ss.TestEnv.TrialName.Cur) // !accumulate
	ss.TestEnv.Trial.Cur = cur
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.SetParams("Network", false)
	ss.ConfigTestEnv()
	ss.TstTrlLog.SetNumRows(0)
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

/////////////////////////////////////////////////////////////////////////
//   Params setting

// ParamsName returns name of current set of parameters
func (ss *Sim) ParamsName() string {
	if ss.ParamSet == "" {
		return "Base"
	}
	return ss.ParamSet
}

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
		err = ss.SetParamsSet(ss.ParamSet, sheet, setMsg)
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
	// patgen.ReshapeCppFile(ss.TrainPats, "ss_train_all.dat", "train_pats.tsv")          // one-time reshape
	// patgen.ReshapeCppFile(ss.ProbePats, "Probe.dat", "probe.tsv")                      // one-time reshape
	// patgen.ReshapeCppFile(ss.BesnerPats, "Besner.dat", "besner.tsv")                   // one-time reshape
	// patgen.ReshapeCppFile(ss.GlushkoPats, "Glushko.dat", "glushko.tsv")                // one-time reshape
	// patgen.ReshapeCppFile(ss.TarabanPats, "Taraban_Nw.dat", "taraban.tsv") // one-time reshape
	// patgen.ReshapeCppFile(ss.PhonConsPats, "std_phon_pt_cons.dat", "phon_cons.tsv")    // one-time reshape
	// patgen.ReshapeCppFile(ss.PhonVowelPats, "std_phon_pt_vowel.dat", "phon_vowel.tsv") // one-time reshape
	// patgen.ReshapeCppFile(ss.CloseSems, "SemClosePats.dat", "close_sems.tsv")       // one-time reshape
	ss.OpenPatAsset(ss.TrainPats, "train_pats.tsv", "TrainPats", "Training patterns")
	ss.OpenPatAsset(ss.ProbePats, "probe.tsv", "Probe", "Probe test patterns (words)")
	ss.OpenPatAsset(ss.BesnerPats, "besner.tsv", "Besner", "nonword test patterns")
	ss.OpenPatAsset(ss.GlushkoPats, "glushko.tsv", "Glushko", "nonword test patterns")
	ss.OpenPatAsset(ss.TarabanPats, "taraban.tsv", "Taraban", "nonword test patterns")
	ss.OpenPatAsset(ss.PhonConsPats, "phon_cons.tsv", "PhonCons", "Phonology patterns -- consonants")
	ss.OpenPatAsset(ss.PhonVowelPats, "phon_vowel.tsv", "PhonVowel", "Phonology patterns -- vowels")
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

// RunName returns a name for this run that combines Tag and Params -- add this to
// any file names that are saved.
func (ss *Sim) RunName() string {
	if ss.Tag != "" {
		return ss.Tag + "_" + ss.ParamsName()
	} else {
		return ss.ParamsName()
	}
}

// RunEpochName returns a string with the run and epoch numbers with leading zeros, suitable
// for using in weights file names.  Uses 3, 5 digits for each.
func (ss *Sim) RunEpochName(run, epc int) string {
	return fmt.Sprintf("%03d_%05d", run, epc)
}

// WeightsFileName returns default current weights file name
func (ss *Sim) WeightsFileName() string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + ss.RunEpochName(ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur) + ".wts.gz"
}

// LogFileName returns default log file name
func (ss *Sim) LogFileName(lognm string) string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + lognm + ".csv"
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
	ss.EpcPctNameErr = float64(ss.SumNameErr) / nt
	ss.SumNameErr = 0
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

	if ss.LastEpcTime.IsZero() {
		ss.EpcPerTrlMSec = 0
	} else {
		iv := time.Now().Sub(ss.LastEpcTime)
		ss.EpcPerTrlMSec = float64(iv) / (nt * float64(time.Millisecond))
	}
	ss.LastEpcTime = time.Now()

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("SSE", row, ss.EpcSSE)
	dt.SetCellFloat("AvgSSE", row, ss.EpcAvgSSE)
	dt.SetCellFloat("PctErr", row, ss.EpcPctErr)
	dt.SetCellFloat("PctCor", row, ss.EpcPctCor)
	dt.SetCellFloat("PctNameErr", row, ss.EpcPctNameErr)
	dt.SetCellFloat("CosDiff", row, ss.EpcCosDiff)
	dt.SetCellFloat("PerTrlMSec", row, ss.EpcPerTrlMSec)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+" ActAvg", row, float64(ly.Pools[0].ActAvg.ActPAvgEff))
	}

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
		{"PctNameErr", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"PerTrlMSec", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " ActAvg", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "SpellSound Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
	plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PctNameErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PerTrlMSec", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActAvg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 0.5)
	}
	return plt
}

//////////////////////////////////////////////
//  TstTrlLog

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *etable.Table) {
	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value

	trl := ss.TestEnv.Trial.Cur
	row := trl

	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("Type", row, ss.TestEnv.GroupName.Cur)
	dt.SetCellString("TrialName", row, ss.TrlName)
	dt.SetCellString("Phon", row, ss.TrlPhon)
	dt.SetCellFloat("PhonSSE", row, ss.TrlPhonSSE)
	dt.SetCellFloat("TrlNameErr", row, ss.TrlNameErr)
	dt.SetCellFloat("RTCycles", row, ss.TrlRTCycles)
	dt.SetCellFloat("Err", row, ss.TrlErr)
	dt.SetCellFloat("SSE", row, ss.TrlSSE)
	dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	// note: essential to use Go version of update when called from another goroutine
	ss.TstTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTstTrlLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	// nt := ss.TestEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"Type", etensor.STRING, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"Phon", etensor.STRING, nil, nil},
		{"PhonSSE", etensor.FLOAT64, nil, nil},
		{"TrlNameErr", etensor.FLOAT64, nil, nil},
		{"RTCycles", etensor.FLOAT64, nil, nil},
		{"Err", etensor.FLOAT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "SpellSound Test Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Type", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Phon", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PhonSSE", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("TrlNameErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("RTCycles", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Err", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	return plt
}

//////////////////////////////////////////////
//  TstEpcLog

func (ss *Sim) LogTstEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	trl := ss.TstTrlLog
	epc := ss.TrainEnv.Epoch.Prv // ?
	tix := etable.NewIdxView(trl)
	spl := split.GroupBy(tix, []string{"TrialName"})
	split.Agg(spl, "TrlNameErr", agg.AggMin)
	split.Agg(spl, "RTCycles", agg.AggMean)
	minerr := spl.AggsToTableCopy(etable.ColNameOnly)

	trlerr := minerr.ColByName("TrlNameErr")
	// noerr := etable.NewIdxView(minerr)
	// noerr.Filter(func(et *etable.Table, row int) bool {
	// 	// filter return value is for what to *keep* (=true), not exclude
	// 	// here we keep any row with a name that contains the string "in"
	// 	return trlerr.FloatVal1D(row) == 0
	// })
	allerr := etable.NewIdxView(minerr)
	allerr.Filter(func(et *etable.Table, row int) bool {
		// filter return value is for what to *keep* (=true), not exclude
		// here we keep any row with a name that contains the string "in"
		return trlerr.FloatVal1D(row) > 0
	})
	ss.TstErrLog = allerr.NewTable()

	rtspl := split.GroupBy(tix, []string{"Type"})
	split.Agg(rtspl, "RTCycles", agg.AggMean)
	split.Agg(rtspl, "RTCycles", agg.AggSem)
	ss.TstRTStats = rtspl.AggsToTable(etable.AddAggName)
	ss.ConfigTstRTPlot(ss.TstRTPlot, ss.TstRTStats)

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellString("TestEnv", row, ss.TestingEnv.String())
	dt.SetCellFloat("PctCor", row, 1-float64(ss.TstErrLog.Rows)/float64(minerr.Rows))

	// note: essential to use Go version of update when called from another goroutine
	ss.TstEpcPlot.GoUpdate()
}

func (ss *Sim) ConfigTstRTPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "SpellSound Test RT Plot"
	plt.Params.XAxisCol = "Type"
	plt.Params.Type = eplot.Bar
	if dt == nil {
		return plt
	}
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Type", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	cp := plt.SetColParams("RTCycles:Mean", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	cp.ErrCol = "RTCycles:Sem"
	return plt
}

func (ss *Sim) ConfigTstEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstEpcLog")
	dt.SetMetaData("desc", "Summary stats for testing trials")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"TestEnv", etensor.STRING, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "SpellSound Testing Epoch Plot"
	plt.Params.XAxisCol = "TestEnv"
	plt.Params.Type = eplot.Bar
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TestEnv", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctCor", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)

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
	nlast := 5
	if nlast > epcix.Len()-1 {
		nlast = epcix.Len() - 1
	}
	epcix.Idxs = epcix.Idxs[epcix.Len()-nlast:]

	params := ""

	dt.SetCellFloat("Run", row, float64(run))
	dt.SetCellString("Params", row, params)
	dt.SetCellFloat("FirstZero", row, float64(ss.FirstZero))
	dt.SetCellFloat("SSE", row, agg.Mean(epcix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, agg.Mean(epcix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, agg.Mean(epcix, "PctErr")[0])
	dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])

	runix := etable.NewIdxView(dt)
	spl := split.GroupBy(runix, []string{"Params"})
	split.Desc(spl, "FirstZero")
	split.Desc(spl, "PctCor")
	ss.RunStats = spl.AggsToTable(false)

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

	dt.SetFromSchema(etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Params", etensor.STRING, nil, nil},
		{"FirstZero", etensor.FLOAT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}, 0)
}

func (ss *Sim) ConfigRunPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "SpellSound Run Plot"
	plt.Params.XAxisCol = "Run"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("FirstZero", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0) // default plot
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
	nv.Scene().Camera.Pose.Pos.Set(0, 1.05, 2.75)
	nv.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("ss")
	gi.SetAppAbout(`explores the way that regularities and exceptions are learned in the mapping between spelling (orthography) and sound (phonology), in the context of a "direct pathway" mapping between these two forms of word representations. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch9/ss/README.md">README.md on GitHub</a>.</p>`)

	win := gi.NewMainWindow("ss", "SpellSound", width, height)
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = mat32.X
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

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstRTPlot").(*eplot.Plot2D)
	ss.TstRTPlot = ss.ConfigTstRTPlot(plt, nil)

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

	tbar.AddAction(gi.ActOpts{Label: "Test Item", Icon: "step-fwd", Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		gi.StringPromptDialog(vp, "", "Test Item",
			gi.DlgOpts{Title: "Test Item", Prompt: "Enter the Name of a given input pattern to test (case insensitive, contains given string."},
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				dlg := send.(*gi.Dialog)
				if sig == int64(gi.DialogAccepted) {
					val := gi.StringPromptDialogValue(dlg)
					idxs := ss.TestEnv.Table.RowsByString("Name", val, etable.Contains, etable.IgnoreCase)
					if len(idxs) == 0 {
						gi.PromptDialog(nil, gi.DlgOpts{Title: "Name Not Found", Prompt: "No patterns found containing: " + val}, gi.AddOk, gi.NoCancel, nil, nil)
					} else {
						if !ss.IsRunning {
							ss.IsRunning = true
							fmt.Printf("testing index: %v\n", idxs[0])
							ss.TestItem(idxs[0])
							ss.IsRunning = false
							vp.SetNeedsFullRender()
						}
					}
				}
			})
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

	tbar.AddSeparator("log")

	tbar.AddAction(gi.ActOpts{Label: "Open Trained Wts", Icon: "update", Tooltip: "open weights trained for 250 epochs with default params", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.OpenTrainedWts()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Reset RunLog", Icon: "update", Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.RunLog.SetNumRows(0)
			ss.RunPlot.Update()
		})

	tbar.AddSeparator("misc")

	tbar.AddAction(gi.ActOpts{Label: "New Seed", Icon: "new", Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.NewRndSeed()
			vp.SetNeedsFullRender()
		})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch9/ss/README.md")
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
			Prompt: "Are you <i>sure</i> you want to quit and lose any unsaved params, weights, logs, etc?"}, gi.AddOk, gi.AddCancel,
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
			Prompt: "Are you <i>sure</i> you want to close the window?  This will Quit the App as well, losing all unsaved params, weights, logs, etc"}, gi.AddOk, gi.AddCancel,
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
		{"LesionNet", ki.Props{
			"desc": "Lesion the network using given type of lesion, and given proportion of neurons (0 < Proportion < 1)",
			"icon": "cut",
			"Args": ki.PropSlice{
				{"Lesion Type", ki.Props{}},
				{"Proportion", ki.Props{}},
			},
		}},
	},
}

func (ss *Sim) CmdArgs() {
	ss.NoGui = true
	var nogui bool
	var saveEpcLog bool
	var saveRunLog bool
	flag.StringVar(&ss.ParamSet, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.IntVar(&ss.MaxRuns, "runs", 1, "number of runs to do (note that MaxEpcs is in paramset)")
	flag.BoolVar(&ss.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.SaveWts, "wts", true, "if true, save final weights after each run")
	flag.BoolVar(&saveEpcLog, "epclog", true, "if true, save train epoch log to file")
	flag.BoolVar(&saveRunLog, "runlog", true, "if true, save run epoch log to file")
	flag.BoolVar(&nogui, "nogui", true, "if not passing any other args and want to run nogui, use nogui")
	flag.Parse()
	ss.Init()

	if ss.ParamSet != "" {
		fmt.Printf("Using ParamSet: %s\n", ss.ParamSet)
	}

	if saveEpcLog {
		var err error
		fnm := ss.LogFileName("epc")
		ss.TrnEpcFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TrnEpcFile = nil
		} else {
			fmt.Printf("Saving epoch log to: %v\n", fnm)
			defer ss.TrnEpcFile.Close()
		}
	}
	if saveRunLog {
		var err error
		fnm := ss.LogFileName("run")
		ss.RunFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.RunFile = nil
		} else {
			fmt.Printf("Saving run log to: %v\n", fnm)
			defer ss.RunFile.Close()
		}
	}
	if ss.SaveWts {
		fmt.Printf("Saving final weights per run\n")
	}
	fmt.Printf("Running %d Runs\n", ss.MaxRuns)
	ss.Train()
}
