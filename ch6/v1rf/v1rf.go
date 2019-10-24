// Copyright (c) 2019, The Emergent Authors. All rights reserved.
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
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/etview" // include to get gui views
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
			{Sel: "Prjn", Desc: "no extra learning factors, hebbian learning",
				Params: params.Params{
					"Prjn.Learn.Norm.On":      "false",
					"Prjn.Learn.Momentum.On":  "false",
					"Prjn.Learn.WtBal.On":     "false",
					"Prjn.Learn.XCal.MLrn":    "0", // pure hebb
					"Prjn.Learn.XCal.SetLLrn": "true",
					"Prjn.Learn.XCal.LLrn":    "1",
					"Prjn.Learn.WtSig.Gain":   "1", // key: more graded weights
				}},
			{Sel: "Layer", Desc: "needs some special inhibition and learning params",
				Params: params.Params{
					"Layer.Learn.AvgL.Gain":   "1", // this is critical! much lower
					"Layer.Learn.AvgL.Min":    "0.01",
					"Layer.Learn.AvgL.Init":   "0.2",
					"Layer.Inhib.Layer.Gi":    "2",
					"Layer.Inhib.Layer.FBTau": "3",
					"Layer.Inhib.ActAvg.Init": "0.2",
					"Layer.Act.Gbar.L":        "0.1",
					"Layer.Act.Noise.Dist":    "Gaussian",
					"Layer.Act.Noise.Var":     "0.02",
					"Layer.Act.Noise.Type":    "GeNoise",
					"Layer.Act.Noise.Fixed":   "false",
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.2",
				}},
			{Sel: ".ExciteLateral", Desc: "lateral excitatory connection",
				Params: params.Params{
					"Prjn.WtInit.Mean": ".5",
					"Prjn.WtInit.Var":  "0",
					"Prjn.WtInit.Sym":  "false",
					"Prjn.WtScale.Rel": "0.2",
				}},
			{Sel: ".InhibLateral", Desc: "lateral inhibitory connection",
				Params: params.Params{
					"Prjn.WtInit.Mean": "0",
					"Prjn.WtInit.Var":  "0",
					"Prjn.WtInit.Sym":  "false",
					"Prjn.WtScale.Abs": "0.2",
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
	ExcitLateralScale float32           `def:"0.2" desc:"excitatory lateral (recurrent) WtScale.Rel value"`
	InhibLateralScale float32           `def:"0.2" desc:"inhibitory lateral (recurrent) WtScale.Abs value"`
	ExcitLateralLearn bool              `def:"true" desc:"do excitatory lateral (recurrent) connections learn?"`
	Net               *leabra.Network   `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Probes            *etable.Table     `view:"no-inline" desc:"probe inputs"`
	TrnEpcLog         *etable.Table     `view:"no-inline" desc:"training epoch-level log data"`
	TstEpcLog         *etable.Table     `view:"no-inline" desc:"testing epoch-level log data"`
	TstTrlLog         *etable.Table     `view:"no-inline" desc:"testing trial-level log data"`
	RunLog            *etable.Table     `view:"no-inline" desc:"summary log of each run"`
	RunStats          *etable.Table     `view:"no-inline" desc:"aggregate stats on all runs"`
	Params            params.Sets       `view:"no-inline" desc:"full collection of param sets"`
	V1onWts           *etensor.Float32  `view:"-" desc:"weights from input to V1 layer"`
	V1offWts          *etensor.Float32  `view:"-" desc:"weights from input to V1 layer"`
	V1Wts             *etensor.Float32  `view:"no-inline" desc:"net on - off weights from input to V1 layer"`
	MaxRuns           int               `desc:"maximum number of model runs to perform"`
	MaxEpcs           int               `desc:"maximum number of epochs to run per model run"`
	MaxTrls           int               `desc:"maximum number of training trials per epoch"`
	NZeroStop         int               `desc:"if a positive number, training will stop after this many epochs with zero SSE"`
	TrainEnv          ImgEnv            `desc:"Training environment -- visual images"`
	TestEnv           env.FixedTable    `desc:"Testing environment -- manages iterating over testing"`
	Time              leabra.Time       `desc:"leabra timing parameters and state"`
	ViewOn            bool              `desc:"whether to update the network view while running"`
	TrainUpdt         leabra.TimeScales `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt          leabra.TimeScales `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	LayStatNms        []string          `desc:"names of layers to collect more detailed stats on (avg act, etc)"`

	// statistics: note use float64 as that is best for etable.Table
	Win         *gi.Window                  `view:"-" desc:"main GUI window"`
	NetView     *netview.NetView            `view:"-" desc:"the network viewer"`
	ToolBar     *gi.ToolBar                 `view:"-" desc:"the master toolbar"`
	CurImgGrid  *etview.TensorGrid          `view:"-" desc:"the current image grid view"`
	WtsGrid     *etview.TensorGrid          `view:"-" desc:"the weights grid view"`
	TrnEpcPlot  *eplot.Plot2D               `view:"-" desc:"the training epoch plot"`
	TstEpcPlot  *eplot.Plot2D               `view:"-" desc:"the testing epoch plot"`
	TstTrlPlot  *eplot.Plot2D               `view:"-" desc:"the test-trial plot"`
	RunPlot     *eplot.Plot2D               `view:"-" desc:"the run plot"`
	TrnEpcFile  *os.File                    `view:"-" desc:"log file"`
	RunFile     *os.File                    `view:"-" desc:"log file"`
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
	ss.ExcitLateralScale = 0.2
	ss.InhibLateralScale = 0.2
	ss.ExcitLateralLearn = true
	ss.Net = &leabra.Network{}
	ss.Probes = &etable.Table{}
	ss.TrnEpcLog = &etable.Table{}
	ss.TstEpcLog = &etable.Table{}
	ss.TstTrlLog = &etable.Table{}
	ss.V1onWts = &etensor.Float32{}
	ss.V1offWts = &etensor.Float32{}
	ss.V1Wts = &etensor.Float32{}
	ss.RunLog = &etable.Table{}
	ss.RunStats = &etable.Table{}
	ss.Params = ParamSets
	ss.RndSeed = 1
	ss.ViewOn = true
	ss.TrainUpdt = leabra.AlphaCycle
	ss.TestUpdt = leabra.Cycle
	ss.LayStatNms = []string{"V1"}
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
		ss.MaxEpcs = 100
		ss.NZeroStop = -1
	}
	if ss.MaxTrls == 0 { // allow user override
		ss.MaxTrls = 100
	}

	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	ss.TrainEnv.Defaults()
	ss.TrainEnv.ImageFiles = []string{"v1rf_img1.jpg", "v1rf_img2.jpg", "v1rf_img3.jpg", "v1rf_img4.jpg"}
	ss.TrainEnv.OpenImagesAsset()
	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually
	ss.TrainEnv.Trial.Max = ss.MaxTrls

	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Dsc = "testing (probe) params and state"
	ss.TestEnv.Table = etable.NewIdxView(ss.Probes)
	ss.TestEnv.Sequential = true
	ss.TestEnv.Validate()

	ss.TrainEnv.Init(0)
	ss.TestEnv.Init(0)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.InitName(net, "V1Rf")
	lgnOn := net.AddLayer2D("LGNon", 12, 12, emer.Input)
	lgnOff := net.AddLayer2D("LGNoff", 12, 12, emer.Input)
	v1 := net.AddLayer2D("V1", 14, 14, emer.Hidden)

	full := prjn.NewFull()
	net.ConnectLayers(lgnOn, v1, full, emer.Forward)
	net.ConnectLayers(lgnOff, v1, full, emer.Forward)

	circ := prjn.NewCircle()
	circ.Radius = 4
	circ.Sigma = .75

	rec := net.ConnectLayers(v1, v1, circ, emer.Lateral)
	rec.SetClass("ExciteLateral")

	inh := net.ConnectLayers(v1, v1, full, emer.Inhib)
	inh.SetClass("InhibLateral")

	lgnOff.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "LGNon", YAlign: relpos.Front, Space: 2})
	v1.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "LGNon", XAlign: relpos.Left, YAlign: relpos.Front, XOffset: 5, Space: 2})

	net.Defaults()
	ss.SetParams("Network", false) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	ss.InitWts(net)
}

func (ss *Sim) InitWts(net *leabra.Network) {
	// set scales after building but before InitWts
	v1 := net.LayerByName("V1")
	v1v1 := v1.RecvPrjns().SendName("V1").(*leabra.Prjn) // first one is excite
	pat := v1v1.Pattern().(*prjn.Circle)
	v1v1.SetScalesFunc(pat.GaussWts)
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
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%v\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Trial.Cur, ss.Time.Cycle, ss.TrainEnv.String())
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

	lays := []string{"LGNon", "LGNoff"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(*leabra.Layer)
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

	ss.ApplyInputs(&ss.TrainEnv)
	ss.AlphaCyc(true)   // train
	ss.TrialStats(true) // accumulate
	if ss.CurImgGrid != nil {
		ss.CurImgGrid.UpdateSig()
	}
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
	ss.Time.Reset()
	ss.InitWts(ss.Net)
	ss.InitStats()
	ss.TrnEpcLog.SetNumRows(0)
	ss.TstEpcLog.SetNumRows(0)
	ss.NeedsNewRun = false
}

// InitStats initializes all the statistics, especially important for the
// cumulative epoch stats -- called at start of new run
func (ss *Sim) InitStats() {
}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func (ss *Sim) TrialStats(accum bool) {
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

// SaveWts saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWts(filename gi.FileName) {
	ss.Net.SaveWtsJSON(filename)
}

// OpenRec2Wts opens trained weights w/ rec=0.2
func (ss *Sim) OpenRec2Wts() {
	ab, err := Asset("v1rf_rec2.wts") // embedded in executable
	if err != nil {
		log.Println(err)
	}
	ss.Net.ReadWtsJSON(bytes.NewBuffer(ab))
	// ss.Net.OpenWtsJSON("v1rf_rec2.wts.gz")
}

// OpenRec05Wts opens trained weights w/ rec=0.05
func (ss *Sim) OpenRec05Wts() {
	ab, err := Asset("v1rf_rec05.wts") // embedded in executable
	if err != nil {
		log.Println(err)
	}
	ss.Net.ReadWtsJSON(bytes.NewBuffer(ab))
	// ss.Net.OpenWtsJSON("v1rf_rec05.wts.gz")
}

func (ss *Sim) V1RFs() {
	onVals := ss.V1onWts.Values
	offVals := ss.V1offWts.Values
	netVals := ss.V1Wts.Values
	on := ss.Net.LayerByName("LGNon").(*leabra.Layer)
	off := ss.Net.LayerByName("LGNoff").(*leabra.Layer)
	isz := on.Shape().Len()
	v1 := ss.Net.LayerByName("V1").(*leabra.Layer)
	ysz := v1.Shape().Dim(0)
	xsz := v1.Shape().Dim(1)
	for y := 0; y < ysz; y++ {
		for x := 0; x < xsz; x++ {
			ui := (y*xsz + x)
			ust := ui * isz
			onvls := onVals[ust : ust+isz]
			offvls := offVals[ust : ust+isz]
			netvls := netVals[ust : ust+isz]
			on.SendPrjnVals(&onvls, "Wt", v1, ui)
			off.SendPrjnVals(&offvls, "Wt", v1, ui)
			for ui := 0; ui < isz; ui++ {
				netvls[ui] = 1.5 * (onvls[ui] - offvls[ui])
			}
		}
	}
	if ss.WtsGrid != nil {
		ss.WtsGrid.UpdateSig()
	}
}

func (ss *Sim) ConfigWts(dt *etensor.Float32) {
	dt.SetShape([]int{14, 14, 12, 12}, nil, nil)
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

	nt := ss.Net
	v1 := nt.LayerByName("V1").(*leabra.Layer)
	elat := v1.RcvPrjns[2].(*leabra.Prjn)
	elat.WtScale.Rel = ss.ExcitLateralScale
	elat.Learn.Learn = ss.ExcitLateralLearn
	ilat := v1.RcvPrjns[3].(*leabra.Prjn)
	ilat.WtScale.Abs = ss.InhibLateralScale

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

func (ss *Sim) OpenPats() {
	// patgen.ReshapeCppFile(ss.Probes, "ProbeInputData.dat", "probes.csv") // one-time reshape
	dt := ss.Probes
	dt.SetMetaData("name", "Probes")
	dt.SetMetaData("desc", "Probe inputs for testing")
	// err := dt.OpenCSV("easy.dat", etable.Tab)
	ab, err := Asset("probes.tsv") // embedded in executable
	if err != nil {
		log.Println(err)
	}
	err = dt.ReadCSV(bytes.NewBuffer(ab), etable.Tab)
	if err != nil {
		log.Println(err)
	}
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

	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value
	// nt := float64(ss.TrainEnv.Trial.Max)

	ss.V1RFs()

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(*leabra.Layer)
		dt.SetCellFloat(ly.Nm+" ActAvg", row, float64(ly.Pools[0].ActAvg.ActPAvgEff))
	}

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
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " ActAvg", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
	ss.ConfigWts(ss.V1onWts)
	ss.ConfigWts(ss.V1offWts)
	ss.ConfigWts(ss.V1Wts)
}

func (ss *Sim) ConfigTrnEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "V1 Receptive Field Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActAvg", true, true, 0, true, .5)
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
	dt.SetCellString("TrialName", row, ss.TestEnv.TrialName)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(*leabra.Layer)
		dt.SetCellFloat(ly.Nm+" ActM.Avg", row, float64(ly.Pools[0].ActM.Avg))
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
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " ActM.Avg", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "V1 Receptive Field Test Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("Trial", false, true, 0, false, 0)
	plt.SetColParams("TrialName", false, true, 0, false, 0)

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActM.Avg", false, true, 0, true, .5)
	}
	return plt
}

//////////////////////////////////////////////
//  TstEpcLog

func (ss *Sim) LogTstEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	// trl := ss.TstTrlLog
	// tix := etable.NewIdxView(trl)
	epc := ss.TrainEnv.Epoch.Prv // ?

	// note: this shows how to use agg methods to compute summary data from another
	// data table, instead of incrementing on the Sim
	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))

	// note: essential to use Go version of update when called from another goroutine
	ss.TstEpcPlot.GoUpdate()
}

func (ss *Sim) ConfigTstEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstEpcLog")
	dt.SetMetaData("desc", "Summary stats for testing trials")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	dt.SetFromSchema(etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
	}, 0)
}

func (ss *Sim) ConfigTstEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "V1 Receptive Field Testing Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
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
	nlast := 10
	if nlast > epcix.Len()-1 {
		nlast = epcix.Len() - 1
	}
	epcix.Idxs = epcix.Idxs[epcix.Len()-nlast-1:]

	// params := ss.Params.Name
	params := "params"

	dt.SetCellFloat("Run", row, float64(run))
	dt.SetCellString("Params", row, params)

	// runix := etable.NewIdxView(dt)
	// spl := split.GroupBy(runix, []string{"Params"})
	// split.Desc(spl, "FirstZero")
	// split.Desc(spl, "PctCor")
	// ss.RunStats = spl.AggsToTable(false)

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

	dt.SetFromSchema(etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Params", etensor.STRING, nil, nil},
	}, 0)
}

func (ss *Sim) ConfigRunPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "V1 Receptive Field Run Plot"
	plt.Params.XAxisCol = "Run"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	cam := &(nv.Scene().Camera)
	cam.Pose.Pos.Set(0.0, 1.733, 2.3)
	cam.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
	// cam.Pose.Quat.SetFromAxisAngle(mat32.Vec3{-1, 0, 0}, 0.4077744)
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("v1rf")
	gi.SetAppAbout(`This simulation illustrates how self-organizing learning in response to natural images produces the oriented edge detector receptive field properties of neurons in primary visual cortex (V1). This provides insight into why the visual system encodes information in the way it does, while also providing an important test of the biological relevance of our computational models. See <a href="href="https://github.com/CompCogNeuro/sims/ch6/v1rf/README.md">README.md on GitHub</a>.</p>`)

	win := gi.NewWindow2D("v1rf", "V1 Receptive Fields", width, height, true)
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
	// nv.ViewDefaults()
	ss.ConfigNetView(nv)

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TrnEpcPlot").(*eplot.Plot2D)
	ss.TrnEpcPlot = ss.ConfigTrnEpcPlot(plt, ss.TrnEpcLog)

	tg := tv.AddNewTab(etview.KiT_TensorGrid, "Image").(*etview.TensorGrid)
	tg.SetStretchMax()
	ss.CurImgGrid = tg
	tg.SetTensor(&ss.TrainEnv.Vis.ImgTsr)

	tg = tv.AddNewTab(etview.KiT_TensorGrid, "V1 RFs").(*etview.TensorGrid)
	tg.SetStretchMax()
	ss.WtsGrid = tg
	tg.SetTensor(ss.V1Wts)

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

	tbar.AddSeparator("spec")

	tbar.AddAction(gi.ActOpts{Label: "Open Rec=.2 Wts", Icon: "updt", Tooltip: "Open weights trained with excitatory lateral (recurrent) con scale = .2.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.OpenRec2Wts()
	})

	tbar.AddAction(gi.ActOpts{Label: "Open Rec=.05 Wts", Icon: "updt", Tooltip: "Open weights trained with excitatory lateral (recurrent) con scale = .05.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.OpenRec05Wts()
	})

	tbar.AddAction(gi.ActOpts{Label: "V1 RFs", Icon: "file-image", Tooltip: "Update the V1 Receptive Field (Weights) plot in V1 RFs tab.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.V1RFs()
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
					idxs := ss.TestEnv.Table.RowsByString("Name", val, true, true) // contains, ignoreCase
					if len(idxs) == 0 {
						gi.PromptDialog(nil, gi.DlgOpts{Title: "Name Not Found", Prompt: "No patterns found containing: " + val}, true, false, nil, nil)
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

	tbar.AddAction(gi.ActOpts{Label: "Reset RunLog", Icon: "reset", Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.RunLog.SetNumRows(0)
			ss.RunPlot.Update()
		})

	tbar.AddSeparator("misc")

	tbar.AddAction(gi.ActOpts{Label: "New Seed", Icon: "new", Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.NewRndSeed()
		})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch6/v1rf/README.md")
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
		{"SaveWts", ki.Props{
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
