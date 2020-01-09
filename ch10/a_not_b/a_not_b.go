// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
a_not_b explores how the development of PFC active maintenance abilities can help
o make behavior more flexible, in the sense that it can rapidly shift with changes
in the environment. The development of flexibility has been extensively explored
in the context of Piaget's famous A-not-B task, where a toy is first hidden several
times in one hiding location (A), and then hidden in a new location (B). Depending
on various task parameters, young kids reliably reach back at A instead of updating
to B.
*/
package main

import (
	"bytes"
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

// Delays is delay case to use
type Delays int32

//go:generate stringer -type=Delays

var KiT_Delays = kit.Enums.AddEnum(DelaysN, kit.NotBitFlag, nil)

func (ev Delays) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *Delays) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

const (
	Delay3 Delays = iota
	Delay5
	Delay1
	DelaysN
)

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "lower lrate, uniform init, fixed LLrn, no MLrn",
				Params: params.Params{
					"Prjn.Learn.Norm.On":      "false",
					"Prjn.Learn.Momentum.On":  "false",
					"Prjn.Learn.WtBal.On":     "false",
					"Prjn.Learn.Lrate":        "0.02",
					"Prjn.WtInit.Mean":        "0.3",
					"Prjn.WtInit.Var":         "0",
					"Prjn.Learn.XCal.SetLLrn": "true",
					"Prjn.Learn.XCal.LLrn":    "1",
					"Prjn.Learn.XCal.MLrn":    "0",
				}},
			{Sel: "Layer", Desc: "high inhibition, layer act avg",
				Params: params.Params{
					"Layer.Act.XX1.Gain":       "20",
					"Layer.Act.Dt.VmTau":       "10",
					"Layer.Act.Init.Decay":     "0",
					"Layer.Learn.AvgL.Gain":    "0.6", // critical params here
					"Layer.Learn.AvgL.Init":    "0.2",
					"Layer.Learn.AvgL.Min":     "0.1",
					"Layer.Learn.AvgL.Tau":     "100",
					"Layer.Inhib.Layer.Gi":     "1.3",
					"Layer.Inhib.Layer.FB":     "0.5",
					"Layer.Inhib.ActAvg.Init":  "0.05",
					"Layer.Inhib.ActAvg.Fixed": "true",
				}},
			{Sel: "#Reach", Desc: "higher gain, inhib",
				Params: params.Params{
					"Layer.Act.XX1.Gain":   "100",
					"Layer.Inhib.Layer.Gi": "1.7",
					"Layer.Inhib.Layer.FB": "1",
				}},
			{Sel: "#GazeExpect", Desc: "higher inhib",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "2",
				}},
			{Sel: "#LocationToHidden", Desc: "strong",
				Params: params.Params{
					"Prjn.WtScale.Abs": "1.5",
				}},
			{Sel: "#CoverToHidden", Desc: "strong, slow",
				Params: params.Params{
					"Prjn.WtScale.Abs": "1.5",
					"Prjn.Learn.Lrate": "0.005",
				}},
			{Sel: "#ToyToHidden", Desc: "strong, slow",
				Params: params.Params{
					"Prjn.WtScale.Abs": "1.5",
					"Prjn.Learn.Lrate": "0.005",
				}},
			{Sel: "#HiddenToHidden", Desc: "recurrent",
				Params: params.Params{
					"Prjn.WtInit.Mean": "0.4",
					"Prjn.Learn.Learn": "false",
				}},
			{Sel: "#HiddenToGazeExpect", Desc: "strong",
				Params: params.Params{
					"Prjn.WtScale.Abs": "1.5",
				}},
			{Sel: "#GazeExpectToGazeExpect", Desc: "recurrent",
				Params: params.Params{
					"Prjn.WtInit.Mean": "0.3",
					"Prjn.Learn.Learn": "false",
				}},
			{Sel: "#HiddenToReach", Desc: "no learn to reach",
				Params: params.Params{
					"Prjn.Learn.Learn": "false",
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
	Delay       Delays            `desc:"which delay to use -- pres Init when changing"`
	RecurrentWt float32           `def:"0.4" step:"0.01" desc:"strength of recurrent weight in Hidden layer from each unit back to self"`
	Net         *leabra.Network   `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Delay3Pats  *etable.Table     `view:"no-inline" desc:"delay 3 patterns"`
	Delay5Pats  *etable.Table     `view:"no-inline" desc:"delay 5 patterns"`
	Delay1Pats  *etable.Table     `view:"no-inline" desc:"delay 1 patterns"`
	TrnTrlLog   *etable.Table     `view:"no-inline" desc:"testing trial-level log data"`
	Params      params.Sets       `view:"no-inline" desc:"full collection of param sets"`
	MaxRuns     int               `desc:"maximum number of model runs to perform"`
	MaxEpcs     int               `desc:"maximum number of epochs to run per model run"`
	TrainEnv    env.FixedTable    `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	Time        leabra.Time       `desc:"leabra timing parameters and state"`
	ViewOn      bool              `desc:"whether to update the network view while running"`
	TrainUpdt   leabra.TimeScales `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TstRecLays  []string          `desc:"names of layers to record activations etc of during testing"`

	// statistics: note use float64 as that is best for etable.Table
	PrvGpName   string                      `view:"-" desc:"previous group name"`
	Win         *gi.Window                  `view:"-" desc:"main GUI window"`
	NetView     *netview.NetView            `view:"-" desc:"the network viewer"`
	ToolBar     *gi.ToolBar                 `view:"-" desc:"the master toolbar"`
	TrnTrlTable *etview.TableView           `view:"-" desc:"the train trial table view"`
	TrnTrlPlot  *eplot.Plot2D               `view:"-" desc:"the train trial plot"`
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
	ss.Defaults()
	ss.Net = &leabra.Network{}
	ss.Delay3Pats = &etable.Table{}
	ss.Delay5Pats = &etable.Table{}
	ss.Delay1Pats = &etable.Table{}
	ss.TrnTrlLog = &etable.Table{}
	ss.Params = ParamSets
	ss.RndSeed = 1
	ss.ViewOn = true
	ss.TrainUpdt = leabra.Quarter
	ss.TstRecLays = []string{"Location", "Cover", "Toy", "Hidden", "GazeExpect", "Reach"}
	ss.Time.CycPerQtr = 4 // key!
}

func (ss *Sim) Defaults() {
	ss.RecurrentWt = 0.4
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.OpenPats()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigTrnTrlLog(ss.TrnTrlLog)
}

func (ss *Sim) ConfigEnv() {
	if ss.MaxRuns == 0 { // allow user override
		ss.MaxRuns = 1
	}
	if ss.MaxEpcs == 0 { // allow user override
		ss.MaxEpcs = 1
	}

	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	switch ss.Delay {
	case Delay3:
		ss.TrainEnv.Table = etable.NewIdxView(ss.Delay3Pats)
	case Delay5:
		ss.TrainEnv.Table = etable.NewIdxView(ss.Delay5Pats)
	case Delay1:
		ss.TrainEnv.Table = etable.NewIdxView(ss.Delay1Pats)
	}
	ss.TrainEnv.Sequential = true
	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually

	ss.TrainEnv.Init(0)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.InitName(net, "AnotB")
	loc := net.AddLayer2D("Location", 1, 3, emer.Input)
	cvr := net.AddLayer2D("Cover", 1, 2, emer.Input)
	toy := net.AddLayer2D("Toy", 1, 2, emer.Input)
	hid := net.AddLayer2D("Hidden", 1, 3, emer.Hidden)
	gze := net.AddLayer2D("GazeExpect", 1, 3, emer.Compare)
	rch := net.AddLayer2D("Reach", 1, 3, emer.Compare)

	full := prjn.NewFull()
	self := prjn.NewOneToOne()
	net.ConnectLayers(loc, hid, full, emer.Forward)
	net.ConnectLayers(cvr, hid, full, emer.Forward)
	net.ConnectLayers(toy, hid, full, emer.Forward)
	net.ConnectLayers(hid, hid, self, emer.Lateral)
	net.ConnectLayers(hid, gze, full, emer.Forward)
	net.ConnectLayers(hid, rch, full, emer.Forward)
	net.ConnectLayers(gze, gze, self, emer.Lateral)

	cvr.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Location", YAlign: relpos.Front, Space: 1})
	toy.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Cover", YAlign: relpos.Front, Space: 1})
	hid.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Cover", YAlign: relpos.Front, XAlign: relpos.Left, YOffset: 1, XOffset: -1})
	gze.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Hidden", YAlign: relpos.Front, XAlign: relpos.Left, XOffset: -4})
	rch.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "GazeExpect", YAlign: relpos.Front, Space: 4})

	net.Defaults()
	ss.SetParams("Network", false) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	ss.InitWts(ss.Net)
}

func (ss *Sim) InitWts(net *leabra.Network) {
	net.InitWts()
	hid := ss.Net.LayerByName("Hidden").(leabra.LeabraLayer).AsLeabra()
	fmloc := hid.RcvPrjns.SendName("Location").(leabra.LeabraPrjn).AsLeabra()
	gze := ss.Net.LayerByName("GazeExpect").(leabra.LeabraLayer).AsLeabra()
	hidgze := gze.RcvPrjns.SendName("Hidden").(leabra.LeabraPrjn).AsLeabra()
	rch := ss.Net.LayerByName("Reach").(leabra.LeabraLayer).AsLeabra()
	hidrch := rch.RcvPrjns.SendName("Hidden").(leabra.LeabraPrjn).AsLeabra()
	for i := 0; i < 3; i++ {
		fmloc.SetSynVal("Wt", i, i, 0.7)
		hidgze.SetSynVal("Wt", i, i, 0.7)
		hidrch.SetSynVal("Wt", i, i, 0.7)
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	rand.Seed(ss.RndSeed)
	ss.StopNow = false
	ss.SetParams("", false) // all sheets
	ss.ConfigEnv()
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
	return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%v\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Trial.Cur, ss.Time.Cycle, ss.TrainEnv.TrialName.Cur)
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

	// update prior weight changes at start, so any DWt values remain visible at end
	// you might want to do this less frequently to achieve a mini-batch update
	// in which case, move it out to the TrainTrial method where the relevant
	// counters are being dealt with.
	ss.Net.WtFmDWt()

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

	lays := []string{"Location", "Cover", "Toy", "Reach"}
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

	rch := ss.Net.LayerByName("Reach").(leabra.LeabraLayer).AsLeabra()
	if strings.Contains(ss.TrainEnv.TrialName.Cur, "choice") {
		rch.SetType(emer.Compare)
	} else {
		rch.SetType(emer.Input)
	}

	if ss.TrainEnv.GroupName.Cur != ss.PrvGpName { // init at start of new group
		ss.Net.InitActs()
		ss.PrvGpName = ss.TrainEnv.GroupName.Cur
	}
	train := true
	if strings.Contains(ss.TrainEnv.TrialName.Cur, "delay") {
		train = false // don't learn on delay trials
	}

	ss.ApplyInputs(&ss.TrainEnv)
	ss.AlphaCyc(train)
	ss.TrialStats(true) // accumulate
	ss.LogTrnTrl(ss.TrnTrlLog, ss.TrainEnv.Trial.Cur, ss.TrainEnv.TrialName.Cur)
}

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	run := ss.TrainEnv.Run.Cur
	ss.TrainEnv.Init(run)
	ss.Time.Reset()
	ss.InitWts(ss.Net)
	ss.InitStats()
	ss.TrnTrlLog.SetNumRows(len(ss.TrainEnv.Order))
	ss.PrvGpName = ""
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
func (ss *Sim) TrialStats(accum bool) (sse, avgsse, cosdiff float64) {
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
		hid := ss.Net.LayerByName("Hidden").(leabra.LeabraLayer).AsLeabra()
		fmhid := hid.RcvPrjns.SendName("Hidden").(leabra.LeabraPrjn).AsLeabra()
		fmhid.WtInit.Mean = float64(ss.RecurrentWt)
	}

	if sheet == "" || sheet == "Sim" {
		simp, ok := pset.Sheets["Sim"]
		if ok {
			simp.Apply(ss, setMsg)
		}
	}
	// note: if you have more complex environments with parameters, definitely add
	// sheets for them, e.g., "TrainEnv"..
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
	// patgen.ReshapeCppFile(ss.Delay3Pats, "ABInput_Delay3.dat", "a_not_b_delay3.tsv") // one-time reshape
	// patgen.ReshapeCppFile(ss.Delay5Pats, "ABInput_Delay5.dat", "a_not_b_delay5.tsv") // one-time reshape
	// patgen.ReshapeCppFile(ss.Delay1Pats, "ABInput_Delay1.dat", "a_not_b_delay1.tsv") // one-time reshape
	ss.OpenPatAsset(ss.Delay3Pats, "a_not_b_delay3.tsv", "AnotB Delay=3", "AnotB input patterns")
	ss.OpenPatAsset(ss.Delay5Pats, "a_not_b_delay5.tsv", "AnotB Delay=5", "AnotB input patterns")
	ss.OpenPatAsset(ss.Delay1Pats, "a_not_b_delay1.tsv", "AnotB Delay=1", "AnotB input patterns")
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
//  TrnTrlLog

// LogTrnTrl adds data from current trial to the TrnTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTrnTrl(dt *etable.Table, trl int, trlnm string) {
	row := trl
	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("Group", row, ss.TrainEnv.GroupName.Cur)
	dt.SetCellString("TrialName", row, trlnm)

	for _, lnm := range ss.TstRecLays {
		tsr := ss.ValsTsr(lnm)
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		ly.UnitValsTensor(tsr, "ActM") // get minus phase act
		dt.SetCellTensor(lnm, row, tsr)
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TrnTrlPlot.GoUpdate()
	ss.TrnTrlTable.UpdateSig()
}

func (ss *Sim) ConfigTrnTrlLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TrainEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Trial", etensor.INT64, nil, nil},
		{"Group", etensor.STRING, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
	}
	for _, lnm := range ss.TstRecLays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		sch = append(sch, etable.Column{lnm, etensor.FLOAT64, ly.Shp.Shp, nil})
	}
	dt.SetFromSchema(sch, nt)

	// gze := dt.ColByName("GazeExpect")
	// gze.SetMetaData("min", "0")
	// rch := dt.ColByName("Reach")
	// rch.SetMetaData("min", "0")
	// hid := dt.ColByName("Hidden")
	// hid.SetMetaData("min", "0")
}

func (ss *Sim) ConfigTrnTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "A not B Train Trial Plot"
	plt.Params.XAxisCol = "TrialName"
	plt.Params.Type = eplot.Bar
	plt.SetTable(dt) // this sets defaults so set params after
	plt.Params.BarWidth = 2
	plt.Params.XAxisRot = 90
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Group", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

	for _, lnm := range ss.TstRecLays {
		if lnm == "Reach" {
			cp := plt.SetColParams(lnm, eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
			cp.TensorIdx = -1 // plot all
		} else {
			cp := plt.SetColParams(lnm, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
			cp.TensorIdx = -1 // plot all
		}
	}
	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.ViewDefaults()
	nv.Scene().Camera.Pose.Pos.Set(0.1, 1.8, 3.5)
	nv.Scene().Camera.LookAt(mat32.Vec3{0.1, 0.15, 0}, mat32.Vec3{0, 1, 0})
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("a_not_b")
	gi.SetAppAbout(`explores how the development of PFC active maintenance abilities can help to make behavior more flexible, in the sense that it can rapidly shift with changes in the environment. The development of flexibility has been extensively explored in the context of Piaget's famous A-not-B task, where a toy is first hidden several times in one hiding location (A), and then hidden in a new location (B). Depending on various task parameters, young kids reliably reach back at A instead of updating to B. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch10/a_not_b/README.md">README.md on GitHub</a>.</p>`)

	win := gi.NewMainWindow("a_not_b", "A not B", width, height)
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

	tabv := tv.AddNewTab(etview.KiT_TableView, "TrnTrlTable").(*etview.TableView)
	tabv.SetTable(ss.TrnTrlLog, nil)
	ss.TrnTrlTable = tabv

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TrnTrlPlot").(*eplot.Plot2D)
	ss.TrnTrlPlot = ss.ConfigTrnTrlPlot(plt, ss.TrnTrlLog)

	split.SetSplits(.3, .7)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Train", Icon: "run", Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training.",
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

	tbar.AddSeparator("misc")

	tbar.AddAction(gi.ActOpts{Label: "New Seed", Icon: "new", Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.NewRndSeed()
		})

	tbar.AddAction(gi.ActOpts{Label: "Defaults", Icon: "update", Tooltip: "Restore initial default parameters.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Defaults()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch10/a_not_b/README.md")
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
	},
}

func mainrun() {
	TheSim.New()
	TheSim.Config()

	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}
