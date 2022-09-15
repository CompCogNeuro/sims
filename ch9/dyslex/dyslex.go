// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
dyslex simulates normal and disordered (dyslexic) reading performance in terms of a distributed representation of word-level knowledge across Orthography, Semantics, and Phonology. It is based on a model by Plaut and Shallice (1993). Note that this form of dyslexia is *aquired* (via brain lesions such as stroke) and not the more prevalent developmental variety.
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
	"github.com/emer/etable/agg"
	"github.com/emer/etable/clust"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/metric"
	"github.com/emer/etable/simat"
	"github.com/emer/etable/split"
	"github.com/emer/leabra/leabra"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// this is the stub main for gogi that calls our actual mainrun function, at end of file
func main() {
	gimain.Main(func() {
		mainrun()
	})
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// LesionTypes is the type of lesion
type LesionTypes int32

//go:generate stringer -type=LesionTypes

var KiT_LesionTypes = kit.Enums.AddEnum(LesionTypesN, kit.NotBitFlag, nil)

func (ev LesionTypes) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *LesionTypes) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

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

	LesionTypesN
)

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "no extra learning factors",
				Params: params.Params{
					"Prjn.Learn.Norm.On":     "false",
					"Prjn.Learn.Momentum.On": "false",
					"Prjn.Learn.WtBal.On":    "false",
					"Prjn.Learn.Lrate":       "0.04",
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
			{Sel: ".Back", Desc: "there is no back / forward direction here..",
				Params: params.Params{
					"Prjn.WtScale.Rel": "1",
				}},
			{Sel: ".Lateral", Desc: "self cons are weaker",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.3",
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
	Lesion       LesionTypes       `inactive:"+" desc:"type of lesion -- use Lesion button to lesion"`
	LesionProp   float32           `inactive:"+" desc:"proportion of neurons lesioned -- use Lesion button to lesion"`
	Net          *leabra.Network   `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	TrainPats    *etable.Table     `view:"no-inline" desc:"training patterns"`
	Semantics    *etable.Table     `view:"no-inline" desc:"properties of semantic features "`
	CloseOrthos  *etable.Table     `view:"no-inline" desc:"list of items that are close in orthography (for visual error scoring)"`
	CloseSems    *etable.Table     `view:"no-inline" desc:"list of items that are close in semantics (for semantic error scoring)"`
	TrnEpcLog    *etable.Table     `view:"no-inline" desc:"training epoch-level log data"`
	TstEpcLog    *etable.Table     `view:"no-inline" desc:"testing epoch-level log data"`
	TstTrlLog    *etable.Table     `view:"no-inline" desc:"testing trial-level log data"`
	TstStats     *etable.Table     `view:"no-inline" desc:"aggregate testing stats"`
	RunLog       *etable.Table     `view:"no-inline" desc:"summary log of each run"`
	RunStats     *etable.Table     `view:"no-inline" desc:"aggregate stats on all runs"`
	SemClustPlot *eplot.Plot2D     `view:"no-inline" desc:"semantics cluster plot"`
	Params       params.Sets       `view:"no-inline" desc:"full collection of param sets"`
	ParamSet     string            `view:"-" desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set -- can use multiple names separated by spaces (don't put spaces in ParamSet names!)"`
	MaxRuns      int               `desc:"maximum number of model runs to perform"`
	MaxEpcs      int               `desc:"maximum number of epochs to run per model run"`
	NZeroStop    int               `desc:"if a positive number, training will stop after this many epochs with zero SSE"`
	TrainEnv     env.FixedTable    `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	TestEnv      env.FixedTable    `desc:"Testing environment -- manages iterating over testing"`
	Time         leabra.Time       `desc:"leabra timing parameters and state"`
	ViewOn       bool              `desc:"whether to update the network view while running"`
	TrainUpdt    leabra.TimeScales `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt     leabra.TimeScales `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	TestInterval int               `desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`
	LayStatNms   []string          `desc:"names of layers to collect more detailed stats on (avg act, etc)"`

	// statistics: note use float64 as that is best for etable.Table
	TrlName      string  `inactive:"+" desc:"name of current input pattern"`
	TrlPhon      string  `inactive:"+" desc:"name of closest phonology pattern"`
	TrlPhonSSE   float64 `inactive:"+" desc:"SSE for closest phonology pattern -- > 3 = blend"`
	TrlConAbs    float64 `inactive:"+" desc:"0 = concrete, 1 = abstract"`
	TrlVisErr    float64 `inactive:"+" desc:"visual error -- close to similar other"`
	TrlSemErr    float64 `inactive:"+" desc:"semantic error -- close to similar other"`
	TrlVisSemErr float64 `inactive:"+" desc:"visual + semantic error -- close to similar other"`
	TrlBlendErr  float64 `inactive:"+" desc:"blend error"`
	TrlOtherErr  float64 `inactive:"+" desc:"some other error"`
	TrlErr       float64 `inactive:"+" desc:"1 if trial was error, 0 if correct -- based on *closest pattern* stat, not SSE"`
	TrlSSE       float64 `inactive:"+" desc:"current trial's sum squared error"`
	TrlAvgSSE    float64 `inactive:"+" desc:"current trial's average sum squared error"`
	TrlCosDiff   float64 `inactive:"+" desc:"current trial's cosine difference"`
	EpcSSE       float64 `inactive:"+" desc:"last epoch's total sum squared error"`
	EpcAvgSSE    float64 `inactive:"+" desc:"last epoch's average sum squared error (average over trials, and over units within layer)"`
	EpcPctErr    float64 `inactive:"+" desc:"last epoch's average TrlErr"`
	EpcPctCor    float64 `inactive:"+" desc:"1 - last epoch's average TrlErr"`
	EpcCosDiff   float64 `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	FirstZero    int     `inactive:"+" desc:"epoch at when SSE first went to zero"`
	NZero        int     `inactive:"+" desc:"number of epochs in a row with zero SSE"`

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
	ValsTsrs    map[string]*etensor.Float32 `view:"-" desc:"for holding layer values"`
	IsRunning   bool                        `view:"-" desc:"true if sim is running"`
	StopNow     bool                        `view:"-" desc:"flag to stop running"`
	NeedsNewRun bool                        `view:"-" desc:"flag to initialize NewRun if last one finished"`
	RndSeed     int64                       `inactive:"+" desc:"the current random seed"`
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
	ss.Semantics = &etable.Table{}
	ss.CloseOrthos = &etable.Table{}
	ss.CloseSems = &etable.Table{}
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
	ss.TestInterval = 10
	ss.LayStatNms = []string{"OShidden"}
}

func (ss *Sim) Defaults() {
	ss.Lesion = NoLesion
	ss.LesionProp = 0
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
		ss.MaxEpcs = 250
		ss.NZeroStop = -1
	}

	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	ss.TrainEnv.Table = etable.NewIdxView(ss.TrainPats)
	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually

	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Dsc = "testing params and state"
	ss.TestEnv.Table = etable.NewIdxView(ss.TrainPats)
	ss.TestEnv.Sequential = true
	ss.TestEnv.Validate()

	ss.TrainEnv.Init(0)
	ss.TestEnv.Init(0)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.InitName(net, "Dyslex")
	ort := net.AddLayer2D("Orthography", 6, 8, emer.Input)
	oph := net.AddLayer2D("OPhidden", 7, 7, emer.Hidden)
	phn := net.AddLayer4D("Phonology", 1, 7, 7, 2, emer.Target)
	osh := net.AddLayer2D("OShidden", 10, 7, emer.Hidden)
	sph := net.AddLayer2D("SPhidden", 10, 7, emer.Hidden)
	sem := net.AddLayer2D("Semantics", 10, 12, emer.Target)

	full := prjn.NewFull()
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

	oph.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Orthography", YAlign: relpos.Front, Space: 1})
	phn.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "OPhidden", YAlign: relpos.Front, Space: 1})
	osh.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Orthography", YAlign: relpos.Front, XAlign: relpos.Left, XOffset: 4})
	sph.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Phonology", YAlign: relpos.Front, XAlign: relpos.Left, XOffset: 2})
	sem.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "OShidden", YAlign: relpos.Front, XAlign: relpos.Left, XOffset: 4})

	net.Defaults()
	ss.SetParams("Network", false) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.InitWts()
}

// LesionNet does lesion of network with given proportion of neurons damaged
// 0 < prop < 1.
func (ss *Sim) LesionNet(les LesionTypes, prop float32) {
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
		ss.LesionNetImpl(net, les, prop)
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
		net.LayerByName("OShidden").SetOff(true)
		net.LayerByName("Semantics").SetOff(true)
		net.LayerByName("SPhidden").SetOff(true)
	case DirectFull:
		net.LayerByName("OPhidden").SetOff(true)
	case OShidden:
		net.LayerByName("OShidden").(leabra.LeabraLayer).AsLeabra().LesionNeurons(prop)
	case SPhidden:
		net.LayerByName("SPhidden").(leabra.LeabraLayer).AsLeabra().LesionNeurons(prop)
	case OPhidden:
		net.LayerByName("OPhidden").(leabra.LeabraLayer).AsLeabra().LesionNeurons(prop)
	case OShidDirectFull:
		net.LayerByName("OPhidden").SetOff(true)
		net.LayerByName("OShidden").(leabra.LeabraLayer).AsLeabra().LesionNeurons(prop)
	case SPhidDirectFull:
		net.LayerByName("OPhidden").SetOff(true)
		net.LayerByName("SPhidden").(leabra.LeabraLayer).AsLeabra().LesionNeurons(prop)
	case OPhidSemanticsFull:
		net.LayerByName("OShidden").SetOff(true)
		net.LayerByName("Semantics").SetOff(true)
		net.LayerByName("SPhidden").SetOff(true)
		net.LayerByName("OPhidden").(leabra.LeabraLayer).AsLeabra().LesionNeurons(prop)
	}
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
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}

	ss.Net.AlphaCycInit(train)
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdt {
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
			case viewUpdt == leabra.Cycle:
				ss.UpdateView(train, ss.Time.Cycle)
			case viewUpdt <= leabra.Quarter:
				ss.UpdateView(train, -1)
			case viewUpdt == leabra.Phase:
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
	if ss.ViewOn && viewUpdt == leabra.AlphaCycle {
		ss.UpdateView(train, -1)
	}
}

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"Orthography", "Semantics", "Phonology"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := en.State(ly.Nm)
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
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		if i == layno {
			ly.SetType(emer.Input)
		} else {
			if test {
				ly.SetType(emer.Compare)
			} else {
				ly.SetType(emer.Target)
			}
		}
	}
}

// SetRndInputLayer sets one of 3 visible layers as input at random
func (ss *Sim) SetRndInputLayer() {
	ss.SetInputLayer(rand.Intn(3))
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
			ss.UpdateView(true, -1)
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

	// note: type must be in place before apply inputs
	if ss.TrainEnv.Epoch.Cur < ss.MaxEpcs-10 {
		ss.SetRndInputLayer()
	} else {
		ss.SetInputLayer(0) // final training on Ortho reading
	}
	ss.ApplyInputs(&ss.TrainEnv)
	ss.AlphaCyc(true)                              // train
	ss.TrialStats(true, ss.TrainEnv.TrialName.Cur) // accumulate
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
	ss.TrlVisErr = 0
	ss.TrlSemErr = 0
	ss.TrlVisSemErr = 0
	ss.TrlBlendErr = 0
	ss.TrlOtherErr = 0
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
	ss.TrlCosDiff = 0
	ss.TrlSSE, ss.TrlAvgSSE = 0, 0
	ntrg := 0
	for _, lyi := range ss.Net.Layers {
		ly := lyi.(leabra.LeabraLayer).AsLeabra()
		if ly.Typ != emer.Target {
			continue
		}
		ss.TrlCosDiff += float64(ly.CosDiff.Cos)
		sse, avgsse := ly.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
		ss.TrlSSE += sse
		ss.TrlAvgSSE += avgsse
		ntrg++
	}
	if ntrg > 0 {
		ss.TrlCosDiff /= float64(ntrg)
		ss.TrlSSE /= float64(ntrg)
		ss.TrlAvgSSE /= float64(ntrg)
	}

	if ss.TrlSSE == 0 {
		ss.TrlErr = 0
	} else {
		ss.TrlErr = 1
	}
	if accum {
		ss.SumErr += ss.TrlErr
		ss.SumSSE += ss.TrlSSE
		ss.SumAvgSSE += ss.TrlAvgSSE
		ss.SumCosDiff += ss.TrlCosDiff
	}

	ss.TrlName = trlnm
	pidx := ss.TrainPats.RowsByString("Name", trlnm, etable.Equals, etable.UseCase)[0]
	if pidx < 20 {
		ss.TrlConAbs = 0
	} else {
		ss.TrlConAbs = 1
	}
	if !accum { // test
		ss.DyslexStats(ss.Net)
	}

	return
}

// DyslexStats computes dyslexia pronunciation, semantics stats
func (ss *Sim) DyslexStats(net emer.Network) {
	_, sse, cnm := ss.ClosestStat(net, "Phonology", "ActM", ss.TrainPats, "Phonology", "Name")
	ss.TrlPhon = cnm
	ss.TrlPhonSSE = float64(sse)
	if sse > 3 { // 3 is the threshold for blend errors
		ss.TrlBlendErr = 1
	} else {
		ss.TrlBlendErr = 0
		ss.TrlVisErr = 0
		ss.TrlSemErr = 0
		ss.TrlVisSemErr = 0
		ss.TrlOtherErr = 0
		if ss.TrlName != ss.TrlPhon {
			ss.TrlVisErr = ss.ClosePat(ss.TrlName, ss.TrlPhon, ss.CloseOrthos)
			ss.TrlSemErr = ss.ClosePat(ss.TrlName, ss.TrlPhon, ss.CloseSems)
			if ss.TrlVisErr > 0 && ss.TrlSemErr > 0 {
				ss.TrlVisSemErr = 1
			}
			if ss.TrlVisErr == 0 && ss.TrlSemErr == 0 {
				ss.TrlOtherErr = 1
			}
		}
	}
}

// ClosestStat finds the closest pattern in given column of given table to
// given layer activation pattern using given variable.  Returns the row number,
// sse value, and value of a column named namecol for that row if non-empty.
// Column must be etensor.Float32
func (ss *Sim) ClosestStat(net emer.Network, lnm, varnm string, dt *etable.Table, colnm, namecol string) (int, float32, string) {
	vt := ss.ValsTsr(lnm)
	ly := net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
	ly.UnitValsTensor(vt, varnm)
	col := dt.ColByName(colnm)
	row, sse := metric.ClosestRow32(vt, col.(*etensor.Float32), metric.SumSquaresBinTol32)
	nm := ""
	if namecol != "" {
		nm = dt.CellString(namecol, row)
	}
	return row, sse, nm
}

// ClosePat looks up phon pattern name in given table of close names -- if found returns 1, else 0
func (ss *Sim) ClosePat(trlnm, phon string, clsdt *etable.Table) float64 {
	rws := clsdt.RowsByString(trlnm, phon, etable.Equals, etable.UseCase)
	return float64(len(rws))
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
	switch epc {
	case 100:
		ss.Net.LrateMult(1) // not using -- not necc.
		// fmt.Printf("dropped lrate 0.5 at epoch: %d\n", epc)
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
			ss.UpdateView(false, -1)
		}
		ss.LogTstEpc(ss.TstEpcLog)
		if returnOnChg {
			return
		}
	}

	// note: type must be in place before apply inputs
	ss.SetInputLayer(3) // 3 is testing with compare on other layers, ortho input
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
	// ss.Net.LayerByName("Output").SetType(emer.Compare)
	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc(false)                             // !train
	ss.TrialStats(false, ss.TestEnv.TrialName.Cur) // !accumulate
	ss.TestEnv.Trial.Cur = cur
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.SetParams("Network", false)
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
	// patgen.ReshapeCppFile(ss.TrainPats, "TrainEnv.dat", "train_pats.tsv")    // one-time reshape
	// patgen.ReshapeCppFile(ss.Semantics, "FeatureNames.dat", "semantics.tsv") // one-time reshape
	// patgen.ReshapeCppFile(ss.CloseOrthos, "OrthoClosePats.dat", "close_orthos.tsv") // one-time reshape
	// patgen.ReshapeCppFile(ss.CloseSems, "SemClosePats.dat", "close_sems.tsv")       // one-time reshape
	ss.OpenPatAsset(ss.TrainPats, "train_pats.tsv", "TrainPats", "Training patterns")
	ss.OpenPatAsset(ss.Semantics, "semantics.tsv", "Semantics", "Semantics features and properties")
	ss.OpenPatAsset(ss.CloseOrthos, "close_orthos.tsv", "CloseOrthos", "Close Orthography items")
	ss.OpenPatAsset(ss.CloseSems, "close_sems.tsv", "CloseSems", "Close Semantic items")
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
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " ActAvg", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Dyslex Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AvgSSE", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
	plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

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
	dt.SetCellString("TrialName", row, strings.Split(ss.TrlName, "_")[0])
	dt.SetCellString("Phon", row, ss.TrlPhon)
	dt.SetCellFloat("PhonSSE", row, ss.TrlPhonSSE)
	dt.SetCellFloat("ConAbs", row, ss.TrlConAbs)
	dt.SetCellFloat("Vis", row, ss.TrlVisErr)
	dt.SetCellFloat("Sem", row, ss.TrlSemErr)
	dt.SetCellFloat("VisSem", row, ss.TrlVisSemErr)
	dt.SetCellFloat("Blend", row, ss.TrlBlendErr)
	dt.SetCellFloat("Other", row, ss.TrlOtherErr)
	dt.SetCellFloat("Err", row, ss.TrlErr)
	dt.SetCellFloat("SSE", row, ss.TrlSSE)
	dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
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
		{"Phon", etensor.STRING, nil, nil},
		{"PhonSSE", etensor.FLOAT64, nil, nil},
		{"ConAbs", etensor.FLOAT64, nil, nil},
		{"Vis", etensor.FLOAT64, nil, nil},
		{"Sem", etensor.FLOAT64, nil, nil},
		{"VisSem", etensor.FLOAT64, nil, nil},
		{"Blend", etensor.FLOAT64, nil, nil},
		{"Other", etensor.FLOAT64, nil, nil},
		{"Err", etensor.FLOAT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " ActM.Avg", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Dyslex Test Trial Plot"
	plt.Params.XAxisCol = "TrialName"
	plt.Params.Type = eplot.Bar
	plt.SetTable(dt) // this sets defaults so set params after
	plt.Params.XAxisRot = 45
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Phon", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PhonSSE", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("ConAbs", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Vis", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Sem", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("VisSem", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Blend", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Other", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("Err", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActM.Avg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 0.5)
	}
	return plt
}

//////////////////////////////////////////////
//  TstEpcLog

func (ss *Sim) LesionStr() string {
	if ss.Lesion <= DirectFull {
		return ss.Lesion.String()
	}
	return fmt.Sprintf("%s %g", ss.Lesion, ss.LesionProp)
}

func (ss *Sim) LogTstEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	trl := ss.TstTrlLog
	tix := etable.NewIdxView(trl)
	epc := ss.TrainEnv.Epoch.Prv // ?

	cols := []string{"Vis", "Sem", "VisSem", "Blend", "Other"}

	spl := split.GroupBy(tix, []string{"ConAbs"})

	for _, cl := range cols {
		split.Agg(spl, cl, agg.AggSum)
	}
	tst := spl.AggsToTable(etable.ColNameOnly)
	ss.TstStats = tst

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellString("Lesion", row, ss.LesionStr())
	dt.SetCellFloat("LesionProp", row, float64(ss.LesionProp))

	for _, cl := range cols {
		dt.SetCellFloat("Con"+cl, row, tst.CellFloat(cl, 0))
		dt.SetCellFloat("Abs"+cl, row, tst.CellFloat(cl, 1))
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
		{"Lesion", etensor.STRING, nil, nil},
		{"LesionProp", etensor.FLOAT64, nil, nil},
	}
	cols := []string{"Vis", "Sem", "VisSem", "Blend", "Other"}
	for _, ty := range []string{"Con", "Abs"} {
		for _, cl := range cols {
			sch = append(sch, etable.Column{ty + cl, etensor.FLOAT64, nil, nil})
		}
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Dyslex Testing Epoch Plot"
	plt.Params.XAxisCol = "Lesion"
	plt.Params.Type = eplot.Bar
	plt.SetTable(dt) // this sets defaults so set params after
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Lesion", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("LesionProp", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	cols := []string{"Vis", "Sem", "VisSem", "Blend", "Other"}
	for _, ty := range []string{"Con", "Abs"} {
		for _, cl := range cols {
			plt.SetColParams(ty+cl, eplot.On, eplot.FixMin, 0, eplot.FixMax, 10)
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
		{"FirstZero", etensor.FLOAT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigRunPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Dyslex Run Plot"
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

// ClustPlots does all cluster plots
func (ss *Sim) ClustPlots() {
	if ss.SemClustPlot == nil {
		ss.SemClustPlot = &eplot.Plot2D{}
	}
	// get rid of _phon in names
	tpcp := ss.TrainPats.Clone()
	nm := tpcp.ColByName("Name")
	for r := 0; r < tpcp.Rows; r++ {
		n := nm.StringVal1D(r)
		n = strings.Split(n, "_")[0]
		nm.SetString1D(r, n)
	}
	ss.ClustPlot(ss.SemClustPlot, etable.NewIdxView(tpcp), "Semantics")
}

// ClustPlot does one cluster plot on given table column
func (ss *Sim) ClustPlot(plt *eplot.Plot2D, ix *etable.IdxView, colNm string) {
	nm, _ := ix.Table.MetaData["name"]
	smat := &simat.SimMat{}
	smat.TableCol(ix, colNm, "Name", false, metric.InvCosine64)
	pt := &etable.Table{}
	clust.Plot(pt, clust.Glom(smat, clust.ContrastDist), smat)
	plt.InitName(plt, colNm)
	plt.Params.Title = "Cluster Plot of: " + nm + " " + colNm
	plt.Params.XAxisCol = "X"
	plt.SetTable(pt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("X", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Y", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Label", true, false, 0, false, 0)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.ViewDefaults()
	nv.Params.Raster.Max = 100
	nv.Scene().Camera.Pose.Pos.Set(0, 1.05, 2.75)
	nv.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("dyslex")
	gi.SetAppAbout(`Simulates normal and disordered (dyslexic) reading performance in terms of a distributed representation of word-level knowledge across Orthography, Semantics, and Phonology. It is based on a model by Plaut and Shallice (1993). Note that this form of dyslexia is *aquired* (via brain lesions such as stroke) and not the more prevalent developmental variety.  See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch9/dyslex/README.md">README.md on GitHub</a>.</p>`)

	win := gi.NewMainWindow("dyslex", "Dyslex", width, height)
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

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "RunPlot").(*eplot.Plot2D)
	ss.RunPlot = ss.ConfigRunPlot(plt, ss.RunLog)

	split.SetSplits(.2, .8)

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

	tbar.AddAction(gi.ActOpts{Label: "Lesion", Icon: "cut", Tooltip: "Lesion network"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			giv.CallMethod(ss, "LesionNet", vp)
			vp.SetNeedsFullRender()
		})

	tbar.AddAction(gi.ActOpts{Label: "Reset RunLog", Icon: "update", Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.RunLog.SetNumRows(0)
			ss.RunPlot.Update()
		})

	tbar.AddSeparator("anal")

	tbar.AddAction(gi.ActOpts{Label: "Cluster Plot", Icon: "file-image", Tooltip: "run cluster plot on input patterns."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.ClustPlots()
			vp.SetNeedsFullRender()
		})

	tbar.AddSeparator("misc")

	tbar.AddAction(gi.ActOpts{Label: "New Seed", Icon: "new", Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.NewRndSeed()
			vp.SetNeedsFullRender()
		})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch9/dyslex/README.md")
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

func mainrun() {
	TheSim.New()
	TheSim.Config()

	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}
