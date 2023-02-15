// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
sg is the sentence gestalt model, which learns to encode both
syntax and semantics of sentences in an integrated "gestalt"
hidden layer. The sentences have simple agent-verb-patient
structure with optional prepositional or adverb modifier
phrase at the end, and can be either in the active or passive
form (80% active, 20% passive). There are ambiguous terms that
need to be resolved via context, showing a key interaction
between syntax and semantics.
*/
package main

import (
	"bytes"
	"flag"
	"fmt"
	"log"
	"math"
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
	"github.com/emer/leabra/deep"
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

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "norm and momentum on is critical, wt bal not as much but fine",
				Params: params.Params{
					"Prjn.Learn.Norm.On":     "true",
					"Prjn.Learn.Momentum.On": "true",
					"Prjn.Learn.WtBal.On":    "true",
					"Prjn.Learn.Lrate":       "0.04", // critical for lrate sched
				}},
			{Sel: "Layer", Desc: "more inhibition is better",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":  "2.4", // 2.4 > 2.6+ > 2.2-
					"Layer.Learn.AvgL.Gain": "2.5", // 2.5 > 3
					"Layer.Act.Gbar.L":      "0.1", // lower leak = better
					"Layer.Act.Init.Decay":  "0",
				}},
			{Sel: "TRCLayer", Desc: "standard weight is .3 here for larger distributed reps. no learn",
				Params: params.Params{
					"Layer.TRC.DriveScale":        "0.8", // using .8 for localist layer
					"Layer.Inhib.ActAvg.UseFirst": "false",
				}},
			{Sel: "CTLayer", Desc: "don't use first as it is typically very low",
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
			{Sel: ".Back", Desc: "weaker back -- .2 > .3, .1",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.2",
				}},
			{Sel: "CTCtxtPrjn", Desc: "yes weight balance",
				Params: params.Params{
					"Prjn.Learn.WtBal.On": "true", // true > false
					"Prjn.WtScale.Rel":    "1",    // 1 > 2
				}},
			{Sel: ".CTFmSuper", Desc: "from superficial layer",
				Params: params.Params{
					"Prjn.WtInit.Mean": "0.5",
				}},
			{Sel: ".GestSelfCtxt", Desc: "yes weight balance",
				Params: params.Params{
					"Prjn.WtScale.Rel": "3", // 3 > 2 > 4 -- not better to start smaller
				}},
			{Sel: ".EncSelfCtxt", Desc: "yes weight balance",
				Params: params.Params{
					"Prjn.WtScale.Rel": "5", // 5 > 4 > 3 > 6 -- not better to start smaller
				}},
			{Sel: ".CtxtBack", Desc: "gest CT - > encode CT basically",
				Params: params.Params{
					"Prjn.WtScale.Rel": "1",
				}},
			{Sel: ".FmInput", Desc: "from localist inputs -- 1 == .3",
				Params: params.Params{
					"Prjn.WtScale.Rel":      "1",
					"Prjn.Learn.WtSig.Gain": "6", // 1 == 6
				}},
			{Sel: ".EncodePToSuper", Desc: "teaching signal from input pulvinar, to super -- .05 > .2",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.05", // .05 == .02 > .2
				}},
			{Sel: ".EncodePToCT", Desc: "critical to make this small so deep context dominates -- .05",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.05", // .05 == .02
				}},
			{Sel: ".CtxtFmInput", Desc: "making this weaker than 1 causes encodeD to freeze, 1 == 1.5 > lower",
				Params: params.Params{
					"Prjn.WtScale.Rel": "1.0",
				}},
			{Sel: "#DecodeToGestaltCT", Desc: "this leaks current role into context directly",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.2", // .2 > .3 > .1 > .05(bad) > .02(vbad)
				}},
			{Sel: "#GestaltCTToEncodeP", Desc: "eliminating rescues EncodeD -- trying weaker",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.05", // .02 > .05 > .1 > .2 etc -- .02 better than nothing!
				}},
		},
	}},
	{Name: "StrongSelfCtxt", Desc: "increase the self context strength -- not useful", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: ".GestSelfCtxt", Desc: "yes weight balance",
				Params: params.Params{
					"Prjn.WtScale.Rel": "3", // 3 > 2 > 4 (blows up) -- start at 1, then move to 3
				}},
			{Sel: ".EncSelfCtxt", Desc: "yes weight balance",
				Params: params.Params{
					"Prjn.WtScale.Rel": "5", // 5 > 4 > 3 > 6 -- start at 1, then move to 5
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
	Net             *deep.Network            `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	TrnEpcLog       *etable.Table            `view:"no-inline" desc:"training epoch-level log data"`
	TstEpcLog       *etable.Table            `view:"no-inline" desc:"testing epoch-level log data"`
	TrnTrlLog       *etable.Table            `view:"no-inline" desc:"training trial-level log data"`
	TstTrlLog       *etable.Table            `view:"no-inline" desc:"testing trial-level log data"`
	SentProbeTrlLog *etable.Table            `view:"no-inline" desc:"probing trial-level log data"`
	NounProbeTrlLog *etable.Table            `view:"no-inline" desc:"probing trial-level log data"`
	TrnTrlAmbStats  *etable.Table            `view:"no-inline" desc:"aggregate trl stats for last epc"`
	TrnTrlQTypStats *etable.Table            `view:"no-inline" desc:"aggregate trl stats for last epc"`
	RunLog          *etable.Table            `view:"no-inline" desc:"summary log of each run"`
	RunStats        *etable.Table            `view:"no-inline" desc:"aggregate stats on all runs"`
	SimMats         map[string]*simat.SimMat `view:"no-inline" desc:"similarity matricies"`
	Params          params.Sets              `view:"no-inline" desc:"full collection of param sets"`
	ParamSet        string                   `desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set -- can use multiple names separated by spaces (don't put spaces in ParamSet names!)"`
	Tag             string                   `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"`
	MaxRuns         int                      `desc:"maximum number of model runs to perform"`
	MaxEpcs         int                      `desc:"maximum number of epochs to run per model run"`
	NZeroStop       int                      `desc:"if a positive number, training will stop after this many epochs with zero SSE"`
	TrainEnv        SentGenEnv               `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	TestEnv         SentGenEnv               `desc:"Testing environment -- manages iterating over testing"`
	SentProbeEnv    SentGenEnv               `desc:"Probe environment -- manages iterating over testing"`
	NounProbeEnv    ProbeEnv                 `desc:"Probe environment -- manages iterating over testing"`
	Time            leabra.Time              `desc:"leabra timing parameters and state"`
	ViewOn          bool                     `desc:"whether to update the network view while running"`
	TrainUpdt       leabra.TimeScales        `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt        leabra.TimeScales        `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	TestInterval    int                      `desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`
	LayStatNms      []string                 `view:"-" desc:"names of layers to collect more detailed stats on (avg act, etc)"`
	StatLayNms      []string                 `view:"-" desc:"stat layers"`
	StatNms         []string                 `view:"-" desc:"stat short names"`
	ProbeNms        []string                 `view:"-" desc:"layers to probe"`

	// statistics: note use float64 as that is best for etable.Table
	TrlOut        string     `inactive:"+" desc:"output response(s) output units active > .2"`
	TrlPred       string     `inactive:"+" desc:"predicted word(s) active > .2"`
	TrlErr        [2]float64 `inactive:"+" desc:"1 if trial was error, 0 if correct -- based on SSE = 0 (subject to .5 unit-wise tolerance)"`
	TrlSSE        [2]float64 `inactive:"+" desc:"current trial's sum squared error"`
	TrlAvgSSE     [2]float64 `inactive:"+" desc:"current trial's average sum squared error"`
	TrlCosDiff    [2]float64 `inactive:"+" desc:"current trial's cosine difference"`
	EpcSSE        [2]float64 `inactive:"+" desc:"last epoch's total sum squared error"`
	EpcAvgSSE     [2]float64 `inactive:"+" desc:"last epoch's average sum squared error (average over trials, and over units within layer)"`
	EpcPctErr     [2]float64 `inactive:"+" desc:"last epoch's average TrlErr"`
	EpcPctCor     [2]float64 `inactive:"+" desc:"1 - last epoch's average TrlErr"`
	EpcCosDiff    [2]float64 `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	EpcPerTrlMSec float64    `inactive:"+" desc:"how long did the epoch take per trial in wall-clock milliseconds"`
	FirstZero     int        `inactive:"+" desc:"epoch at when SSE first went to zero"`
	NZero         int        `inactive:"+" desc:"number of epochs in a row with zero SSE"`

	// internal state - view:"-"
	SumN               [2]float64                  `view:"-" inactive:"+" desc:"number of each stat"`
	SumErr             [2]float64                  `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumSSE             [2]float64                  `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumAvgSSE          [2]float64                  `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumCosDiff         [2]float64                  `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	Win                *gi.Window                  `view:"-" desc:"main GUI window"`
	NetView            *netview.NetView            `view:"-" desc:"the network viewer"`
	ToolBar            *gi.ToolBar                 `view:"-" desc:"the master toolbar"`
	TrnEpcPlot         *eplot.Plot2D               `view:"-" desc:"the training epoch plot"`
	TstEpcPlot         *eplot.Plot2D               `view:"-" desc:"the testing epoch plot"`
	TrnTrlPlot         *eplot.Plot2D               `view:"-" desc:"the train-trial plot"`
	TstTrlPlot         *eplot.Plot2D               `view:"-" desc:"the test-trial plot"`
	SentProbeClustPlot *eplot.Plot2D               `view:"-" desc:"the probe cluster plot"`
	NounProbeClustPlot *eplot.Plot2D               `view:"-" desc:"the probe cluster plot"`
	RunPlot            *eplot.Plot2D               `view:"-" desc:"the run plot"`
	TrnEpcFile         *os.File                    `view:"-" desc:"log file"`
	RunFile            *os.File                    `view:"-" desc:"log file"`
	ValsTsrs           map[string]*etensor.Float32 `view:"-" desc:"for holding layer values"`
	SaveWts            bool                        `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	NoGui              bool                        `view:"-" desc:"if true, runing in no GUI mode"`
	LogSetParams       bool                        `view:"-" desc:"if true, print message for all params that are set"`
	IsRunning          bool                        `view:"-" desc:"true if sim is running"`
	StopNow            bool                        `view:"-" desc:"flag to stop running"`
	NeedsNewRun        bool                        `view:"-" desc:"flag to initialize NewRun if last one finished"`
	RndSeed            int64                       `view:"-" desc:"the current random seed"`
	LastEpcTime        time.Time                   `view:"-" desc:"timer for last epoch"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &deep.Network{}
	ss.TrnEpcLog = &etable.Table{}
	ss.TstEpcLog = &etable.Table{}
	ss.TrnTrlLog = &etable.Table{}
	ss.TstTrlLog = &etable.Table{}
	ss.SentProbeTrlLog = &etable.Table{}
	ss.NounProbeTrlLog = &etable.Table{}
	ss.RunLog = &etable.Table{}
	ss.RunStats = &etable.Table{}
	ss.SimMats = make(map[string]*simat.SimMat)
	ss.Params = ParamSets
	ss.RndSeed = 10
	ss.ViewOn = true
	ss.TrainUpdt = leabra.AlphaCycle
	ss.TestUpdt = leabra.AlphaCycle
	ss.TestInterval = 5000
	ss.LayStatNms = []string{"Encode", "EncodeCT", "Gestalt", "GestaltCT", "Decode"}
	ss.StatLayNms = []string{"Filler", "EncodeP"}
	ss.StatNms = []string{"Fill", "Inp"}
	ss.ProbeNms = []string{"Gestalt", "GestaltCT"}
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigTrnEpcLog(ss.TrnEpcLog)
	ss.ConfigTstEpcLog(ss.TstEpcLog)
	ss.ConfigTrnTrlLog(ss.TrnTrlLog)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	ss.ConfigSentProbeTrlLog(ss.SentProbeTrlLog)
	ss.ConfigNounProbeTrlLog(ss.NounProbeTrlLog)
	ss.ConfigRunLog(ss.RunLog)
}

func (ss *Sim) ConfigEnv() {
	if ss.MaxRuns == 0 { // allow user override
		ss.MaxRuns = 1
	}
	if ss.MaxEpcs == 0 { // allow user override
		ss.MaxEpcs = 500
		ss.NZeroStop = 5
	}

	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	ss.TrainEnv.Seq.Max = 100 // sequences per epoch training
	ss.TrainEnv.OpenRulesFromAsset("sg_rules.txt")
	// ss.TrainEnv.Rules.OpenRules("sg_rules.txt")
	ss.TrainEnv.PPassive = 0.2
	ss.TrainEnv.Words = SGWords
	ss.TrainEnv.Roles = SGRoles
	ss.TrainEnv.Fillers = SGFillers
	ss.TrainEnv.WordTrans = SGWordTrans
	ss.TrainEnv.AmbigVerbs = SGAmbigVerbs
	ss.TrainEnv.AmbigNouns = SGAmbigNouns
	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually

	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Dsc = "testing params and state"
	ss.TestEnv.Seq.Max = 14
	ss.TestEnv.OpenRulesFromAsset("sg_tests.txt")
	//	ss.TestEnv.Rules.OpenRules("sg_tests.txt")
	ss.TestEnv.PPassive = 0 // passive explicitly marked
	ss.TestEnv.Words = SGWords
	ss.TestEnv.Roles = SGRoles
	ss.TestEnv.Fillers = SGFillers
	ss.TestEnv.WordTrans = SGWordTrans
	ss.TestEnv.AmbigVerbs = SGAmbigVerbs
	ss.TestEnv.AmbigNouns = SGAmbigNouns
	ss.TestEnv.Validate()

	ss.SentProbeEnv.Nm = "SentProbeEnv"
	ss.SentProbeEnv.Dsc = "probe params and state"
	ss.SentProbeEnv.Seq.Max = 17
	ss.SentProbeEnv.OpenRulesFromAsset("sg_probes.txt")
	// ss.SentProbeEnv.Rules.OpenRules("sg_probes.txt")
	ss.SentProbeEnv.PPassive = 0 // passive explicitly marked
	ss.SentProbeEnv.Words = SGWords
	ss.SentProbeEnv.Roles = SGRoles
	ss.SentProbeEnv.Fillers = SGFillers
	ss.SentProbeEnv.WordTrans = SGWordTrans
	ss.SentProbeEnv.AmbigVerbs = SGAmbigVerbs
	ss.SentProbeEnv.AmbigNouns = SGAmbigNouns
	ss.SentProbeEnv.Validate()

	ss.NounProbeEnv.Nm = "NounProbeEnv"
	ss.NounProbeEnv.Dsc = "probe params and state"
	ss.NounProbeEnv.Words = SGWords
	ss.NounProbeEnv.Validate()

	ss.TrainEnv.Init(0)
	ss.TestEnv.Init(0)
	ss.SentProbeEnv.Init(0)
	ss.NounProbeEnv.Init(0)
}

func (ss *Sim) ConfigNet(net *deep.Network) {
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

	net.InitName(net, "SentGestalt")
	in := net.AddLayer2D("Input", 10, 5, emer.Input)
	role := net.AddLayer2D("Role", 9, 1, emer.Input)
	fill := net.AddLayer2D("Filler", 11, 5, emer.Target)
	enc, encct, encp := net.AddDeep2D("Encode", 12, 12) // 12x12 better..
	enc.SetClass("Encode")
	encct.SetClass("Encode")
	dec := net.AddLayer2D("Decode", 12, 12, emer.Hidden)
	gest, gestct := net.AddDeepNoTRC2D("Gestalt", 12, 12) // 12x12 def better with full
	gest.SetClass("Gestalt")
	gestct.SetClass("Gestalt")

	encp.Shape().CopyShape(in.Shape())
	encp.(*deep.TRCLayer).Drivers.Add("Input")

	encp.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Input", YAlign: relpos.Front, Space: 2})
	role.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "EncodeP", YAlign: relpos.Front, Space: 4})
	fill.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Role", YAlign: relpos.Front, Space: 4})
	enc.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Input", YAlign: relpos.Front, XAlign: relpos.Left})
	encct.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Encode", YAlign: relpos.Front, Space: 2})
	dec.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "EncodeCT", YAlign: relpos.Front, Space: 2})
	gest.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Encode", YAlign: relpos.Front, XAlign: relpos.Left})
	gestct.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Gestalt", YAlign: relpos.Front, Space: 2})

	full := prjn.NewFull()
	full.SelfCon = true

	pj := net.ConnectLayers(in, enc, full, emer.Forward)
	pj.SetClass("FmInput")

	pj = net.ConnectLayers(in, gest, full, emer.Forward) // this is key -- skip encoder
	pj.SetClass("FmInput")

	encct.(leabra.LeabraLayer).AsLeabra().SendName("EncodeP").SetClass("EncodePToCT")
	enc.(leabra.LeabraLayer).AsLeabra().SendName("EncodeP").SetClass("EncodePToSuper")

	// gestd gets error from Filler, this communicates Filler to encd -> corrupts prediction
	// net.ConnectLayers(gestd, encd, full, emer.Forward)

	// testing no use of enc at all
	net.BidirConnectLayers(enc, gest, full)

	net.ConnectLayers(gestct, enc, full, emer.Back) // give enc the best of gestd
	// net.ConnectLayers(gestd, gest, full, emer.Back) // not essential?  todo retest

	// this allows current role info to propagate back to input prediction
	// does not seem to be important
	// net.ConnectLayers(gestd, inp, full, emer.Forward) // must be weaker..

	// if gestd not driving inp, then this is bad -- .005 MIGHT be tiny bit beneficial but not worth it
	// net.ConnectLayers(inp, gestd, full, emer.Back).SetClass("EncodePToGestalt")
	// net.ConnectLayers(inp, gest, full, emer.Back).SetClass("EncodePToGestalt")

	net.BidirConnectLayers(gest, dec, full)
	net.BidirConnectLayers(gestct, dec, full) // bidir is essential here to get error signal
	// directly into context layer -- has rel of 0.2

	// net.BidirConnectLayers(enc, dec, full) // not beneficial

	net.BidirConnectLayers(dec, role, full)
	net.BidirConnectLayers(dec, fill, full)

	// add extra deep context
	net.ConnectCtxtToCT(encct, encct, full).SetClass("EncSelfCtxt") // one2one doesn't work
	net.ConnectCtxtToCT(in, encct, full).SetClass("CtxtFmInput")
	net.ConnectCtxtToCT(gestct, encct, full).SetClass("CtxtBack")

	// add extra deep context
	net.ConnectCtxtToCT(gestct, gestct, full).SetClass("GestSelfCtxt") // full > one2one
	// net.ConnectCtxtToCT(in, gestct, full).SetClass("CtxtFmInput")
	// net.ConnectLayers(encp, gestct, full, emer.Back).SetClass("EncodePToCT") // actually bad
	net.ConnectCtxtToCT(enc, gestct, full).SetClass("CtxtFmInput") // better than direct from in

	net.Defaults()
	ss.SetParams("Network", ss.LogSetParams) // only set Network params
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
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.StopNow = false
	ss.SetParams("", ss.LogSetParams) // all sheets
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
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tSeq:\t%d\tTick:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Seq.Cur, ss.TrainEnv.Tick.Cur, ss.TrainEnv.Trial.Cur, ss.Time.Cycle, ss.TrainEnv.String())
	} else {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tSeq:\t%d\tTick:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TestEnv.Seq.Cur, ss.TestEnv.Tick.Cur, ss.TestEnv.Trial.Cur, ss.Time.Cycle, ss.TestEnv.String())
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
		// if ss.TrainEnv.Tick.Cur > 0 { // first unlearnable
		ss.Net.DWt()
		if ss.NetView != nil && ss.NetView.IsVisible() {
			ss.NetView.RecordSyns()
		}
		ss.Net.WtFmDWt()
		// }
	}
	if ss.ViewOn && viewUpdt == leabra.AlphaCycle {
		ss.UpdateView(train, -1)
	}
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"Input", "Role", "Filler"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := en.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// ApplyInputsProbe applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputsProbe(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"Input"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := en.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
	ss.LogRun(ss.RunLog)
	if ss.SaveWts {
		fnm := ss.WeightsFileName()
		fmt.Printf("Saving Weights to: %s\n", fnm)
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
	ss.InitStats()
	ss.TrnEpcLog.SetNumRows(0)
	ss.TstEpcLog.SetNumRows(0)
	ss.NeedsNewRun = false
}

// InitStats initializes all the statistics, especially important for the
// cumulative epoch stats -- called at start of new run
func (ss *Sim) InitStats() {
	// accumulators
	ss.FirstZero = -1
	ss.NZero = 0
	for i := 0; i < 2; i++ {
		ss.SumN[i] = 0
		ss.SumErr[i] = 0
		ss.SumSSE[i] = 0
		ss.SumAvgSSE[i] = 0
		ss.SumCosDiff[i] = 0
		// clear rest just to make Sim look initialized
		ss.TrlErr[i] = 0
		ss.TrlSSE[i] = 0
		ss.TrlAvgSSE[i] = 0
		ss.EpcSSE[i] = 0
		ss.EpcAvgSSE[i] = 0
		ss.EpcPctErr[i] = 0
		ss.EpcCosDiff[i] = 0
	}
}

// ActiveUnitNames reports names of units ActM active > thr, using list of names for units
func (ss *Sim) ActiveUnitNames(lnm string, nms []string, thr float32) []string {
	var acts []string
	ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.ActM > thr {
			acts = append(acts, nms[ni])
		}
	}
	return acts
}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func (ss *Sim) TrialStats(accum bool) {
	for li, lnm := range ss.StatLayNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		ss.TrlCosDiff[li] = float64(ly.CosDiff.Cos)
		sse, avgsse := ly.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
		ss.TrlSSE[li] = sse
		ss.TrlAvgSSE[li] = avgsse
		if ss.TrlSSE[li] > 0 {
			ss.TrlErr[li] = 1
		} else {
			ss.TrlErr[li] = 0
		}
		if lnm == "EncodeP" && ss.TrainEnv.Tick.Cur == 1 { // first sent trial unpredictable
			ss.TrlCosDiff[li] = 0
			ss.TrlSSE[li] = 0
			ss.TrlAvgSSE[li] = 0
			ss.TrlErr[li] = 0
		}
		if accum {
			if ss.TrainEnv.Tick.Cur == 0 && li == 0 {
				continue
			}
			ss.SumN[li] += 1
			ss.SumErr[li] += ss.TrlErr[li]
			ss.SumSSE[li] += ss.TrlSSE[li]
			ss.SumAvgSSE[li] += ss.TrlAvgSSE[li]
			ss.SumCosDiff[li] += ss.TrlCosDiff[li]
		}
		switch lnm {
		case "Filler":
			ss.TrlOut = strings.Join(ss.ActiveUnitNames(lnm, ss.TrainEnv.Fillers, .2), ", ")
		case "EncodeP":
			ss.TrlPred = strings.Join(ss.ActiveUnitNames(lnm, ss.TrainEnv.Words, .2), ", ")
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
		ss.TrainEnv.Trial.Cur = 0
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

	fill := ss.Net.LayerByName("Filler").(leabra.LeabraLayer).AsLeabra()
	fill.SetType(emer.Target)

	ss.ApplyInputs(&ss.TrainEnv)
	ss.AlphaCyc(true)   // train
	ss.TrialStats(true) // accumulate
	ss.LogTrnTrl(ss.TrnTrlLog)
}

// TrainSeq runs training trials for remainder of this sequence
func (ss *Sim) TrainSeq() {
	ss.StopNow = false
	curSeq := ss.TrainEnv.Seq.Cur
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Seq.Cur != curSeq {
			break
		}
	}
	ss.Stopped()
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

// LrateSched implements the learning rate schedule
// also CT self-context strength schedule!
func (ss *Sim) LrateSched(epc int) {
	switch epc {
	// case 2: // this is not any better than setting at the start
	// 	ss.SetParamsSet("StrongSelfCtxt", "Network", true)
	case 200:
		ss.Net.LrateMult(0.5)
		fmt.Printf("dropped lrate 0.5 at epoch: %d\n", epc)
	case 300:
		ss.Net.LrateMult(0.2)
		fmt.Printf("dropped lrate 0.2 at epoch: %d\n", epc)
	case 400:
		ss.Net.LrateMult(0.1)
		fmt.Printf("dropped lrate 0.1 at epoch: %d\n", epc)
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

// OpenWts opens trained weights
func (ss *Sim) OpenWts() {
	ab, err := Asset("trained.wts") // embedded in executable
	if err != nil {
		log.Println(err)
	}
	ss.Net.ReadWtsJSON(bytes.NewBuffer(ab))
	// ss.Net.OpenWtsJSON("trained.wts.gz")
}

// SaveWeights saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWeights(filename gi.FileName) {
	ss.Net.SaveWtsJSON(filename)
}

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// InitTest initializes testing state
func (ss *Sim) InitTest() {
	rand.Seed(ss.RndSeed)
	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
	ss.TstTrlLog.SetNumRows(0)
	ss.Net.InitActs()
	ss.UpdateView(false, -1)
}

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

	fill := ss.Net.LayerByName("Filler").(leabra.LeabraLayer).AsLeabra()
	fill.SetType(emer.Compare)

	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc(false)   // !train
	ss.TrialStats(false) // !accumulate
	ss.LogTstTrl(ss.TstTrlLog)
}

// TestSeq runs testing trials for remainder of this sequence
func (ss *Sim) TestSeq() {
	ss.StopNow = false
	curSeq := ss.TestEnv.Seq.Cur
	for {
		ss.TestTrial(true)
		if ss.StopNow || ss.TestEnv.Seq.Cur != curSeq {
			break
		}
	}
	ss.Stopped()
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.InitTest()
	for {
		ss.TestTrial(true) // return on change -- don't wrap
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

// ProbeAll runs all probes
func (ss *Sim) ProbeAll() {
	ss.InitTest()
	ss.SentProbeEnv.Init(ss.TrainEnv.Run.Cur)
	ss.SentProbeTrlLog.SetNumRows(0)

	fill := ss.Net.LayerByName("Filler").(leabra.LeabraLayer).AsLeabra()
	fill.SetType(emer.Compare)

	for {
		ss.SentProbeEnv.Step()
		if ss.SentProbeEnv.Seq.Cur == 0 {
			break
		}
		ss.ApplyInputs(&ss.SentProbeEnv)
		ss.AlphaCyc(false)   // !train
		ss.TrialStats(false) // !accumulate
		ss.LogSentProbeTrl(ss.SentProbeTrlLog)
	}

	ss.NounProbeEnv.Init(ss.TrainEnv.Run.Cur)
	ss.NounProbeTrlLog.SetNumRows(0)
	epc := ss.NounProbeEnv.Epoch.Cur
	for {
		ss.NounProbeEnv.Step()
		if ss.NounProbeEnv.Epoch.Cur != epc {
			break
		}
		ss.Net.InitActs()
		ss.ApplyInputsProbe(&ss.NounProbeEnv)
		ss.AlphaCyc(false)   // !train
		ss.TrialStats(false) // !accumulate
		ss.LogNounProbeTrl(ss.NounProbeTrlLog)
	}
	ss.ProbeClusterPlot()
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
	return ss.Net.Nm + "_" + ss.RunName() + "_" + lognm + ".tsv"
}

//////////////////////////////////////////////
//  TrnTrlLog

// LogTrnTrl adds data from current trial to the TrnTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTrnTrl(dt *etable.Table) {
	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value
	trl := ss.TrainEnv.Trial.Cur
	row := trl

	if trl == 0 {
		dt.SetNumRows(0)
	}

	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	cur := ss.TrainEnv.CurInputs()

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Seq", row, float64(ss.TrainEnv.Seq.Prv))
	dt.SetCellFloat("Tick", row, float64(ss.TrainEnv.Tick.Cur))
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("TrialName", row, ss.TrainEnv.String())
	dt.SetCellString("Input", row, cur[0])
	dt.SetCellString("Pred", row, ss.TrlPred)
	dt.SetCellString("Role", row, cur[1])
	dt.SetCellString("Filler", row, cur[2])
	dt.SetCellString("Output", row, ss.TrlOut)
	dt.SetCellString("QType", row, cur[3])
	dt.SetCellFloat("AmbigVerb", row, float64(ss.TrainEnv.NAmbigVerbs))
	dt.SetCellFloat("AmbigNouns", row, math.Min(float64(ss.TrainEnv.NAmbigNouns), 1))
	for li, lnm := range ss.StatNms {
		dt.SetCellFloat(lnm+"Err", row, ss.TrlErr[li])
		dt.SetCellFloat(lnm+"SSE", row, ss.TrlSSE[li])
		dt.SetCellFloat(lnm+"AvgSSE", row, ss.TrlAvgSSE[li])
		dt.SetCellFloat(lnm+"CosDiff", row, ss.TrlCosDiff[li])
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TrnTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTrnTrlLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnTrlLog")
	dt.SetMetaData("desc", "Record of training per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	// nt := ss.TrainEnv.Trial.Prv
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Seq", etensor.INT64, nil, nil},
		{"Tick", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"Input", etensor.STRING, nil, nil},
		{"Pred", etensor.STRING, nil, nil},
		{"Role", etensor.STRING, nil, nil},
		{"Filler", etensor.STRING, nil, nil},
		{"Output", etensor.STRING, nil, nil},
		{"QType", etensor.STRING, nil, nil},
		{"AmbigVerb", etensor.FLOAT64, nil, nil},
		{"AmbigNouns", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.StatNms {
		sch = append(sch, etable.Column{lnm + "Err", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "SSE", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "AvgSSE", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "CosDiff", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Sentence Gestalt Train Trial Plot"
	plt.Params.XAxisCol = "Trial"

	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Seq", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Tick", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Input", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Pred", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Role", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Filler", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Output", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("QType", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AmbigVerb", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("AmbigNouns", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)

	for _, lnm := range ss.StatNms {
		plt.SetColParams(lnm+"Err", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
		plt.SetColParams(lnm+"SSE", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
		plt.SetColParams(lnm+"AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
		plt.SetColParams(lnm+"CosDiff", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	}
	return plt
}

//////////////////////////////////////////////
//  TrnEpcLog

// HogDead computes the proportion of units in given layer name with ActAvg over hog thr
// and under dead threshold
func (ss *Sim) HogDead(lnm string) (hog, dead float64) {
	ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
	n := float64(len(ly.Neurons))
	if n == 0 {
		return
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.ActAvg > 0.3 {
			hog += 1
		} else if nrn.ActAvg < 0.01 {
			dead += 1
		}
	}
	hog /= n
	dead /= n
	return
}

// LogTrnEpc adds data from current epoch to the TrnEpcLog table.
// computes epoch averages prior to logging.
func (ss *Sim) LogTrnEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epc := ss.TrainEnv.Epoch.Prv     // this is triggered by increment so use previous value
	nt := float64(ss.TrnTrlLog.Rows) // number of trials in view

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))

	if ss.LastEpcTime.IsZero() {
		ss.EpcPerTrlMSec = 0
	} else {
		iv := time.Now().Sub(ss.LastEpcTime)
		ss.EpcPerTrlMSec = float64(iv) / (nt * float64(time.Millisecond))
	}
	ss.LastEpcTime = time.Now()
	dt.SetCellFloat("PerTrlMSec", row, ss.EpcPerTrlMSec)

	for li, lnm := range ss.StatNms {
		ss.EpcSSE[li] = ss.SumSSE[li] / ss.SumN[li]
		ss.SumSSE[li] = 0
		ss.EpcAvgSSE[li] = ss.SumAvgSSE[li] / ss.SumN[li]
		ss.SumAvgSSE[li] = 0
		ss.EpcPctErr[li] = float64(ss.SumErr[li]) / ss.SumN[li]
		ss.SumErr[li] = 0
		ss.EpcPctCor[li] = 1 - ss.EpcPctErr[li]
		ss.EpcCosDiff[li] = ss.SumCosDiff[li] / ss.SumN[li]
		ss.SumCosDiff[li] = 0
		ss.SumN[li] = 0
		dt.SetCellFloat(lnm+"SSE", row, ss.EpcSSE[li])
		dt.SetCellFloat(lnm+"AvgSSE", row, ss.EpcAvgSSE[li])
		dt.SetCellFloat(lnm+"PctErr", row, ss.EpcPctErr[li])
		dt.SetCellFloat(lnm+"PctCor", row, ss.EpcPctCor[li])
		dt.SetCellFloat(lnm+"CosDiff", row, ss.EpcCosDiff[li])
	}

	ss.LogEpcStats(dt, ss.TrnTrlLog)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+" ActAvg", row, float64(ly.Pools[0].ActAvg.ActPAvgEff))
		hog, dead := ss.HogDead(lnm)
		dt.SetCellFloat(ly.Nm+" Hog", row, hog)
		dt.SetCellFloat(ly.Nm+" Dead", row, dead)
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

// LogEpcStats does extra SG epoch-level stats on a given trial table
func (ss *Sim) LogEpcStats(dt *etable.Table, trlog *etable.Table) {
	row := dt.Rows - 1
	tix := etable.NewIdxView(trlog)
	ambspl := split.GroupBy(tix, []string{"AmbigNouns"})
	qtspl := split.GroupBy(tix, []string{"QType"})

	// // no ambig and no final question
	// noambtix := etable.NewIdxView(trlog)
	// noambtix.Filter(func(et *etable.Table, row int) bool {
	// 	return et.CellFloat("AmbigNouns", row) == 0
	// })
	// rolespl := split.GroupBy(noambtix, []string{"Role"})

	cols := []string{"Err", "SSE", "CosDiff"}
	for _, lnm := range ss.StatNms {
		for _, cl := range cols {
			split.Agg(ambspl, lnm+cl, agg.AggMean)
			split.Agg(qtspl, lnm+cl, agg.AggMean)
			// split.Agg(rolespl, lnm+cl, agg.AggMean)
		}
	}
	ambst := ambspl.AggsToTable(etable.ColNameOnly)
	ss.TrnTrlAmbStats = ambst
	// rolest := rolespl.AggsToTable(etable.ColNameOnly)
	// ss.TrnTrlRoleStats = rolest

	if ambst != nil && ambst.Rows == 2 {
		for _, cl := range cols {
			dt.SetCellFloat("UnAmbFill"+cl, row, ambst.CellFloat("Fill"+cl, 0))
			dt.SetCellFloat("AmbFill"+cl, row, ambst.CellFloat("Fill"+cl, 1))
		}
	}

	qtst := qtspl.AggsToTable(etable.ColNameOnly)
	ss.TrnTrlQTypStats = qtst
	if qtst != nil && qtst.Rows == 2 {
		for _, cl := range cols {
			dt.SetCellFloat("CurQFill"+cl, row, qtst.CellFloat("Fill"+cl, 0))
			dt.SetCellFloat("RevQFill"+cl, row, qtst.CellFloat("Fill"+cl, 1))
		}
	}

	// for _, lnm := range ss.StatNms {
	// 	for ri, rl := range SGRoles {
	// 		for _, cl := range cols {
	// 			dt.SetCellFloat(rl+lnm+cl, row, rolest.CellFloat(lnm+cl, ri))
	// 		}
	// 	}
	// }
}

func (ss *Sim) ConfigTrnEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnEpcLog")
	dt.SetMetaData("desc", "Record of performance over epochs of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"PerTrlMSec", etensor.FLOAT64, nil, nil},
	}

	for _, lnm := range ss.StatNms {
		sch = append(sch, etable.Column{lnm + "SSE", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "AvgSSE", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "PctErr", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "PctCor", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "CosDiff", etensor.FLOAT64, nil, nil})
	}

	cols := []string{"Err", "SSE", "CosDiff"}
	for _, cl := range cols {
		sch = append(sch, etable.Column{"UnAmbFill" + cl, etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{"AmbFill" + cl, etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{"CurQFill" + cl, etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{"RevQFill" + cl, etensor.FLOAT64, nil, nil})
	}

	// for _, lnm := range ss.StatNms {
	// 	for _, rl := range SGRoles {
	// 		for _, cl := range cols {
	// 			sch = append(sch, etable.Column{rl + lnm + cl, etensor.FLOAT64, nil, nil})
	// 		}
	// 	}
	// }

	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " ActAvg", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + " Hog", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + " Dead", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Sentence Gestalt Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)

	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PerTrlMSec", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

	for _, lnm := range ss.StatNms {
		plt.SetColParams(lnm+"SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
		plt.SetColParams(lnm+"AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
		plt.SetColParams(lnm+"PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
		plt.SetColParams(lnm+"PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
		plt.SetColParams(lnm+"CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	}

	cols := []string{"Err", "SSE", "CosDiff"}
	for _, cl := range cols {
		plt.SetColParams("UnAmbFill"+cl, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
		plt.SetColParams("AmbFill"+cl, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
		plt.SetColParams("CurQFill"+cl, (cl == "Err"), eplot.FixMin, 0, eplot.FixMax, 1)
		plt.SetColParams("RevQFill"+cl, (cl == "Err"), eplot.FixMin, 0, eplot.FixMax, 1)
	}

	// for _, lnm := range ss.StatNms {
	// 	for _, rl := range SGRoles {
	// 		for _, cl := range cols {
	// 			plt.SetColParams(rl+lnm+cl, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	// 		}
	// 	}
	// }

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActAvg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, .5)
		plt.SetColParams(lnm+" Hog", eplot.Off, eplot.FixMin, 0, eplot.FixMax, .5)
		plt.SetColParams(lnm+" Dead", eplot.Off, eplot.FixMin, 0, eplot.FixMax, .5)
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
	row := dt.Rows

	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	cur := ss.TestEnv.CurInputs()

	st := ""
	for n, _ := range ss.TestEnv.Rules.Fired {
		if n != "Sentences" {
			st = n
		}
	}

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Seq", row, float64(ss.TestEnv.Seq.Prv))
	dt.SetCellString("SentType", row, st)
	dt.SetCellFloat("Tick", row, float64(ss.TestEnv.Tick.Cur))
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("TrialName", row, ss.TestEnv.String())
	dt.SetCellString("Input", row, cur[0])
	dt.SetCellString("Pred", row, ss.TrlPred)
	dt.SetCellString("Role", row, cur[1])
	dt.SetCellString("Filler", row, cur[2])
	dt.SetCellString("Output", row, ss.TrlOut)
	dt.SetCellString("QType", row, cur[3])
	dt.SetCellFloat("AmbigVerb", row, float64(ss.TestEnv.NAmbigVerbs))
	dt.SetCellFloat("AmbigNouns", row, math.Min(float64(ss.TestEnv.NAmbigNouns), 1))
	for li, lnm := range ss.StatNms {
		dt.SetCellFloat(lnm+"Err", row, ss.TrlErr[li])
		dt.SetCellFloat(lnm+"SSE", row, ss.TrlSSE[li])
		dt.SetCellFloat(lnm+"AvgSSE", row, ss.TrlAvgSSE[li])
		dt.SetCellFloat(lnm+"CosDiff", row, ss.TrlCosDiff[li])
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
		{"Seq", etensor.INT64, nil, nil},
		{"SentType", etensor.STRING, nil, nil},
		{"Tick", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"Input", etensor.STRING, nil, nil},
		{"Pred", etensor.STRING, nil, nil},
		{"Role", etensor.STRING, nil, nil},
		{"Filler", etensor.STRING, nil, nil},
		{"Output", etensor.STRING, nil, nil},
		{"QType", etensor.STRING, nil, nil},
		{"AmbigVerb", etensor.FLOAT64, nil, nil},
		{"AmbigNouns", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.StatNms {
		sch = append(sch, etable.Column{lnm + "Err", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "SSE", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "AvgSSE", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "CosDiff", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Sentence Gestalt Test Trial Plot"
	plt.Params.XAxisCol = "TrialName"
	plt.Params.Type = eplot.Bar
	plt.SetTable(dt)
	plt.Params.XAxisRot = 45

	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Seq", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SentType", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Tick", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Input", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Pred", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Role", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Filler", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Output", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("QType", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AmbigVerb", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("AmbigNouns", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	for _, lnm := range ss.StatNms {
		plt.SetColParams(lnm+"Err", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
		plt.SetColParams(lnm+"SSE", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
		plt.SetColParams(lnm+"AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
		plt.SetColParams(lnm+"CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	}

	plt.SetColParams("FillErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)

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

	for _, lnm := range ss.StatNms {
		dt.SetCellFloat(lnm+"SSE", row, agg.Mean(tix, lnm+"SSE")[0])
		dt.SetCellFloat(lnm+"AvgSSE", row, agg.Mean(tix, lnm+"AvgSSE")[0])
		dt.SetCellFloat(lnm+"PctErr", row, agg.Mean(tix, lnm+"Err")[0])
		dt.SetCellFloat(lnm+"CosDiff", row, agg.Mean(tix, lnm+"CosDiff")[0])
	}

	ss.LogEpcStats(dt, ss.TstTrlLog)

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
	}

	for _, lnm := range ss.StatNms {
		sch = append(sch, etable.Column{lnm + "SSE", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "AvgSSE", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "PctErr", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "CosDiff", etensor.FLOAT64, nil, nil})
	}

	cols := []string{"Err", "SSE", "CosDiff"}
	for _, cl := range cols {
		sch = append(sch, etable.Column{"UnAmbFill" + cl, etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{"AmbFill" + cl, etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{"CurQFill" + cl, etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{"RevQFill" + cl, etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Sentence Gestalt Testing Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

	for _, lnm := range ss.StatNms {
		plt.SetColParams(lnm+"SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
		plt.SetColParams(lnm+"AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
		plt.SetColParams(lnm+"PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
		plt.SetColParams(lnm+"CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	}

	cols := []string{"Err", "SSE", "CosDiff"}
	for _, cl := range cols {
		plt.SetColParams("UnAmbFill"+cl, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
		plt.SetColParams("AmbFill"+cl, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
		plt.SetColParams("CurQFill"+cl, (cl == "Err"), eplot.FixMin, 0, eplot.FixMax, 1)
		plt.SetColParams("RevQFill"+cl, (cl == "Err"), eplot.FixMin, 0, eplot.FixMax, 1)
	}
	return plt
}

//////////////////////////////////////////////
//  SentProbeTrlLog

// LogSentProbeTrl adds data from current trial to the SentProbeTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogSentProbeTrl(dt *etable.Table) {
	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value
	trl := ss.SentProbeEnv.Trial.Cur
	row := dt.Rows

	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	cur := ss.SentProbeEnv.CurInputs()
	st := ""
	for n, _ := range ss.SentProbeEnv.Rules.Fired {
		if n != "Sentences" {
			st = n
		}
	}

	dt.SetCellFloat("Run", row, float64(ss.SentProbeEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Seq", row, float64(ss.SentProbeEnv.Seq.Prv))
	dt.SetCellString("SentType", row, st)
	dt.SetCellFloat("Tick", row, float64(ss.SentProbeEnv.Tick.Cur))
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("TrialName", row, ss.SentProbeEnv.String())
	dt.SetCellString("Input", row, cur[0])
	dt.SetCellString("Pred", row, ss.TrlPred)
	dt.SetCellString("Role", row, cur[1])
	dt.SetCellString("Filler", row, cur[2])
	dt.SetCellString("Output", row, ss.TrlOut)
	dt.SetCellString("QType", row, cur[3])

	for _, lnm := range ss.ProbeNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		vt := ss.ValsTsr(lnm)
		ly.UnitValsTensor(vt, "ActM")
		dt.SetCellTensor(lnm, row, vt)
	}
}

func (ss *Sim) ConfigSentProbeTrlLog(dt *etable.Table) {
	dt.SetMetaData("name", "SentProbeTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Seq", etensor.INT64, nil, nil},
		{"SentType", etensor.STRING, nil, nil},
		{"Tick", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"Input", etensor.STRING, nil, nil},
		{"Pred", etensor.STRING, nil, nil},
		{"Role", etensor.STRING, nil, nil},
		{"Filler", etensor.STRING, nil, nil},
		{"Output", etensor.STRING, nil, nil},
		{"QType", etensor.STRING, nil, nil},
	}
	for _, lnm := range ss.ProbeNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		sch = append(sch, etable.Column{lnm, etensor.FLOAT64, ly.Shp.Shp, nil})
	}

	dt.SetFromSchema(sch, 0)
}

//////////////////////////////////////////////
//  NounProbeTrlLog

// LogNounProbeTrl adds data from current trial to the NounProbeTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogNounProbeTrl(dt *etable.Table) {
	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value
	trl := ss.NounProbeEnv.Trial.Cur
	row := dt.Rows

	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	dt.SetCellFloat("Run", row, float64(ss.NounProbeEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("TrialName", row, ss.NounProbeEnv.String())

	for _, lnm := range ss.ProbeNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		vt := ss.ValsTsr(lnm)
		ly.UnitValsTensor(vt, "ActM")
		dt.SetCellTensor(lnm, row, vt)
	}
}

func (ss *Sim) ConfigNounProbeTrlLog(dt *etable.Table) {
	dt.SetMetaData("name", "NounProbeTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
	}
	for _, lnm := range ss.ProbeNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		sch = append(sch, etable.Column{lnm, etensor.FLOAT64, ly.Shp.Shp, nil})
	}

	dt.SetFromSchema(sch, 0)
}

// SimMatByName gets SimMat by name
func (ss *Sim) SimMatByName(nm string) *simat.SimMat {
	sm, ok := ss.SimMats[nm]
	if !ok {
		sm = &simat.SimMat{}
		ss.SimMats[nm] = sm
	}
	return sm
}

// ProbeClustPlot does cluster plotting of probe data
func (ss *Sim) ProbeClusterPlot() {
	stix := etable.NewIdxView(ss.SentProbeTrlLog)
	stix.Filter(func(et *etable.Table, row int) bool {
		return et.CellFloat("Tick", row) == 5 // last of each sequence
	})
	ss.ClustPlot(ss.SentProbeClustPlot, stix, "GestaltCT", "SentType", clust.ContrastDist)
	ss.SentProbeClustPlot.Update()

	ntix := etable.NewIdxView(ss.NounProbeTrlLog)
	ss.ClustPlot(ss.NounProbeClustPlot, ntix, "Gestalt", "TrialName", clust.MaxDist)
	ss.NounProbeClustPlot.Update()
}

// ClustPlot does one cluster plot on given table column
func (ss *Sim) ClustPlot(plt *eplot.Plot2D, ix *etable.IdxView, colNm, lblNm string, dfunc clust.DistFunc) {
	nm, _ := ix.Table.MetaData["name"]
	smat := ss.SimMatByName(nm)
	smat.TableCol(ix, colNm, lblNm, false, metric.Euclidean64)
	smat.Mat.SetMetaData("colormap", "Viridis")
	smat.Mat.SetMetaData("fix-max", "false")

	pt := &etable.Table{}
	clust.Plot(pt, clust.Glom(smat, dfunc), smat)
	plt.InitName(plt, colNm)
	plt.Params.Title = "Cluster Plot of: " + nm + " " + colNm
	plt.Params.XAxisCol = "X"
	plt.SetTable(pt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("X", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Y", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Label", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
}

//////////////////////////////////////////////
//  RunLog

// LogRun adds data from current run to the RunLog table.
func (ss *Sim) LogRun(dt *etable.Table) {
	return
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

	params := ss.RunName() // includes tag

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
	plt.Params.Title = "Sentence Gestalt Run Plot"
	plt.Params.XAxisCol = "Run"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("FirstZero", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0) // default plot
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
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
	nv.Params.Raster.Max = 100
	nv.Scene().Camera.Pose.Pos.Set(0, 1.2, 3.0) // more "head on" than default which is more "top down"
	nv.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("SG")
	gi.SetAppAbout(`This is the sentence gestalt model, which learns to encode both syntax and semantics of sentences in an integrated "gestalt" hidden layer. The sentences have simple agent-verb-patient structure with optional prepositional or adverb modifier phrase at the end, and can be either in the active or passive form (80% active, 20% passive). There are ambiguous terms that need to be resolved via context, showing a key interaction between syntax and semantics. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch9/sg/README.md">README.md on GitHub</a>.</p>`)

	win := gi.NewMainWindow("SG", "Sentence Gestalt", width, height)
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

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TrnTrlPlot").(*eplot.Plot2D)
	ss.TrnTrlPlot = ss.ConfigTrnTrlPlot(plt, ss.TrnTrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstTrlPlot").(*eplot.Plot2D)
	ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstEpcPlot").(*eplot.Plot2D)
	ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.TstEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "SentProbeClustPlot").(*eplot.Plot2D)
	ss.SentProbeClustPlot = plt

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "NounProbeClustPlot").(*eplot.Plot2D)
	ss.NounProbeClustPlot = plt

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

	tbar.AddAction(gi.ActOpts{Label: "Step Seq", Icon: "fast-fwd", Tooltip: "Advances one sequence (sentence) at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainSeq()
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

	tbar.AddAction(gi.ActOpts{Label: "Open Weights", Icon: "update", Tooltip: "Open trained weights ", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.OpenWts()
	})

	tbar.AddAction(gi.ActOpts{Label: "Init Test", Icon: "update", Tooltip: "Initialize to start of testing items.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.InitTest()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Test Trial", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TestTrial(false) // don't return on change -- wrap
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Test Seq", Icon: "fast-fwd", Tooltip: "Advances one sequence (sentence) at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TestSeq()
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

	tbar.AddSeparator("log")

	tbar.AddAction(gi.ActOpts{Label: "Reset TstTrlLog", Icon: "update", Tooltip: "Reset the Testing trial log, so it is easier to read"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.TstTrlLog.SetNumRows(0)
			ss.TstTrlPlot.Update()
		})

	tbar.AddAction(gi.ActOpts{Label: "Probe All", Icon: "fast-fwd", Tooltip: "probe inputs.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.ProbeAll()
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddSeparator("misc")

	tbar.AddAction(gi.ActOpts{Label: "New Seed", Icon: "new", Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.NewRndSeed()
		})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch9/sg/README.md")
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

func (ss *Sim) CmdArgs() {
	ss.NoGui = true
	var nogui bool
	var saveEpcLog bool
	var saveRunLog bool
	var note string
	flag.StringVar(&ss.ParamSet, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.StringVar(&note, "note", "", "user note -- describe the run params etc")
	flag.IntVar(&ss.MaxRuns, "runs", 1, "number of runs to do (note that MaxEpcs is in paramset)")
	flag.BoolVar(&ss.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.SaveWts, "wts", true, "if true, save final weights after each run")
	flag.BoolVar(&saveEpcLog, "epclog", true, "if true, save train epoch log to file")
	flag.BoolVar(&saveRunLog, "runlog", true, "if true, save run epoch log to file")
	flag.BoolVar(&nogui, "nogui", true, "if not passing any other args and want to run nogui, use nogui")
	flag.Parse()
	ss.Init()

	if note != "" {
		fmt.Printf("note: %s\n", note)
	}
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
			fmt.Printf("Saving epoch log to: %s\n", fnm)
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
			fmt.Printf("Saving run log to: %s\n", fnm)
			defer ss.RunFile.Close()
		}
	}
	if ss.SaveWts {
		fmt.Printf("Saving final weights per run\n")
	}
	fmt.Printf("Running %d Runs\n", ss.MaxRuns)
	ss.Train()
}
