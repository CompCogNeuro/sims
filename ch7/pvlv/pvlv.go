// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	_ "reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/emer/emergent/env"
	"github.com/emer/emergent/stepper"
	_ "github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/etview"
	_ "github.com/emer/etable/split"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ints"
	"github.com/goki/mat32"

	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"

	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/leabra/leabra"
	"github.com/emer/leabra/pvlv"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"

	"github.com/emer/leabra/examples/pvlv/data"
)

var TheSim Sim // this is in a global mainly for debugging--otherwise it can be impossible to find

func main() {
	// TheSim is the overall state for this simulation
	TheSim.VerboseInit, TheSim.LayerThreads = TheSim.CmdArgs() // doesn't return if nogui command line arg set
	TheSim.New()
	TheSim.Config()
	gimain.Main(func() { // this starts the GUI
		guirun(&TheSim)
	})
}

func guirun(ss *Sim) {
	ss.InitSim()
	win := ss.ConfigGui()
	win.StartEventLoop()
}

// MonitorVal is similar to the Neuron field mechanism, but allows us to implement monitors for arbitrary
// quanities without messing with fields that are intrinsic to the workings of our model.
type MonitorVal interface {
	GetMonitorVal([]string) float64
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that

type Sim struct {
	RunParamsNm       string                `inactive:"+" desc:"Name of the current run. Use menu above to set"`
	RunParams         *data.RunParams       `desc:"For sequences of conditions"`
	ConditionParamsNm string                `inactive:"+" desc:"name of current ConditionParams"`
	ConditionParams   *data.ConditionParams `desc:"pointer to current ConditionParams"`
	Tag               string                `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"`
	Params            params.Sets           `view:"no-inline" desc:"pvlv-specific network parameters"`
	ParamSet          string
	//StableParams                 params.Set        `view:"no-inline" desc:"shouldn't need to change these'"`
	//MiscParams                   params.Set        `view:"no-inline" desc:"misc params -- network specs"`
	//AnalysisParams               params.Set        `view:"no-inline" desc:"??"`
	Env PVLVEnv `desc:"environment -- PVLV environment"`
	//TestEnv                      PVLVEnv           `desc:"Testing environment -- PVLV environment"`
	devMenuSetup                 bool              `view:"-" desc:"stepping menu layout. Default is one button, true means original \"wide\" setup"`
	StepsToRun                   int               `view:"-" desc:"number of StopStepGrain steps to execute before stopping"`
	nStepsBox                    *gi.SpinBox       `view:"-"`
	OrigSteps                    int               `view:"-" desc:"saved number of StopStepGrain steps to execute before stopping"`
	StepGrain                    StepGrain         `view:"-" desc:"granularity for the Step command"`
	StopStepCondition            StopStepCond      `desc:"granularity for conditional stop"`
	StopConditionTrialNameString string            `desc:"if StopStepCond is TrialName or NotTrialName, this string is used for matching the current AlphaTrialName"`
	StopStepCounter              env.Ctr           `inactive:"+" view:"-" desc:"number of times we've hit whatever StopStepGrain is set to'"`
	StepMode                     bool              `view:"-" desc:"running from Step command?"`
	TestMode                     bool              `inactive:"+" desc:"testing mode, no training"`
	CycleLogUpdt                 leabra.TimeScales `desc:"time scale for updating CycleOutputData. NOTE: Only Cycle and Quarter are currently implemented"`
	NetTimesCycleQtr             bool              `desc:"turn this OFF to see cycle-level updating"`
	TrialAnalysisTimeLogInterval int
	TrialAnalUpdateCmpGraphs     bool                `desc:"turn off to preserve existing cmp graphs - else saves cur as cmp for new run"`
	Net                          *pvlv.Network       `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	CycleOutputDataRows          int                 `desc:"maximum number of rows for CycleOutputData"`
	CycleOutputData              *etable.Table       `view:"no-inline" desc:"Cycle-level output data"`
	CycleDataPlot                *eplot.Plot2D       `view:"no-inline" desc:"Fine-grained trace data"`
	CycleOutputMetadata          map[string][]string `view:"-"`
	TimeLogBlock                 int                 `desc:"current block within current run phase"`
	TimeLogBlockAll              int                 `desc:"current block across all phases of the run"`
	Time                         leabra.Time         `desc:"leabra timing parameters and state"`
	ViewOn                       bool                `desc:"whether to update the network view while running"`
	TrainUpdt                    leabra.TimeScales   `desc:"at what time scale to update the display during training?  Anything longer than TrialGp updates at TrialGp in this model"`
	TestUpdt                     leabra.TimeScales   `desc:"at what time scale to update the display during testing?  Anything longer than TrialGp updates at TrialGp in this model"`
	TstRecLays                   []string            `view:"-" desc:"names of layers to record activations etc of during testing"`
	ContextModel                 ContextModel        `desc:"how to treat multi-part contexts. elemental=all parts, conjunctive=single context encodes parts, both=parts plus conjunctively encoded"`
	// internal state - view:"-"
	Win                       *gi.Window                  `view:"-" desc:"main GUI window"`
	NetView                   *netview.NetView            `view:"-" desc:"the network viewer"`
	ToolBar                   *gi.ToolBar                 `view:"-" desc:"the master toolbar"`
	WtsGrid                   *etview.TensorGrid          `view:"-" desc:"the weights grid view"`
	TrialTypeData             *etable.Table               `view:"no-inline" desc:"data for the TrialTypeData plot"`
	TrialTypeBlockFirstLog    *etable.Table               `view:"no-inline" desc:"data for the TrialTypeData plot"`
	TrialTypeBlockFirstLogCmp *etable.Table               `view:"no-inline" desc:"data for the TrialTypeData plot"`
	TrialTypeDataPlot         *eplot.Plot2D               `view:"no-inline" desc:"multiple views for different type of trials"`
	TrialTypeDataPerBlock     bool                        `desc:"clear the TrialTypeData plot between parts of a run"`
	TrialTypeSet              map[string]int              `view:"-"`
	GlobalTrialTypeSet        map[string]int              `view:"-"`
	TrialTypeBlockFirst       *eplot.Plot2D               `view:"-" desc:"block plot"`
	TrialTypeBlockFirstCmp    *eplot.Plot2D               `view:"-" desc:"block plot"`
	HistoryGraph              *eplot.Plot2D               `view:"-" desc:"trial history"`
	RealTimeData              *eplot.Plot2D               `view:"-" desc:"??"`
	SaveWts                   bool                        `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	NoGui                     bool                        `view:"-" desc:"if true, runing in no GUI mode"`
	RndSeed                   int64                       `desc:"the current random seed"`
	Stepper                   *stepper.Stepper            `view:"-"`
	SimHasRun                 bool                        `view:"-"`
	IsRunning                 bool                        `view:"-"`
	InitHasRun                bool                        `view:"-"`
	VerboseInit               bool                        `view:"-"`
	LayerThreads              bool                        `desc:"use per-layer threads"`
	TrialTypeBlockFirstLogged map[string]bool             `view:"-"`
	RunPlot                   *eplot.Plot2D               `view:"-" desc:"the run plot"`
	TrnEpcFile                *os.File                    `view:"-" desc:"log file"`
	RunFile                   *os.File                    `view:"-" desc:"log file"`
	ValsTsrs                  map[string]*etensor.Float32 `view:"-" desc:"for holding layer values"`
	LogSetParams              bool                        `view:"-" desc:"if true, print message for all params that are set"`
	Interactive               bool                        `view:"-" desc:"true iff running through the GUI"`
	StructView                *giv.StructView             `view:"-" desc:"structure view for this struct"`
	InputShapes               map[string][]int            `view:"-"`

	// master lists of various kinds of parameters
	MasterRunParams        data.RunParamsMap       `view:"no-inline" desc:"master list of RunParams records"`
	MasterConditionParams  data.ConditionParamsMap `view:"no-inline" desc:"master list of ConditionParams records"`
	MasterTrialBlockParams data.TrialBlockMap      `desc:"master list of BlockParams (sets of trial groups) records"`
	MaxConditions          int                     `view:"-" desc:"maximum number of conditions to run through"` // for non-GUI runs
	simOneTimeInit         sync.Once
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

func (ss *Sim) OpenCemerWeights(fName string) {
	err := ss.Net.OpenWtsCpp(gi.FileName(fName))
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	}
}

func (ss *Sim) New() {
	ss.InputShapes = map[string][]int{
		"StimIn":    pvlv.StimInShape,
		"ContextIn": pvlv.ContextInShape,
		"USTimeIn":  pvlv.USTimeInShape, // valence, cs, time, us
		"PosPV":     pvlv.USInShape,
		"NegPV":     pvlv.USInShape,
	}
	ss.Net = &pvlv.Network{}
	ss.CycleOutputData = &etable.Table{}
	ss.TrialTypeData = &etable.Table{}
	ss.TrialTypeBlockFirstLog = &etable.Table{}
	ss.TrialTypeBlockFirstLogCmp = &etable.Table{}
	ss.TrialTypeSet = map[string]int{}
	ss.GlobalTrialTypeSet = map[string]int{}
	ss.simOneTimeInit.Do(func() {
		ss.ValidateRunParams()
		ss.MasterConditionParams = data.AllConditionParams()
		ss.MasterRunParams = data.AllRunParams()
		ss.MasterTrialBlockParams = data.AllTrialBlocks()
		ss.Env = PVLVEnv{Nm: "Env", Dsc: "run environment"}
		ss.Env.New(ss)
		ss.StepsToRun = 1
		ss.StepGrain = SGTrial
		ss.StopStepCondition = SSNone
		ss.Stepper = stepper.New()
		ss.Stepper.StopCheckFn = ss.CheckStopCondition
		ss.Stepper.PauseNotifyFn = ss.NotifyPause
	})
	ss.Defaults()
	ss.Params = ParamSets
	ss.CycleOutputDataRows = 10000
	ss.InitHasRun = false

}

func (ss *Sim) Defaults() {
	defaultRunSeqNm := "PosAcq"
	ss.ContextModel = CONJUNCTIVE
	ss.RunParamsNm = defaultRunSeqNm
	err := ss.SetRunParams()
	if err != nil {
		panic(err)
	}
	ss.TrainUpdt = leabra.AlphaCycle
	ss.TestUpdt = leabra.AlphaCycle
	ss.CycleLogUpdt = leabra.Quarter
	ss.NetTimesCycleQtr = true
	ss.TrialAnalysisTimeLogInterval = 1
	ss.TrialAnalUpdateCmpGraphs = true
	ss.TrialTypeDataPerBlock = true
	ss.StopConditionTrialNameString = "_t3"
	ss.ViewOn = true
	ss.RndSeed = 1
}

func (ss *Sim) MaybeUpdate(train, exact bool, checkTS leabra.TimeScales) {
	if !ss.ViewOn {
		return
	}
	var ts leabra.TimeScales
	if train {
		ts = ss.TrainUpdt
	} else {
		ts = ss.TestUpdt
	}
	if (exact && ts == checkTS) || ts <= checkTS {
		ss.UpdateView()
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Top-level Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigOutputData()
	ss.InitSim()
}

func (ss *Sim) ConfigEnv() {
	ss.Env.Init(ss, true)
}

////////////////////////////////////////////////////////////////////////////////
// Init, utils

func (ss *Sim) Init(aki ki.Ki) {
	//ss.Layout.Init(aki)
}

func (ss *Sim) Ki() *Sim {
	return ss
}

type StopStepCond int

const (
	SSNone              StopStepCond = iota // None
	SSTrialNameMatch                        // Trial Name
	SSTrialNameNonmatch                     // Not Trial Name
	StopStepCondN
)

////go:generate stringer -type=StopStepCond -linecomment // moved to stringers.go
var KiT_StopStepCond = kit.Enums.AddEnum(StopStepCondN, kit.NotBitFlag, nil)

// Init restarts the run, and initializes everything, including network weights
// and resets the block log table
func (ss *Sim) InitSim() {
	ev := &ss.Env
	rand.Seed(ss.RndSeed)
	ss.Stepper.Init()
	ev.TrialInstances = data.NewTrialInstanceRecs(nil)
	err := ss.SetParams("", ss.VerboseInit) // all sheets
	if err != nil {
		fmt.Println(err)
	}
	err = ss.InitCondition(true)
	if err != nil {
		fmt.Println("ERROR: InitCondition failed in InitSim")
	}
	ss.Net.InitWts()
	ss.InitHasRun = true
	ss.VerboseInit = false
}

// NewRndSeed gets a new random seed based on current time -- otherwise uses
// the same random seed for every run
func (ss *Sim) NewRndSeed() {
	ss.RndSeed = time.Now().UnixNano()
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters() string {
	ev := &ss.Env
	return fmt.Sprintf("Condition:\t%d(%s)\tBlock:\t%03d\tTrial:\t%02d\tAlpha:\t%01d\tCycle:\t%03d\t\tName:\t%12v\t\t\t",
		ev.ConditionCt.Cur, ev.CurConditionParams.TrialBlkNm, ev.TrialBlockCt.Cur, ev.TrialCt.Cur, ev.AlphaCycle.Cur,
		ss.Time.Cycle, ev.AlphaTrialName) //, ev.USTimeInStr)
}

func (ss *Sim) UpdateView() {
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters())
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate()
	}
}

// Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
func (ss *Sim) NotifyStopped() {
	ss.Stepper.Stop()
	if ss.Win != nil {
		vp := ss.Win.WinViewport2D()
		if ss.ToolBar != nil {
			ss.ToolBar.UpdateActions()
		}
		ss.UpdateView()
		vp.SetNeedsFullRender()
	}
	fmt.Println("stopped")
}

// configure output data tables
func (ss *Sim) ConfigCycleOutputData(dt *etable.Table) {
	dt.SetMetaData("name", "CycleOutputData")
	dt.SetMetaData("desc", "Cycle-level output data")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	floatCols := []string{
		"VSPatchPosD1_0_Act", "VSPatchPosD2_0_Act",
		"VSPatchNegD1_0_Act", "VSPatchNegD2_0_Act",
		"VSMatrixPosD1_0_Act", "VSMatrixPosD2_0_Act",
		"VSMatrixNegD1_0_Act", "VSMatrixNegD2_0_Act",
		//
		//"VSMatrixPosD1_0_ModNet", "VSMatrixPosD1_0_DA",
		//"VSMatrixPosD2_0_ModNet", "VSMatrixPosD2_0_DA",

		"PosPV_0_Act", "StimIn_0_Act", "ContextIn_0_Act", "USTimeIn_0_Act",

		"VTAp_0_Act", "LHbRMTg_0_Act", "PPTg_0_Act",
		//"VTAp_0_PPTgDApt", "VTAp_0_LHbDA", "VTAp_0_PosPVAct", "VTAp_0_VSPosPVI", "VTAp_0_VSNegPVI", "VTAp_0_BurstLHbDA",
		//"VTAp_0_DipLHbDA", "VTAp_0_TotBurstDA", "VTAp_0_TotDipDA", "VTAp_0_NetDipDA", "VTAp_0_NetDA", "VTAp_0_SendVal",
		//
		//"LHbRMTg_0_VSPatchPosD1", "LHbRMTg_0_VSPatchPosD2","LHbRMTg_0_VSPatchNegD1","LHbRMTg_0_VSPatchNegD2",
		"LHbRMTg_0_VSMatrixPosD1", "LHbRMTg_0_VSMatrixPosD2", "LHbRMTg_0_VSMatrixNegD1", "LHbRMTg_0_VSMatrixNegD2",
		//"LHbRMTg_0_VSPatchPosNet", "LHbRMTg_0_VSPatchNegNet","LHbRMTg_0_VSMatrixPosNet","LHbRMTg_0_VSMatrixNegNet",
		//"LHbRMTg_0_PosPV", "LHbRMTg_0_NegPV","LHbRMTg_0_NetPos","LHbRMTg_0_NetNeg",

		//"CElAcqPosD1_0_ModAct", "CElAcqPosD1_0_PVAct",
		//"CElAcqPosD1_0_ModLevel", "CElAcqPosD1_0_ModLrn",
		//"CElAcqPosD1_0_Act", "CElAcqPosD1_0_ActP", "CElAcqPosD1_0_ActQ0", "CElAcqPosD1_0_ActM",
		//"CElAcqPosD1_1_ModPoolAvg", "CElAcqPosD1_1_PoolActAvg", "CElAcqPosD1_1_PoolActMax",
		//
		//"CElExtPosD2_0_ModAct", "CElExtPosD2_0_ModLevel", "CElExtPosD2_0_ModNet", "CElExtPosD2_0_ModLrn",
		//"CElExtPosD2_0_Act", "CElExtPosD2_0_Ge", "CElExtPosD2_0_Gi", "CElExtPosD2_0_Inet", "CElExtPosD2_0_GeRaw",
		//
		//"BLAmygPosD1_3_Act", "BLAmygPosD1_3_ModAct", "BLAmygPosD1_3_ActDiff", "BLAmygPosD1_3_ActQ0",
		//"BLAmygPosD1_3_ModLevel", "BLAmygPosD1_3_ModNet", "BLAmygPosD1_3_ModLrn", "BLAmygPosD1_3_DA",
		//"BLAmygPosD1_1_PoolActAvg", "BLAmygPosD1_1_PoolActMax", "BLAmygPosD1_2_PoolActAvg", "BLAmygPosD1_2_PoolActMax",
		//
		//"BLAmygPosD2_5_Act", "BLAmygPosD2_5_ModAct", "BLAmygPosD2_5_ActDiff", "BLAmygPosD2_5_ActQ0",
		//"BLAmygPosD2_5_ModLevel", "BLAmygPosD2_5_ModNet", "BLAmygPosD2_5_ModLrn", "BLAmygPosD2_5_DA",

		"CEmPos_0_Act",
	}
	ss.CycleOutputMetadata = make(map[string][]string, len(floatCols))

	sch := etable.Schema{}
	sch = append(sch, etable.Column{Name: "Cycle", Type: etensor.INT32})
	sch = append(sch, etable.Column{Name: "GlobalStep", Type: etensor.INT32})
	for _, colName := range floatCols {
		parts := strings.Split(colName, "_")
		idx := parts[1]
		val := parts[2]
		var md []string
		sch = append(sch, etable.Column{Name: colName, Type: etensor.FLOAT64})
		md = append(md, val)
		md = append(md, idx)
		ss.CycleOutputMetadata[colName] = md
	}
	dt.SetFromSchema(sch, ss.CycleOutputDataRows)
}

func (ss *Sim) ConfigCycleOutputDataPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "CycleOutputData"
	plt.Params.XAxisCol = "GlobalStep"
	plt.Params.XAxisLabel = "Cycle"
	plt.Params.Type = eplot.XY
	plt.SetTable(dt)

	for iCol := range dt.ColNames {
		colName := dt.ColNames[iCol]
		var colOnOff, colFixMin, colFixMax bool
		var colMin, colMax float64
		switch colName {
		case "Cycle":
			colOnOff = eplot.Off
			colFixMin = eplot.FixMin
			colMin = 0
			colFixMax = eplot.FixMax
			colMax = 99
		case "GlobalStep":
			colOnOff = eplot.Off
			colFixMin = eplot.FixMin
			colMin = 0
			colFixMax = eplot.FixMax
			colMax = float64(dt.Rows - 1)
		case "StimIn_0_Act", "ContextIn_0_Act", "PosPV_0_Act":
			colOnOff = eplot.Off
			colFixMin = eplot.FixMin
			colMin = -1.25
			colFixMax = eplot.FixMax
			colMax = 1.25
		default:
			colOnOff = eplot.Off
			colFixMin = eplot.FixMin
			colMin = -1.25
			colFixMax = eplot.FixMax
			colMax = 1.25
		}
		plt.SetColParams(colName, colOnOff, colFixMin, colMin, colFixMax, colMax)
	}
	return plt
}

func (ss *Sim) ConfigOutputData() {
	ss.ConfigCycleOutputData(ss.CycleOutputData)
	ss.ConfigTrialTypeTables(0)
}

func (ss *Sim) ConfigTrialTypeTables(nRows int) {
	ss.ConfigTrialTypeBlockFirstLog(ss.TrialTypeBlockFirstLog, "TrialTypeBlockFirst", nRows)
	ss.ConfigTrialTypeBlockFirstLog(ss.TrialTypeBlockFirstLogCmp, "TrialTypeBlockFirstCmp", nRows)
	ss.ConfigTrialTypeData(ss.TrialTypeData)
}

// end output data config

// configure plots
func (ss *Sim) ConfigTrialTypeBlockFirstLog(dt *etable.Table, name string, nRows int) {
	dt.SetMetaData("name", name)
	dt.SetMetaData("desc", "Multi-block monitor")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))
	sch := etable.Schema{}

	colNames := []string{
		"GlobalTrialBlock", "VTAp_act", "BLAmygD1_US0_act", "BLAmygD2_US0_act",
		"CElAcqPosD1_US0_act", "CElExtPosD2_US0_act", "CElAcqNegD2_US0_act",
		"VSMatrixPosD1_US0_act", "VSMatrixPosD2_US0_act",
	}

	for _, colName := range colNames {
		if colName == "GlobalTrialBlock" {
			sch = append(sch, etable.Column{Name: colName, Type: etensor.INT64})
		} else {
			sch = append(sch, etable.Column{Name: colName, Type: etensor.FLOAT64, CellShape: []int{nRows, 1}, DimNames: []string{"Tick", "Value"}})
		}
	}
	dt.SetFromSchema(sch, nRows)
	dt.SetNumRows(nRows)
}

func (ss *Sim) ConfigTrialTypeData(dt *etable.Table) {
	dt.SetMetaData("name", "TrialTypeData")
	dt.SetMetaData("desc", "Plot of activations for different trial types")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	colNames := []string{
		"TrialType",
		"GlobalTrialBlock",
		"VTAp_act", "LHbRMTg_act",
		"CElAcqPosD1_US0_act", "CElExtPosD2_US0_act",
		"VSPatchPosD1_US0_act", "VSPatchPosD2_US0_act",
		"VSPatchNegD1_US0_act", "VSPatchNegD2_US0_act",
		"VSMatrixPosD1_US0_act", "VSMatrixPosD2_US0_act",
		"VSMatrixNegD1_US0_act", "VSMatrixNegD2_US0_act",
		"CElAcqNegD2_US0_act", "CElExtNegD1_US0_act",
		"CEmPos_US0_act", "VTAn_act",
	}
	sch := etable.Schema{}

	for _, colName := range colNames {
		if colName == "TrialType" {
			sch = append(sch, etable.Column{Name: colName, Type: etensor.STRING})
		} else {
			sch = append(sch, etable.Column{Name: colName, Type: etensor.FLOAT64})
		}
	}
	dt.SetFromSchema(sch, len(ss.TrialTypeSet))
}

func (ss *Sim) ConfigTrialTypeDataPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "TrialTypeData"
	plt.Params.XAxisCol = "TrialType"
	plt.Params.XAxisLabel = " "
	plt.Params.XAxisRot = 45.0
	plt.Params.Type = eplot.XY
	plt.Params.LineWidth = 1.25
	plt.SetTable(dt)

	for _, colName := range dt.ColNames {
		var colOnOff, colFixMin, colFixMax bool
		var colMin, colMax float64
		colFixMin = eplot.FixMin
		colMin = -2
		colFixMax = eplot.FixMax
		colMax = 2
		switch colName {
		case "VTAp_act", "CElAcqPosD1_US0_act", "VSPatchPosD1_US0_act", "VSPatchPosD2_US0_act", "LHbRMTg_act":
			colOnOff = eplot.On
		default:
			colOnOff = eplot.Off
		}
		plt.SetColParams(colName, colOnOff, colFixMin, colMin, colFixMax, colMax)
	}
	return plt
}

func (ss *Sim) ConfigTrialTypeBlockFirstPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "All Trial Blocks"
	plt.Params.XAxisCol = "GlobalTrialBlock"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("GlobalTrialBlock", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	for _, colNm := range []string{"VTAp_act",
		"BLAmygD2_US0_act", "BLAmygD1_US0_act",
		"CElAcqPosD1_US0_act", "CElExtPosD2_US0_act",
		"VSMatrixPosD1_US0_act", "VSMatrixPosD2_US0_act", "CElAcqNegD2_US0_act"} {
		plt.SetColParams(colNm, eplot.Off, eplot.FixMin, -1, eplot.FixMax, 1)
		cp := plt.ColParams(colNm)
		cp.TensorIdx = -1
		if colNm == "VTAp_act" {
			cp.On = eplot.On
		}
	}

	return plt
}

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.ViewDefaults()
	pos := nv.Scene().Camera.Pose.Pos
	nv.Scene().Camera.Pose.Pos.Set(pos.X, pos.Y, 2)
	nv.Scene().Camera.LookAt(mat32.Vec3{Y: 0.5, Z: 1}, mat32.Vec3{Y: 1})
	ctrs := nv.Counters()
	ctrs.SetProp("font-family", "Go Mono")
	nv.Record(ss.Counters())
}

func (ss *Sim) Stopped() bool {
	return ss.Stepper.RunState == stepper.Stopped
}

func (ss *Sim) Paused() bool {
	return ss.Stepper.RunState == stepper.Paused
}

var CemerWtsFname = ""

func FileViewLoadCemerWts(vp *gi.Viewport2D) {
	giv.FileViewDialog(vp, CemerWtsFname, ".svg", giv.DlgOpts{Title: "Open SVG"}, nil,
		vp.Win, func(recv, send ki.Ki, sig int64, data interface{}) {
			if sig == int64(gi.DialogAccepted) {
				dlg, _ := send.(*gi.Dialog)
				CemerWtsFname = giv.FileViewDialogValue(dlg)
				err := TheSim.Net.OpenWtsCpp(gi.FileName(CemerWtsFname))
				if err != nil {
					fmt.Println(err)
				}
			}
		})
}

func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1600
	gi.SetAppName("pvlv")
	gi.SetAppAbout(`Current version of the Primary Value Learned Value model of the phasic dopamine signaling system. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch7/pvlv/README.md">README.md on GitHub</a>.</p>`)

	win := gi.NewMainWindow("pvlv", "PVLV", width, height)
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
	ss.StructView = sv

	tv := gi.AddNewTabView(split, "tv")

	nv := tv.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	nv.Var = "Act"
	nv.Params.LayNmSize = 0.02
	nv.SetNet(ss.Net)
	ss.NetView = nv
	ss.ConfigNetView(nv)

	cb := gi.AddNewComboBox(tbar, "RunParams")
	var seqKeys []string
	for key := range ss.MasterRunParams {
		seqKeys = append(seqKeys, key)
	}
	sort.Strings(seqKeys)
	cb.ItemsFromStringList(seqKeys, false, 50)
	cb.ComboSig.Connect(mfr.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.RunParamsNm = data.(string)
		err := ss.SetRunParams()
		if err != nil {
			fmt.Printf("error setting run sequence: %v\n", err)
		}
		fmt.Printf("ComboBox %v selected index: %v data: %v\n", send.Name(), sig, data)
	})
	cb.SetCurVal(ss.RunParamsNm)

	eplot.PlotColorNames = []string{ // these are set to give a good set of colors in the TrialTypeData plot
		"yellow", "black", "blue", "red", "ForestGreen", "lightgreen", "purple", "orange", "brown", "navy",
		"cyan", "magenta", "tan", "salmon", "blue", "SkyBlue", "pink", "chartreuse"}

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TrialTypeData").(*eplot.Plot2D)
	ss.TrialTypeDataPlot = ss.ConfigTrialTypeDataPlot(plt, ss.TrialTypeData)

	frm := gi.AddNewFrame(tv, "TrialTypeBlockFirst", gi.LayoutVert)
	frm.SetStretchMax()
	pltCmp := frm.AddNewChild(eplot.KiT_Plot2D, "TrialTypeBlockFirst_cmp").(*eplot.Plot2D)
	pltCmp.SetStretchMax()
	pltLower := frm.AddNewChild(eplot.KiT_Plot2D, "TrialTypeBlockFirst").(*eplot.Plot2D)
	pltLower.SetStretchMax()
	tv.AddTab(frm, "TrialTypeBlockFirst")
	ss.TrialTypeBlockFirst = ss.ConfigTrialTypeBlockFirstPlot(pltLower, ss.TrialTypeBlockFirstLog)
	ss.TrialTypeBlockFirstCmp = ss.ConfigTrialTypeBlockFirstPlot(pltCmp, ss.TrialTypeBlockFirstLogCmp)

	//plt = tv.AddNewTab(eplot.KiT_Plot2D, "HistoryGraph").(*eplot.Plot2D)
	//ss.HistoryGraph = ss.ConfigHistoryGraph(plt, ss.HistoryGraphData)
	//
	//plt = tv.AddNewTab(eplot.KiT_Plot2D, "RealTimeData").(*eplot.Plot2D)
	//ss.RealTimeData = ss.ConfigRealTimeData(plt, ss.RealTimeDataLog)

	input := tv.AddNewTab(etview.KiT_TableView, "StdInputData").(*etview.TableView)
	input.SetName("StdInputData")
	input.SetTable(ss.Env.StdInputData, nil)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "CycleOutputData").(*eplot.Plot2D)
	ss.CycleDataPlot = ss.ConfigCycleOutputDataPlot(plt, ss.CycleOutputData)

	split.SetSplits(.3, .7)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update", Tooltip: "Block init code. Global variables retain current values unless reset in the init code", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Stepper.Stop()
		if !ss.InitHasRun {
			ss.InitSim()
		}
		answeredInitWts := true // hack to workaround lack of a true modal dialog
		if ss.SimHasRun {
			answeredInitWts = false
			gi.ChoiceDialog(ss.Win.Viewport, gi.DlgOpts{Title: "Init weights?", Prompt: "Initialize network weights?"},
				[]string{"Yes", "No"}, ss.Win.This(),
				func(recv, send ki.Ki, sig int64, data interface{}) {
					if sig == 0 {
						fmt.Println("initializing weights")
						ss.Net.InitWts()
						ss.SimHasRun = false
					}
					answeredInitWts = true
				})
		}
		for answeredInitWts == false {
			time.Sleep(1 * time.Second)
		}
		_ = ss.InitRun()
		ss.UpdateView()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Run", Icon: "run", Tooltip: "Run the currently selected scenario. If not initialized, will run initialization first",
		UpdateFunc: func(act *gi.Action) {
			act.SetActiveStateUpdt(!ss.IsRunning)
		}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.IsRunning = true
		tbar.UpdateActions()
		if !ss.InitHasRun {
			fmt.Println("Initializing...")
			ss.InitSim()
		}
		if !ss.SimHasRun {
			_ = ss.InitRun()
		}
		if !ss.Stepper.Active() {
			if ss.Stopped() {
				ss.SimHasRun = true
				ss.Stepper.Enter(stepper.Running)
				go ss.ExecuteRun()
			} else if ss.Paused() {
				ss.Stepper.Enter(stepper.Running)
			}
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Stop the current program at its next natural stopping point (i.e., cleanly stopping when appropriate chunks of computation have completed).", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		fmt.Println("STOP!")
		ss.Stepper.Pause()
		ss.IsRunning = false
		ss.ToolBar.UpdateActions()
		ss.Win.WinViewport2D().SetNeedsFullRender()
	})

	if ss.devMenuSetup {
		tbar.AddSeparator("stepSep")
		stepLabel := gi.AddNewLabel(tbar, "stepLabel", "Step to end of:")
		stepLabel.SetProp("font-size", "large")

		tbar.AddAction(gi.ActOpts{Label: "Cycle", Icon: "step-fwd", Tooltip: "Step to the end of a Cycle.",
			UpdateFunc: func(act *gi.Action) {
				act.SetActiveStateUpdt(!ss.IsRunning)
			}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.RunSteps(Cycle, tbar)
		})

		tbar.AddAction(gi.ActOpts{Label: "Quarter", Icon: "step-fwd", Tooltip: "Step to the end of a Quarter.",
			UpdateFunc: func(act *gi.Action) {
				act.SetActiveStateUpdt(!ss.IsRunning)
			}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.RunSteps(Quarter, tbar)
		})

		tbar.AddAction(gi.ActOpts{Label: "Minus Phase", Icon: "step-fwd", Tooltip: "Step to the end of the Minus Phase.",
			UpdateFunc: func(act *gi.Action) {
				act.SetActiveStateUpdt(!ss.IsRunning)
			}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.RunSteps(AlphaMinus, tbar)
		})

		//tbar.AddAction(gi.ActOpts{Label: "Plus Phase", Icon: "step-fwd", Tooltip: "Step to the end of the Plus Phase.",
		//	UpdateFunc: func(act *gi.Action) {
		//		act.SetActiveStateUpdt(!ss.IsRunning)
		//	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		//	ss.RunSteps(AlphaPlus, tbar)
		//})

		tbar.AddAction(gi.ActOpts{Label: "Alpha Cycle", Icon: "step-fwd", Tooltip: "Step to the end of an Alpha Cycle.",
			UpdateFunc: func(act *gi.Action) {
				act.SetActiveStateUpdt(!ss.IsRunning)
			}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.RunSteps(AlphaFull, tbar)
		})

		tbar.AddAction(gi.ActOpts{Label: "Selected grain -->", Icon: "fast-fwd", Tooltip: "Step by the selected granularity.",
			UpdateFunc: func(act *gi.Action) {
				act.SetActiveStateUpdt(!ss.IsRunning)
			}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.RunSteps(ss.StepGrain, tbar)
		})
	} else {
		tbar.AddAction(gi.ActOpts{Label: "StepRun", Icon: "fast-fwd", Tooltip: "Step by the selected granularity.",
			UpdateFunc: func(act *gi.Action) {
				act.SetActiveStateUpdt(!ss.IsRunning)
			}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.RunSteps(ss.StepGrain, tbar)
		})
		stepLabel := gi.AddNewLabel(tbar, "stepLabel", "StepGrain:")
		stepLabel.SetProp("font-size", "large")

	}

	sg := gi.AddNewComboBox(tbar, "grainMenu")
	sg.Editable = false
	var stepKeys []string
	maxLen := 0
	for i := 0; i < int(StepGrainN); i++ {
		s := StepGrain(i).String()
		maxLen = ints.MaxInt(maxLen, len(s))
		stepKeys = append(stepKeys, s)
	}
	sg.ItemsFromStringList(stepKeys, false, maxLen)
	sg.ComboSig.Connect(tbar, func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.StepGrain = StepGrain(sig)
	})
	sg.SetCurVal(ss.StepGrain.String())

	nLabel := gi.AddNewLabel(tbar, "n", "StepN:")
	nLabel.SetProp("font-size", "large")
	ss.nStepsBox = gi.AddNewSpinBox(tbar, "nStepsSpinbox")
	stepsProps := ki.Props{"has-min": true, "min": 1, "has-max": false, "step": 1, "pagestep": 10}
	ss.nStepsBox.SetProps(stepsProps, true)
	ss.nStepsBox.SetValue(1)
	ss.nStepsBox.SpinBoxSig.Connect(tbar.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.StepsToRun = int(ss.nStepsBox.Value)
	})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch7/pvlv/README.md")
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

	fmen := win.MainMenu.ChildByName("File", 0).(*gi.Action)

	fmen.Menu.AddAction(gi.ActOpts{Label: "Load CEmer weights", Tooltip: "load a CEmer weights file", Data: ss}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		FileViewLoadCemerWts(vp)
	})

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

	ss.UpdateView()
	vp.SetNeedsFullRender()
	ss.UpdateView()
	vp.SetNeedsFullRender()
	return win

}

func (ss *Sim) RunSteps(grain StepGrain, tbar *gi.ToolBar) {
	//fmt.Printf("ss.StepsToRun=%d, widget=%d, stepper=%d\n", ss.StepsToRun, int(ss.nStepsBox.Value), ss.Stepper.StepsPer)
	if !ss.IsRunning {
		ss.IsRunning = true
		tbar.UpdateActions()
		if !ss.SimHasRun {
			fmt.Println("Initializing...")
			ss.InitSim()
			_ = ss.InitRun()
		}
		if int(ss.nStepsBox.Value) != ss.StepsToRun ||
			int(ss.nStepsBox.Value) != ss.Stepper.StepsPer ||
			ss.StepsToRun != ss.Stepper.StepsPer ||
			ss.Stepper.StepGrain != int(ss.StepGrain) {
			ss.StepsToRun = int(ss.nStepsBox.Value)
			ss.OrigSteps = ss.StepsToRun
			ss.Stepper.ResetParams(ss.StepsToRun, int(ss.StepGrain))
		}
		if ss.Stopped() {
			ss.SimHasRun = true
			ss.OrigSteps = ss.StepsToRun
			ss.Stepper.Start(int(grain), ss.StepsToRun)
			ss.ToolBar.UpdateActions()
			go ss.ExecuteRun()
		} else if ss.Paused() {
			ss.Stepper.StepGrain = int(grain)
			ss.Stepper.StepsPer = ss.StepsToRun
			ss.Stepper.Enter(stepper.Stepping)
			ss.ToolBar.UpdateActions()
		}
	}
}

// SetParams sets the params for "Base" and then current ParamSet.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParams(sheet string, setMsg bool) error {
	if sheet == "" {
		// this is important for catching typos and ensuring that all sheets can be used
		err := ss.Params.ValidateSheets([]string{"Network"})
		if err != nil {
			fmt.Printf("error in validate sheets for Network: %v\n", err)
		}
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
			applied, err := ss.Net.ApplyParams(netp, setMsg)
			if err != nil {
				fmt.Printf("error when applying %v, applied=%v, err=%v\n", netp, applied, err)
			}
		}
	}

	if sheet == "" || sheet == "Sim" {
		simp, ok := pset.Sheets["Sim"]
		if ok {
			applied, err := simp.Apply(ss, setMsg)
			if err != nil {
				fmt.Printf("error when applying %v, applied=%v, err=%v\n", simp, applied, err)
			}
		}
	}
	return err
}

// These props register Save methods so they can be used
var SimProps = ki.Props{
	"max-width":  -1,
	"max-height": -1,
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
		{"OpenCemerWeights", ki.Props{
			"desc": "open network weights from CEmer-format file",
			"icon": "file-open",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".wts,.wts.gz",
				}},
			},
		}},
	},
}

func (ss *Sim) RunEnd() {
	//ss.LogRun(ss.RunLog)
}

// ParamsName returns name of current set of parameters
func (ss *Sim) ParamsName() string {
	if ss.ParamSet == "" {
		return "Base"
	}
	return ss.RunParams.Nm
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

// RunBlockName returns a string with the run and block numbers with leading zeros, suitable
// for using in weights file names.  Uses 3, 5 digits for each.
func (ss *Sim) RunBlockName(run, epc int) string {
	return fmt.Sprintf("%03d_%05d", run, epc)
}

// WeightsFileName returns default current weights file name
func (ss *Sim) WeightsFileName() string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + ss.RunBlockName(ss.Env.ConditionCt.Cur, ss.Env.TrialBlockCt.Cur) + ".wts.gz"
}

// LogFileName returns default log file name
func (ss *Sim) LogFileName(lognm string) string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + lognm + ".csv"
}

//////////////////////////////////////////////
//  TrnEpcLog

// LogTrnBlk adds data from current block to the TrnBlkLog table.
// computes block averages prior to logging.
func (ss *Sim) LogTrnBlk() {
	ss.TrialTypeBlockFirst.GoUpdate()
}

func (ss *Sim) SetRunParams() error {
	var err error = nil
	if ss.RunParams == nil || ss.RunParamsNm != ss.RunParams.Nm {
		oldSeqParams := ss.RunParams
		newSeqParams, found := ss.GetRunParams(ss.RunParamsNm)
		if !found {
			err = errors.New(fmt.Sprintf("RunSeq \"%v\" was not found!", ss.RunParamsNm))
			fmt.Println(err)
			return err
		} else {
			ss.RunParams = newSeqParams
			newBlockParams, found := ss.GetConditionParams(ss.RunParams.Cond1Nm)
			if !found {
				err = errors.New(fmt.Sprintf("RunParams step 1 \"%v\" was not found!", ss.RunParams.Cond1Nm))
				gi.PromptDialog(nil, gi.DlgOpts{Title: "RunParams step not found", Prompt: err.Error()}, gi.AddOk, gi.NoCancel, nil, nil)
				ss.RunParams = oldSeqParams
				return err
			} else {
				ss.ConditionParams = newBlockParams
				ss.Env.CurConditionParams = ss.ConditionParams
				ss.ConditionParamsNm = ss.ConditionParams.Nm
				return nil
			}
		}
	}
	return nil
}

// InitCondition intializes a new run of the model, using the Env.ConditionCt counter
// for the new run value
func (ss *Sim) InitRun() error {
	ev := &ss.Env
	err := ss.SetRunParams()
	if err != nil {
		return err
	}
	ev.GlobalStep = 0
	ss.ClearCycleData()
	ss.TrialTypeData.SetNumRows(0)
	tgNmMap, _, err := ss.RunSeqTrialTypes(ss.RunParams)
	ss.TrialTypeBlockFirstLogged = map[string]bool{}
	for key := range tgNmMap {
		ss.TrialTypeBlockFirstLogged[key] = false
	}
	ss.ConfigTrialTypeTables(len(tgNmMap)) // max number of rows for entire sequence, for TrialTypeBlockFirst only
	err = ss.InitCondition(true)
	if err != nil {
		fmt.Println("ERROR: InitCondition failed")
	}
	ss.UpdateView()
	ss.Win.Viewport.SetNeedsFullRender()
	return nil
}

// InitCondition intializes a new run of the model, using the Env.ConditionCt counter
// for the new run value
func (ss *Sim) InitCondition(firstInSeq bool) (err error) {
	ev := &ss.Env
	err = ss.SetRunParams()
	if err != nil {
		return err
	}
	ss.Time.Reset()
	ss.Net.InitActs()
	ss.TimeLogBlock = 0
	ev.Init(ss, firstInSeq)
	if firstInSeq || ss.TrialTypeDataPerBlock {
		_ = ss.SetTrialTypeDataXLabels()
	}
	return nil
}

func (ss *Sim) GetRunConditions(runParams *data.RunParams) *[5]*data.ConditionParams {
	var found bool
	conditions := &[5]*data.ConditionParams{}
	conditions[0], found = ss.GetConditionParams(runParams.Cond1Nm)
	if !found {
		fmt.Println("Condition", runParams.Cond1Nm, "was not found")
	}
	conditions[1], found = ss.GetConditionParams(runParams.Cond2Nm)
	if !found {
		fmt.Println("Condition", runParams.Cond2Nm, "was not found")
	}
	conditions[2], found = ss.GetConditionParams(runParams.Cond3Nm)
	if !found {
		fmt.Println("Condition", runParams.Cond3Nm, "was not found")
	}
	conditions[3], found = ss.GetConditionParams(runParams.Cond4Nm)
	if !found {
		fmt.Println("Condition", runParams.Cond4Nm, "was not found")
	}
	conditions[4], found = ss.GetConditionParams(runParams.Cond5Nm)
	if !found {
		fmt.Println("Condition", runParams.Cond5Nm, "was not found")
	}
	return conditions
}

// Run
// Block the currently selected sequence of runs
// Each run has its own set of trial types
func (ss *Sim) ExecuteRun() bool {
	ss.Net.InitActs()
	allDone := false
	var err error
	ev := &ss.Env
	conditions := ss.GetRunConditions(ss.RunParams)
	activateCondition := func(i int, blockParams *data.ConditionParams) {
		ev.CurConditionParams = blockParams
		ss.ConditionParams = ev.CurConditionParams
		ss.ConditionParamsNm = ss.ConditionParams.Nm
		err = ss.InitCondition(i == 0)
		if err != nil {
			fmt.Println("ERROR: InitCondition failed in activateCondition")
		}
		ss.Win.WinViewport2D().SetNeedsFullRender()
	}
	ss.TimeLogBlockAll = 0
	for i, condition := range conditions {
		if condition.Nm == "NullStep" {
			allDone = true
			break
		}
		activateCondition(i, condition)
		ss.ExecuteBlocks(true)
		if allDone || ss.Stopped() {
			break
		}
		if ss.ViewOn && ss.TrainUpdt >= leabra.Run {
			ss.UpdateView()
		}
		ss.Stepper.StepPoint(int(Condition))
	}
	ss.Stepper.Stop()
	ss.IsRunning = false
	return allDone
}

// end Run

// Multiple trial types
func (ss *Sim) ExecuteBlocks(seqRun bool) {
	ev := &ss.Env
	if !seqRun {
		ev.CurConditionParams = ss.ConditionParams
	}
	nDone := 0
	for i := 0; i < ev.CurConditionParams.NIters; i++ {
		ev.RunOneTrialBlk(ss)
		nDone++
	}
	ev.ConditionCt.Incr()
}

// end MultiTrial

// CheckStopCondition is called from within the Stepper.
// Since CheckStopCondition is called with the Stepper's lock held,
// it must not call any Stepper methods that set the lock. Rather, Stepper variables
// should be set directly, if need be.
func (ss *Sim) CheckStopCondition(_ int) bool {
	ev := &ss.Env
	ret := false
	switch ss.StopStepCondition {
	case SSNone:
		return false
	case SSTrialNameMatch:
		ret = strings.Contains(ev.AlphaTrialName, ss.StopConditionTrialNameString)
	case SSTrialNameNonmatch:
		ret = !strings.Contains(ev.AlphaTrialName, ss.StopConditionTrialNameString)
	default:
		ret = false
	}
	return ret
}

// NotifyPause is called from within the Stepper, with the Stepper's lock held.
// Stepper variables should be set directly, rather than calling Stepper methods,
// which would try to take the lock and then deadlock.
func (ss *Sim) NotifyPause() {
	if int(ss.StepGrain) != ss.Stepper.StepGrain {
		ss.Stepper.StepGrain = int(ss.StepGrain)
	}
	if ss.StepsToRun != ss.OrigSteps { // User has changed the step count while running
		ss.Stepper.StepsPer = ss.StepsToRun
		ss.OrigSteps = ss.StepsToRun
	}
	ss.IsRunning = false
	ss.ToolBar.UpdateActions()
	ss.UpdateView()
	ss.Win.Viewport.SetNeedsFullRender()
}

// end TrialGp and functions

// Monitors //

func IMax(x, y int) int {
	if x > y {
		return x
	} else {
		return y
	}
}

func (ss *Sim) RunSeqTrialTypes(rs *data.RunParams) (map[string]string, int, error) {
	steps := ss.GetRunConditions(rs)
	ticksPerGroup := 0
	var err error
	types := map[string]string{}
	fullStepMap := map[string]string{}
	for _, step := range steps {
		if step.Nm == "NullStep" {
			break
		}
		tgt, ticks, err := ss.GetBlockTrialTypes(step)
		ticksPerGroup = IMax(ticksPerGroup, ticks)
		if err != nil {
			return nil, 0, err
		}
		for long, short := range tgt {
			types[long] = short
		}
	}
	for long, short := range types {
		for i := 0; i < ticksPerGroup; i++ {
			is := strconv.Itoa(i)
			fullStepMap[long+"_t"+is] = short + is
		}
	}
	stepNames := sort.StringSlice{}
	for val := range fullStepMap {
		stepNames = append(stepNames, val)
	}
	sort.Sort(stepNames)
	ss.GlobalTrialTypeSet = map[string]int{}
	for i, name := range stepNames {
		ss.GlobalTrialTypeSet[name] = i
	}
	return fullStepMap, ticksPerGroup, err
}

func (ss *Sim) GetBlockTrialTypes(rp *data.ConditionParams) (map[string]string, int, error) {
	var err error
	ticks := 0
	cases := map[string]string{}
	ep, found := ss.MasterTrialBlockParams[rp.TrialBlkNm]
	valMap := map[pvlv.Valence]string{pvlv.POS: "+", pvlv.NEG: "-"}
	if !found {
		err := errors.New(fmt.Sprintf("TrialBlockParams %s was not found",
			rp.TrialBlkNm))
		return nil, 0, err
	}
	for _, tg := range ep {
		tSuffix := ""
		oSuffix := "_omit"
		val := tg.ValenceContext
		if strings.Contains(tg.TrialBlkName, "_test") {
			tSuffix = "_test"
		}
		parts := strings.Split(tg.TrialBlkName, "_")
		if parts[1] == "NR" {
			oSuffix = ""
		}
		longNm := fmt.Sprintf("%s_%s", tg.TrialBlkName, val)
		shortNm := tg.CS + valMap[val]
		if strings.Contains(longNm, "_test") {
			parts := strings.Split(longNm, "_")
			longNm = ""
			for i, part := range parts {
				isTest := part == "test"
				if !isTest && i != 0 {
					longNm += "_"
				}
				if !isTest {
					longNm += part
				}
			}
		}
		switch tg.USProb {
		case 1:
			cases[longNm+tSuffix] = shortNm + "*"
		case 0:
			cases[longNm+oSuffix+tSuffix] = shortNm + "~"
		default:
			cases[longNm+oSuffix+tSuffix] = shortNm + "~"
			cases[longNm+tSuffix] = shortNm + "*"
		}
		ticks = IMax(ticks, tg.AlphTicksPerTrialGp)
	}
	return cases, ticks, err
}

func (ss *Sim) SetTrialTypeDataXLabels() (nRows int) {
	tgNmMap := map[string]string{}
	var ticksPerGroup int

	if ss.TrialTypeDataPerBlock {
		types := map[string]string{}
		types, ticksPerGroup, _ = ss.GetBlockTrialTypes(ss.ConditionParams)
		for long, short := range types {
			for i := 0; i < ticksPerGroup; i++ {
				is := strconv.Itoa(i)
				tgNmMap[long+"_t"+is] = short + is
			}
		}
	} else {
		tgNmMap, _, _ = ss.RunSeqTrialTypes(ss.RunParams)
	}
	names := sort.StringSlice{}
	for val := range tgNmMap {
		names = append(names, val)
	}
	nRows = len(names)
	sort.Sort(names)
	dt := ss.TrialTypeData
	ss.TrialTypeSet = map[string]int{}
	for i, name := range names {
		ss.TrialTypeSet[name] = i
		dt.SetCellString("TrialType", i, name)
	}
	dt.UpdateColNameMap()
	dt.SetNumRows(nRows)
	return nRows
}

func (ss *Sim) LogTrialTypeData() {
	ev := &ss.Env
	dt := ss.TrialTypeData
	efdt := ss.TrialTypeBlockFirstLog
	row, _ := ss.TrialTypeSet[ev.AlphaTrialName]
	dt.SetCellString("TrialType", row, ev.AlphaTrialName)
	for _, colNm := range dt.ColNames {
		if colNm != "TrialType" && colNm != "GlobalTrialBlock" {
			parts := strings.Split(colNm, "_")
			lnm := parts[0]
			if parts[1] != "act" {
				// ??
			}
			tsr := ss.ValsTsr(lnm)
			ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
			err := ly.UnitValsTensor(tsr, "Act") // get minus phase act
			if err == nil {
				dt.SetCellTensor(colNm, row, tsr)
			} else {
				fmt.Println(err)
			}
			if !ss.TrialTypeBlockFirstLogged[ev.AlphaTrialName] {
				ss.TrialTypeBlockFirstLogged[ev.AlphaTrialName] = true
				vtaCol := ss.GlobalTrialTypeSet[ev.AlphaTrialName]
				efRow := ss.TimeLogBlockAll
				val := float64(tsr.Values[0])
				if efdt.Rows <= efRow {
					efdt.SetNumRows(efRow + 1)
					if efRow > 0 { // initialize from previous block to avoid weird-looking artifacts
						efdt.SetCellTensor(colNm, efRow, efdt.CellTensor(colNm, efRow-1))
					}
				}
				efdt.SetCellFloat("GlobalTrialBlock", efRow, float64(efRow))
				efdt.SetCellTensorFloat1D(colNm, efRow, vtaCol, val)
			}
		}
	}
	ss.TrialTypeDataPlot.GoUpdate()
}

func GetLeabraMonitorVal(ly *leabra.Layer, data []string) float64 {
	var val float32
	var err error
	var varIdx int
	valType := data[0]
	varIdx, err = pvlv.NeuronVarIdxByName(valType)
	if err != nil {
		varIdx, err = leabra.NeuronVarIdxByName(valType)
		if err != nil {
			fmt.Printf("index lookup failed for %v_%v_%v_%v: \n", ly.Name(), data[1], valType, err)
		}
	}
	unitIdx, err := strconv.Atoi(data[1])
	if err != nil {
		fmt.Printf("string to int conversion failed for %v_%v_%v%v: \n", ly.Name(), data[1], valType, err)
	}
	val = ly.UnitVal1D(varIdx, unitIdx)
	return float64(val)
}

func (ss *Sim) ClearCycleData() {
	for i := 0; i < ss.CycleOutputData.Rows; i++ {
		for _, colName := range ss.CycleOutputData.ColNames {
			ss.CycleOutputData.SetCellFloat(colName, i, 0)
		}
	}
}

func (ss *Sim) LogCycleData() {
	ev := &ss.Env
	var val float64
	dt := ss.CycleOutputData
	row := ev.GlobalStep
	alphaStep := ss.Time.Cycle + ev.AlphaCycle.Cur*100
	for _, colNm := range dt.ColNames {
		if colNm == "GlobalStep" {
			dt.SetCellFloat("GlobalStep", row, float64(ev.GlobalStep))
		} else if colNm == "Cycle" {
			dt.SetCellFloat(colNm, row, float64(alphaStep))
		} else {
			monData := ss.CycleOutputMetadata[colNm]
			parts := strings.Split(colNm, "_")
			lnm := parts[0]
			ly := ss.Net.LayerByName(lnm)
			switch ly.(type) {
			case *leabra.Layer:
				val = GetLeabraMonitorVal(ly.(*leabra.Layer), monData)
			default:
				val = ly.(MonitorVal).GetMonitorVal(monData)
			}
			dt.SetCellFloat(colNm, row, val)
		}
	}
	label := fmt.Sprintf("%20s: %3d", ev.AlphaTrialName, row)
	ss.CycleDataPlot.Params.XAxisLabel = label
	if ss.CycleLogUpdt == leabra.Quarter || row%25 == 0 {
		ss.CycleDataPlot.GoUpdate()
	}
}

// end TrialAnalysis functions

func (ss *Sim) BlockMonitor() {
	ss.LogTrnBlk()
	ss.TimeLogBlock += 1
	ss.TimeLogBlockAll += 1
}

// CmdArgs processes command-line parameters.
func (ss *Sim) CmdArgs() (verbose, threads bool) {
	var nogui bool
	var saveEpcLog bool
	var saveRunLog bool
	var note string
	flag.StringVar(&ss.ParamSet, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.StringVar(&note, "note", "", "user note -- describe the run params etc")
	flag.IntVar(&ss.MaxConditions, "runs", 10, "maximum number of conditions to run")
	flag.BoolVar(&ss.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.SaveWts, "wts", false, "if true, save final weights after each run")
	flag.BoolVar(&saveEpcLog, "blklog", true, "if true, save train block log to file")
	flag.BoolVar(&saveRunLog, "runlog", true, "if true, save run log to file")
	flag.BoolVar(&nogui, "nogui", false, "if not passing any other args and want to run nogui, use nogui")
	flag.BoolVar(&verbose, "verbose", false, "give more feedback during initialization")
	flag.BoolVar(&threads, "threads", false, "use per-layer threads")
	flag.BoolVar(&ss.devMenuSetup, "wide-step-menus", false, "use wide (development) stepping menu setup")
	flag.Parse()

	if !nogui {
		return verbose, threads
	}

	ss.NoGui = nogui
	ss.InitSim()

	if note != "" {
		fmt.Printf("note: %s\n", note)
	}
	if saveEpcLog {
		var err error
		fnm := ss.LogFileName("epc")
		ss.TrnEpcFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TrnEpcFile = nil
		} else {
			fmt.Printf("Saving block log to: %v\n", fnm)
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
	fmt.Printf("Running %d Conditions\n", ss.MaxConditions)
	ss.ExecuteRun()
	return verbose, threads
}

// GetTrialBlockParams looks up a TrialBlockRecs by name. The second return value is true if found.
func (ss *Sim) GetTrialBlockParams(nm string) (*data.TrialBlockRecs, bool) {
	groups, ok := ss.MasterTrialBlockParams[nm]
	ret := data.NewTrialBlockRecs(&groups)
	return ret, ok
}

// GetBlockTrial returns the nth TrialBlockParams record in the currently set TrialBlockParams in the environment.
func (ev *PVLVEnv) GetBlockTrial(n int) *data.TrialBlockParams {
	ret := ev.TrialBlockParams.Records.Get(n).(*data.TrialBlockParams)
	return ret
}

// GetConditionParams returns a pointer to a ConditionParams, and indicates an error if not found.
func (ss *Sim) GetConditionParams(nm string) (*data.ConditionParams, bool) {
	ret, found := ss.MasterConditionParams[nm]
	return &ret, found
}

// GetRunParams returns a pointer to a RunParams, and indicates an error if not found.
func (ss *Sim) GetRunParams(nm string) (*data.RunParams, bool) {
	ret, found := ss.MasterRunParams[nm]
	return &ret, found
}

// ValidateRunParams goes through all defined RunParams and makes sure all names are valid, calling down all the
// way to the TrialBlock level.
func (ss *Sim) ValidateRunParams() {
	allSeqs := data.AllRunParams()
	allRunBlocks := data.AllConditionParams()
	allBlocks := data.AllTrialBlocks()
runsLoop:
	for seqNm, pSeq := range allSeqs {
		if seqNm != pSeq.Nm {
			fmt.Printf("ERROR: Name field \"%s\" does not match key for RunParams \"%s\"\n",
				pSeq.Nm, seqNm)
		}
		blockNms := []string{pSeq.Cond1Nm, pSeq.Cond2Nm, pSeq.Cond3Nm, pSeq.Cond4Nm, pSeq.Cond5Nm}
		for i, blockNm := range blockNms {
			if blockNm == "NullStep" || blockNm == "" {
				continue runsLoop
			}
			pRun, found := allRunBlocks[blockNm]
			if !found {
				fmt.Printf("ERROR: Invalid block name \"%s\" in ConditionParams \"%s\" step %d\n",
					blockNm, seqNm, i+1)
			} else {
				ss.ValidateBlockParams(blockNm, &pRun, allBlocks)
			}
		}
	}
}

// ValidateBlockParams goes through all defined ConditionParams and makes sure all names are valid, calling down all the
// way to the TrialBlock level.
func (ss *Sim) ValidateBlockParams(nm string, pCondition *data.ConditionParams, allBlocks data.TrialBlockMap) {
	if nm != pCondition.Nm {
		fmt.Printf("ERROR: Name field \"%s\" does not match key for ConditionParams \"%s\"\n",
			pCondition.Nm, nm)
	}
	blockNm := pCondition.TrialBlkNm
	_, found := allBlocks[blockNm]
	if !found {
		fmt.Printf("ERROR: Invalid block name \"%s\" in ConditionParams \"%s\"\n",
			blockNm, nm)
	}
}
