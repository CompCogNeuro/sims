// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
face_categ: This project explores how sensory inputs (in this case simple cartoon faces) can be categorized in multiple different ways, to extract the relevant information and collapse across the irrelevant. It allows you to explore both bottom-up processing from face image to categories, and top-down processing from category values to face images (imagery), including the ability to dynamically iterate both bottom-up and top-down to cleanup partial inputs (partially occluded face images).
*/
package main

import (
	"bytes"
	"fmt"
	"log"
	"math/rand"
	"strconv"
	"strings"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/etable/clust"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/metric"
	"github.com/emer/etable/simat"
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

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "no learning",
				Params: params.Params{
					"Prjn.Learn.Learn": "false",
				}},
			{Sel: "Layer", Desc: "fix expected activity levels, reduce leak",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init":  "0.15",
					"Layer.Inhib.ActAvg.Fixed": "true",
					"Layer.Act.Gbar.L":         "0.1", // needs lower leak
				}},
			{Sel: "#Input", Desc: "specific inhibition",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "2.0",
					"Layer.Act.Clamp.Hard": "false",
					"Layer.Act.Clamp.Gain": "0.2",
				}},
			{Sel: "#Identity", Desc: "specific inhbition",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "3.6",
				}},
			{Sel: "#Gender", Desc: "specific inhbition",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.6",
				}},
			{Sel: "#Emotion", Desc: "specific inhbition",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.3",
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
	Net           *leabra.Network   `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Pats          *etable.Table     `view:"no-inline" desc:"click to see the full face testing input patterns to use"`
	PartPats      *etable.Table     `view:"no-inline" desc:"click to see the partial face testing input patterns to use"`
	TstTrlLog     *etable.Table     `view:"no-inline" desc:"testing trial-level log data -- click to see record of network's response to each input"`
	PrjnTable     *etable.Table     `view:"no-inline" desc:"projection of testing data"`
	Params        params.Sets       `view:"no-inline" desc:"full collection of param sets -- not really interesting for this model"`
	ParamSet      string            `view:"-" desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set -- can use multiple names separated by spaces (don't put spaces in ParamSet names!)"`
	TestEnv       env.FixedTable    `desc:"Testing environment -- manages iterating over testing"`
	Time          leabra.Time       `desc:"leabra timing parameters and state"`
	ViewUpdt      leabra.TimeScales `desc:"at what time scale to update the display during testing?  Change to AlphaCyc to make display updating go faster"`
	TstRecLays    []string          `desc:"names of layers to record activations etc of during testing"`
	ClustFaces    *eplot.Plot2D     `view:"no-inline" desc:"cluster plot of faces"`
	ClustEmote    *eplot.Plot2D     `view:"no-inline" desc:"cluster plot of emotions"`
	ClustGend     *eplot.Plot2D     `view:"no-inline" desc:"cluster plot of genders"`
	ClustIdent    *eplot.Plot2D     `view:"no-inline" desc:"cluster plot of identity"`
	PrjnRandom    *eplot.Plot2D     `view:"no-inline" desc:"random projection plot"`
	PrjnEmoteGend *eplot.Plot2D     `view:"no-inline" desc:"projection plot of emotions & gender"`

	// internal state - view:"-"
	Win        *gi.Window                  `view:"-" desc:"main GUI window"`
	NetView    *netview.NetView            `view:"-" desc:"the network viewer"`
	ToolBar    *gi.ToolBar                 `view:"-" desc:"the master toolbar"`
	TstTrlPlot *eplot.Plot2D               `view:"-" desc:"the test-trial plot"`
	ValsTsrs   map[string]*etensor.Float32 `view:"-" desc:"for holding layer values"`
	IsRunning  bool                        `view:"-" desc:"true if sim is running"`
	StopNow    bool                        `view:"-" desc:"flag to stop running"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &leabra.Network{}
	ss.Pats = &etable.Table{}
	ss.PartPats = &etable.Table{}
	ss.TstTrlLog = &etable.Table{}
	ss.PrjnTable = &etable.Table{}
	ss.Params = ParamSets
	ss.ViewUpdt = leabra.Cycle
	ss.TstRecLays = []string{"Input", "Emotion", "Gender", "Identity"}
	ss.ClustFaces = &eplot.Plot2D{}
	ss.ClustEmote = &eplot.Plot2D{}
	ss.ClustGend = &eplot.Plot2D{}
	ss.ClustIdent = &eplot.Plot2D{}
	ss.PrjnRandom = &eplot.Plot2D{}
	ss.PrjnEmoteGend = &eplot.Plot2D{}
	ss.Defaults()
}

// Defaults sets default params
func (ss *Sim) Defaults() {
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.OpenPats()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
}

func (ss *Sim) ConfigEnv() {
	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Dsc = "testing params and state"
	ss.TestEnv.Table = etable.NewIdxView(ss.Pats)
	ss.TestEnv.Sequential = true
	ss.TestEnv.Validate()
	ss.TestEnv.Init(0)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.InitName(net, "FaceCateg")
	inp := net.AddLayer2D("Input", 16, 16, emer.Input)
	emo := net.AddLayer2D("Emotion", 1, 2, emer.Compare)
	gend := net.AddLayer2D("Gender", 1, 2, emer.Compare)
	iden := net.AddLayer2D("Identity", 1, 10, emer.Compare)

	full := prjn.NewFull()

	net.BidirConnectLayers(inp, emo, full)
	net.BidirConnectLayers(inp, gend, full)
	net.BidirConnectLayers(inp, iden, full)

	emo.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Input", YAlign: relpos.Front, XAlign: relpos.Left, Space: 2})
	gend.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Input", YAlign: relpos.Front, XAlign: relpos.Right, Space: 2})
	iden.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Input", YAlign: relpos.Center, XAlign: relpos.Left, Space: 2})

	net.Defaults()
	ss.SetParams("Network", false) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	ss.InitWts(net)
}

// InitWts loads the saved weights
func (ss *Sim) InitWts(net *leabra.Network) {
	net.InitWts()
	ab, err := Asset("faces.wts") // embedded in executable
	if err != nil {
		log.Println(err)
	}
	net.ReadWtsJSON(bytes.NewBuffer(ab))
	// net.OpenWtsJSON("faces.wts")
	// below is one-time conversion from c++ weights
	// net.OpenWtsCpp("FaceNetworkCpp.wts")
	// net.SaveWtsJSON("faces.wts")
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	// ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.TestEnv.Init(0)
	ss.Time.Reset()
	ss.Time.CycPerQtr = 10 // don't need much time
	ss.InitWts(ss.Net)
	ss.StopNow = false
	ss.SetParams("", false) // all sheets
	ss.UpdateView()
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters() string {
	return fmt.Sprintf("Trial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t", ss.TestEnv.Trial.Cur, ss.Time.Cycle, ss.TestEnv.TrialName.Cur)
}

func (ss *Sim) UpdateView() {
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters())
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate() // note: using counters is significantly slower..
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// AlphaCyc runs one alpha-cycle (100 msec, 4 quarters)			 of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// Handles netview updating within scope of AlphaCycle
func (ss *Sim) AlphaCyc() {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdt := ss.ViewUpdt

	// note: this has no learning calls

	ss.Net.AlphaCycInit(false) // depends on this call
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			ss.Time.CycleInc()
			switch viewUpdt {
			case leabra.Cycle:
				if cyc != ss.Time.CycPerQtr-1 { // will be updated by quarter
					ss.UpdateView()
				}
			case leabra.FastSpike:
				if (cyc+1)%10 == 0 {
					ss.UpdateView()
				}
			}
		}
		ss.Net.QuarterFinal(&ss.Time)
		ss.Time.QuarterInc()
		switch {
		case viewUpdt <= leabra.Quarter:
			ss.UpdateView()
		case viewUpdt == leabra.Phase:
			if qtr >= 2 {
				ss.UpdateView()
			}
		}
	}

	if viewUpdt == leabra.AlphaCycle {
		ss.UpdateView()
	}
}

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"Input", "Emotion", "Gender", "Identity"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := en.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
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

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial() {
	ss.TestEnv.Step()

	// Query counters FIRST
	_, _, chg := ss.TestEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewUpdt > leabra.AlphaCycle {
			ss.UpdateView()
		}
		return
	}

	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc()
	ss.LogTstTrl(ss.TstTrlLog)
}

// TestItem tests given item which is at given index in test item list
func (ss *Sim) TestItem(idx int) {
	cur := ss.TestEnv.Trial.Cur
	ss.TestEnv.Trial.Cur = idx
	ss.TestEnv.SetTrialName()
	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc() // !train
	ss.TestEnv.Trial.Cur = cur
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.TestEnv.Init(0)
	for {
		ss.TestTrial()
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

// SetInput sets whether the input to the network comes in bottom-up
// (Input layer) or top-down (Higher-level category layers)
func (ss *Sim) SetInput(topDown bool) {
	inp := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	emo := ss.Net.LayerByName("Emotion").(leabra.LeabraLayer).AsLeabra()
	gend := ss.Net.LayerByName("Gender").(leabra.LeabraLayer).AsLeabra()
	iden := ss.Net.LayerByName("Identity").(leabra.LeabraLayer).AsLeabra()
	if topDown {
		inp.SetType(emer.Compare)
		emo.SetType(emer.Input)
		gend.SetType(emer.Input)
		iden.SetType(emer.Input)
	} else {
		inp.SetType(emer.Input)
		emo.SetType(emer.Compare)
		gend.SetType(emer.Compare)
		iden.SetType(emer.Compare)
	}
}

// SetPats selects which patterns to present: full or partial faces
func (ss *Sim) SetPats(partial bool) {
	if partial {
		ss.TestEnv.Table = etable.NewIdxView(ss.PartPats)
		ss.TestEnv.Validate()
		ss.TestEnv.Init(0)
	} else {
		ss.TestEnv.Table = etable.NewIdxView(ss.Pats)
		ss.TestEnv.Validate()
		ss.TestEnv.Init(0)
	}
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
	// patgen.ReshapeCppFile(ss.Pats, "faces.dat", "faces.tsv")     // one-time reshape
	// patgen.ReshapeCppFile(ss.PartPats, "partial_faces.tsv", "partial_faces.tsv") // one-time reshape
	ss.OpenPatAsset(ss.Pats, "faces.tsv", "FacePats", "Testing Face patterns: full faces")
	// err := ss.Pats.OpenCSV("faces.tsv", etable.Tab)
	ss.OpenPatAsset(ss.PartPats, "partial_faces.tsv", "PartFacePats", "Testing Face patterns: partial faces")
	// err := ss.PartPats.OpenCSV("partial_faces.tsv", etable.Tab)
}

//////////////////////////////////////////////
//  Cluster Plots

// ClusterPlots computes all the cluster plots from the faces input data
func (ss *Sim) ClusterPlots() {
	ss.ClustPlot(ss.ClustFaces, ss.Pats, "Input")
	ss.ClustPlot(ss.ClustEmote, ss.Pats, "Emotion")
	ss.ClustPlot(ss.ClustGend, ss.Pats, "Gender")
	ss.ClustPlot(ss.ClustIdent, ss.Pats, "Identity")

	ss.PrjnPlot()
}

// ClustPlot does one cluster plot on given table column
func (ss *Sim) ClustPlot(plt *eplot.Plot2D, dt *etable.Table, colNm string) {
	ix := etable.NewIdxView(dt)
	smat := &simat.SimMat{}
	smat.TableColStd(ix, colNm, "Name", false, metric.Euclidean)
	pt := &etable.Table{}
	clust.Plot(pt, clust.Glom(smat, clust.MinDist), smat)
	plt.InitName(plt, colNm)
	plt.Params.Title = "Cluster Plot of Faces " + colNm
	plt.Params.XAxisCol = "X"
	plt.SetTable(pt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("X", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Y", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Label", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
}

func (ss *Sim) PrjnPlot() {
	ss.TestAll()

	rvec0 := ss.ValsTsr("rvec0")
	rvec1 := ss.ValsTsr("rvec1")
	rvec0.SetShape([]int{256}, nil, nil)
	rvec1.SetShape([]int{256}, nil, nil)
	for i := range rvec1.Values {
		rvec0.Values[i] = .15 * (2*rand.Float32() - 1)
		rvec1.Values[i] = .15 * (2*rand.Float32() - 1)
	}

	tst := ss.TstTrlLog
	nr := tst.Rows
	dt := ss.PrjnTable
	ss.ConfigPrjnTable(dt)

	for r := 0; r < nr; r++ {
		// single emotion dimension from sad to happy
		emote := 0.5*tst.CellTensorFloat1D("Emotion", r, 0) + -0.5*tst.CellTensorFloat1D("Emotion", r, 1)
		emote += .1 * (2*rand.Float64() - 1) // some jitter so labels are readable
		// single geneder dimension from male to femail
		gend := 0.5*tst.CellTensorFloat1D("Gender", r, 0) + -0.5*tst.CellTensorFloat1D("Gender", r, 1)
		gend += .1 * (2*rand.Float64() - 1) // some jitter so labels are readable
		input := tst.CellTensor("Input", r).(*etensor.Float32)
		rprjn0 := metric.InnerProduct32(rvec0.Values, input.Values)
		rprjn1 := metric.InnerProduct32(rvec1.Values, input.Values)
		dt.SetCellFloat("Trial", r, tst.CellFloat("Trial", r))
		dt.SetCellString("TrialName", r, tst.CellString("TrialName", r))
		dt.SetCellFloat("GendPrjn", r, gend)
		dt.SetCellFloat("EmotePrjn", r, emote)
		dt.SetCellFloat("RndPrjn0", r, float64(rprjn0))
		dt.SetCellFloat("RndPrjn1", r, float64(rprjn1))
	}

	plt := ss.PrjnRandom
	plt.InitName(plt, "PrjnRandom")
	plt.Params.Title = "Face Random Prjn Plot"
	plt.Params.XAxisCol = "RndPrjn0"
	plt.SetTable(dt)
	plt.Params.Lines = false
	plt.Params.Points = true
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("TrialName", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("RndPrjn0", eplot.Off, eplot.FixMin, -1, eplot.FixMax, 1)
	plt.SetColParams("RndPrjn1", eplot.On, eplot.FixMin, -1, eplot.FixMax, 1)

	plt = ss.PrjnEmoteGend
	plt.InitName(plt, "PrjnEmoteGend")
	plt.Params.Title = "Face Emotion / Gender Prjn Plot"
	plt.Params.XAxisCol = "GendPrjn"
	plt.SetTable(dt)
	plt.Params.Lines = false
	plt.Params.Points = true
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("TrialName", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("GendPrjn", eplot.Off, eplot.FixMin, -1, eplot.FixMax, 1)
	plt.SetColParams("EmotePrjn", eplot.On, eplot.FixMin, -1, eplot.FixMax, 1)
}

func (ss *Sim) ConfigPrjnTable(dt *etable.Table) {
	dt.SetMetaData("name", "PrjnTable")
	dt.SetMetaData("desc", "projection of data onto dimension")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TestEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
	}
	sch = append(sch, etable.Column{"GendPrjn", etensor.FLOAT64, nil, nil})
	sch = append(sch, etable.Column{"EmotePrjn", etensor.FLOAT64, nil, nil})
	sch = append(sch, etable.Column{"RndPrjn0", etensor.FLOAT64, nil, nil})
	sch = append(sch, etable.Column{"RndPrjn1", etensor.FLOAT64, nil, nil})
	dt.SetFromSchema(sch, nt)
}

//////////////////////////////////////////////
//  TstTrlLog

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

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *etable.Table) {
	trl := ss.TestEnv.Trial.Cur
	row := trl

	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("TrialName", row, ss.TestEnv.TrialName.Cur)

	for _, lnm := range ss.TstRecLays {
		tsr := ss.ValsTsr(lnm)
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		ly.UnitValsTensor(tsr, "Act")
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
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
	}
	for _, lnm := range ss.TstRecLays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		sch = append(sch, etable.Column{lnm, etensor.FLOAT32, ly.Shp.Shp, nil})
	}
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "FaceCateg Test Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	plt.SetStretchMax()
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)

	for _, lnm := range ss.TstRecLays {
		cp := plt.SetColParams(lnm, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
		cp.TensorIdx = -1 // plot all
	}
	plt.SetColParams("Gender", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // display
	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	labs := []string{"happy sad", "female  male", "Albt Bett Lisa Mrk Wnd Zane"}
	nv.ConfigLabels(labs)
	emot := nv.LayerByName("Emotion")
	hs := nv.LabelByName(labs[0])
	hs.Pose = emot.Pose
	hs.Pose.Pos.Y += .1
	hs.Pose.Scale.SetMulScalar(0.5)

	gend := nv.LayerByName("Gender")
	fm := nv.LabelByName(labs[1])
	fm.Pose = gend.Pose
	fm.Pose.Pos.X -= .05
	fm.Pose.Pos.Y += .1
	fm.Pose.Scale.SetMulScalar(0.5)

	id := nv.LayerByName("Identity")
	nms := nv.LabelByName(labs[2])
	nms.Pose = id.Pose
	nms.Pose.Pos.Y += .1
	nms.Pose.Scale.SetMulScalar(0.5)
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("face_categ")
	gi.SetAppAbout(`face_categ: This project explores how sensory inputs (in this case simple cartoon faces) can be categorized in multiple different ways, to extract the relevant information and collapse across the irrelevant. It allows you to explore both bottom-up processing from face image to categories, and top-down processing from category values to face images (imagery), including the ability to dynamically iterate both bottom-up and top-down to cleanup partial inputs (partially occluded face images).  See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch3/face_categ/README.md">README.md on GitHub</a>.</p>`)

	win := gi.NewMainWindow("face_categ", "Face Categorization", width, height)
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
	// nv.Params.ColorMap = "Jet" // default is ColdHot
	// which fares pretty well in terms of discussion here:
	// https://matplotlib.org/tutorials/colors/colormaps.html
	nv.SetNet(ss.Net)
	ss.NetView = nv

	nv.ViewDefaults()
	ss.ConfigNetView(nv) // add labels etc

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TstTrlPlot").(*eplot.Plot2D)
	ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

	split.SetSplits(.2, .8)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Stop()
	})

	tbar.AddAction(gi.ActOpts{Label: "Test Trial", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TestTrial()
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
							fmt.Printf("testing index: %d\n", idxs[0])
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

	tbar.AddAction(gi.ActOpts{Label: "SetInput", Icon: "gear", Tooltip: "set whether the input to the network comes in bottom-up (Input layer) or top-down (Higher-level category layers)"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			giv.CallMethod(ss, "SetInput", vp)
		})
	tbar.AddAction(gi.ActOpts{Label: "SetPats", Icon: "gear", Tooltip: "set which set of patterns to present -- full or partial faces"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			giv.CallMethod(ss, "SetPats", vp)
		})
	tbar.AddAction(gi.ActOpts{Label: "Cluster Plots", Icon: "image", Tooltip: "generate cluster plots of the different layer patterns"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.ClusterPlots()
			vp.SetNeedsFullRender()
		})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch3/face_categ/README.md")
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
					"ext": ".wts",
				}},
			},
		}},
		{"SetInput", ki.Props{
			"desc": "set whether the input to the network comes in bottom-up (Input layer) or top-down (Higher-level category layers)",
			"icon": "gear",
			"Args": ki.PropSlice{
				{"Top Down", ki.Props{}},
			},
		}},
		{"SetPats", ki.Props{
			"desc": "set which set of patterns to present -- full or partial faces",
			"icon": "gear",
			"Args": ki.PropSlice{
				{"Partial", ki.Props{}},
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
