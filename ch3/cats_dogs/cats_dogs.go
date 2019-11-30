// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
cats_dogs: This project explores a simple **semantic network** intended to represent a (very small) set of relationships among different features used to represent a set of entities in the world.  In our case, we represent some features of cats and dogs: their color, size, favorite food, and favorite toy.
*/
package main

import (
	"bytes"
	"fmt"
	"log"
	"strconv"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
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
			{Sel: "Prjn", Desc: "no learning",
				Params: params.Params{
					"Prjn.Learn.Learn": "false",
				}},
			{Sel: "Layer", Desc: "generic params for all layers: lower gain, slower, soft clamp",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init":  "0.25",
					"Layer.Inhib.ActAvg.Fixed": "true",
					"Layer.Inhib.Layer.FBTau":  "3", // this is key for smoothing bumps
					"Layer.Act.Clamp.Hard":     "false",
					"Layer.Act.Clamp.Gain":     "1",
					"Layer.Act.XX1.Gain":       "40",  // more graded -- key
					"Layer.Act.Dt.VmTau":       "4",   // a bit slower -- not as effective as FBTau
					"Layer.Act.Gbar.L":         "0.1", // needs lower leak
				}},
			{Sel: ".Id", Desc: "specific inhibition for identity, name",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "4.0",
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
	Net        *leabra.Network   `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Pats       *etable.Table     `view:"no-inline" desc:"click to see and edit the testing input patterns to use"`
	TstCycLog  *etable.Table     `view:"no-inline" desc:"testing trial-level log data -- click to see record of network's response to each input"`
	Params     params.Sets       `view:"no-inline" desc:"full collection of param sets -- not really interesting for this model"`
	TestEnv    env.FixedTable    `desc:"Testing environment -- manages iterating over testing"`
	Time       leabra.Time       `desc:"leabra timing parameters and state"`
	ViewUpdt   leabra.TimeScales `desc:"at what time scale to update the display during testing?  Change to AlphaCyc to make display updating go faster"`
	TstRecLays []string          `desc:"names of layers to record activations etc of during testing"`

	// internal state - view:"-"
	Win        *gi.Window                  `view:"-" desc:"main GUI window"`
	NetView    *netview.NetView            `view:"-" desc:"the network viewer"`
	ToolBar    *gi.ToolBar                 `view:"-" desc:"the master toolbar"`
	TstCycPlot *eplot.Plot2D               `view:"-" desc:"the test-trial plot"`
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
	ss.TstCycLog = &etable.Table{}
	ss.Params = ParamSets
	ss.ViewUpdt = leabra.Cycle
	ss.TstRecLays = []string{"Name", "Identity", "Color", "FavoriteFood", "Size", "Species", "FavoriteToy"}
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
	ss.ConfigTstCycLog(ss.TstCycLog)
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
	net.InitName(net, "CatsAndDogs")
	name := net.AddLayer2D("Name", 1, 10, emer.Input)
	iden := net.AddLayer2D("Identity", 1, 10, emer.Input)
	color := net.AddLayer2D("Color", 1, 4, emer.Input)
	food := net.AddLayer2D("FavoriteFood", 1, 4, emer.Input)
	size := net.AddLayer2D("Size", 1, 3, emer.Input)
	spec := net.AddLayer2D("Species", 1, 2, emer.Input)
	toy := net.AddLayer2D("FavoriteToy", 1, 4, emer.Input)

	name.SetClass("Id") // share params
	iden.SetClass("Id")

	net.ConnectLayers(name, iden, prjn.NewOneToOne(), emer.Forward)
	net.ConnectLayers(iden, name, prjn.NewOneToOne(), emer.Back)

	net.ConnectLayers(color, iden, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(iden, color, prjn.NewFull(), emer.Back)

	net.ConnectLayers(food, iden, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(iden, food, prjn.NewFull(), emer.Back)

	net.ConnectLayers(size, iden, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(iden, size, prjn.NewFull(), emer.Back)

	net.ConnectLayers(spec, iden, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(iden, spec, prjn.NewFull(), emer.Back)

	net.ConnectLayers(toy, iden, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(iden, toy, prjn.NewFull(), emer.Back)

	iden.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Name", YAlign: relpos.Front, XAlign: relpos.Left, YOffset: 1})
	color.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Identity", YAlign: relpos.Front, XAlign: relpos.Left, YOffset: 1})
	food.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Identity", YAlign: relpos.Front, XAlign: relpos.Right, YOffset: 1})
	size.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Color", YAlign: relpos.Front, XAlign: relpos.Left})
	spec.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Color", YAlign: relpos.Front, XAlign: relpos.Right, XOffset: 2})
	toy.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "FavoriteFood", YAlign: relpos.Front, XAlign: relpos.Right, XOffset: 1})

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
	ab, err := Asset("cats_dogs.wts") // embedded in executable
	if err != nil {
		log.Println(err)
	}
	net.ReadWtsJSON(bytes.NewBuffer(ab))
	// net.OpenWtsJSON("cats_dogs.wts")
	// below is one-time conversion from c++ weights
	// net.OpenWtsCpp("CatsDogsNet.wts")
	// net.SaveWtsJSON("cats_dogs.wts")
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
	// ss.Time.CycPerQtr = 25 // use full 100 cycles, default
	ss.InitWts(ss.Net)
	ss.StopNow = false
	ss.SetParams("", false) // all sheets
	ss.UpdateView()
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters() string {
	return fmt.Sprintf("Trial:\t%d\tCycle:\t%d\tName:\t%v\t\t\t", ss.TestEnv.Trial.Cur, ss.Time.Cycle, ss.TestEnv.TrialName)
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

	ss.Net.AlphaCycInit()
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
			ss.Time.CycleInc()
			switch viewUpdt {
			case leabra.Cycle:
				ss.UpdateView()
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

	lays := []string{"Name", "Identity", "Color", "FavoriteFood", "Size", "Species", "FavoriteToy"}
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
	// patgen.ReshapeCppFile(ss.Pats, "StdInputData.dat", "cats_dogs_pats.tsv") // one-time reshape
	ss.OpenPatAsset(ss.Pats, "cats_dogs_pats.tsv", "CatAndDogPats", "Testing patterns")
	// dt.OpenCSV("cats_dogs_pats.tsv", etable.Tab)
}

//////////////////////////////////////////////
//  TstCycLog

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

// Harmony computes the harmony (excitatory net input Ge * Act)
func (ss *Sim) Harmony(nt *leabra.Network) float32 {
	harm := float32(0)
	nu := 0
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		lly := ly.(leabra.LeabraLayer).AsLeabra()
		for i := range lly.Neurons {
			nrn := &(lly.Neurons[i])
			harm += nrn.Ge * nrn.Act
			nu++
		}
	}
	if nu > 0 {
		harm /= float32(nu)
	}
	return harm
}

// LogTstCyc adds data from current cycle to the TstCycLog table.
// log always contains number of testing items
func (ss *Sim) LogTstCyc(dt *etable.Table, cyc int) {
	if dt.Rows <= cyc {
		dt.SetNumRows(cyc + 1)
	}
	row := cyc

	harm := ss.Harmony(ss.Net)
	dt.SetCellFloat("Cycle", row, float64(cyc))
	dt.SetCellString("TrialName", row, ss.TestEnv.TrialName)
	dt.SetCellFloat("Harmony", row, float64(harm))

	for _, lnm := range ss.TstRecLays {
		tsr := ss.ValsTsr(lnm)
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		ly.UnitValsTensor(tsr, "Act")
		dt.SetCellTensor(lnm, row, tsr)
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TstCycPlot.GoUpdate()
}

func (ss *Sim) ConfigTstCycLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstCycLog")
	dt.SetMetaData("desc", "Record of testing per cycle")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := 100 // max cycles
	sch := etable.Schema{
		{"Cycle", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"Harmony", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.TstRecLays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		sch = append(sch, etable.Column{lnm, etensor.FLOAT64, ly.Shp.Shp, nil})
	}
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTstCycPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "CatsAndDogs Test Cycle Plot"
	plt.Params.XAxisCol = "Cycle"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Cycle", false, true, 0, false, 0)
	plt.SetColParams("TrialName", false, true, 0, false, 0)
	plt.SetColParams("Harmony", true, true, 0, true, .25)

	for _, lnm := range ss.TstRecLays {
		plt.SetColParams(lnm, false, true, 0, true, 1)
	}
	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.ViewDefaults()
	nv.Scene().Camera.Pose.Pos.Set(0, 1.5, 3.0) // more "head on" than default which is more "top down"
	nv.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})

	labs := []string{" Morr Socks Sylv Garf Fuzz Rex Fido Spot Snoop Butch",
		" black white brown orange", "bugs grass scraps shoe", "small  med  large", "cat     dog", "string feath bone shoe"}
	nv.ConfigLabels(labs)

	lays := []string{"Name", "Color", "FavoriteFood", "Size", "Species", "FavoriteToy"}

	for li, lnm := range lays {
		ly := nv.LayerByName(lnm)
		lbl := nv.LabelByName(labs[li])
		lbl.Pose = ly.Pose
		lbl.Pose.Pos.Y += .2
		lbl.Pose.Pos.Z += .02
		lbl.Pose.Scale.SetMul(mat32.Vec3{0.4, 0.08, 0.5})
	}
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("cat_dogs")
	gi.SetAppAbout(`cats_dogs: This project explores a simple **semantic network** intended to represent a (very small) set of relationships among different features used to represent a set of entities in the world.  In our case, we represent some features of cats and dogs: their color, size, favorite food, and favorite toy.
  See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch3/cats_dogs/README.md">README.md on GitHub</a>.</p>`)

	win := gi.NewWindow2D("cats_dogs", "Cats and Dogs", width, height, true)
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
	ss.ConfigNetView(nv) // add labels etc

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TstCycPlot").(*eplot.Plot2D)
	ss.TstCycPlot = ss.ConfigTstCycPlot(plt, ss.TstCycLog)

	split.SetSplits(.3, .7)

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

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch3/cats_dogs/README.md")
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
					"ext": ".wts",
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
