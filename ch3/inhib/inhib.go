// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
inhib: This simulation explores how inhibitory interneurons can dynamically
control overall activity levels within the network, by providing both
feedforward and feedback inhibition to excitatory pyramidal neurons.
*/
package main

import (
	"fmt"
	"log"
	"strconv"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/patgen"
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
					"Prjn.WtInit.Dist": "Uniform",
					"Prjn.WtInit.Mean": "0.25",
					"Prjn.WtInit.Var":  "0.2",
				}},
			{Sel: "Layer", Desc: "generic params for all layers: lower gain, slower, soft clamp",
				Params: params.Params{
					"Layer.Inhib.Layer.On":     "false",
					"Layer.Inhib.ActAvg.Init":  "0.2",
					"Layer.Inhib.ActAvg.Fixed": "true",
					"Layer.Act.Dt.GTau":        "40",
					"Layer.Act.Gbar.I":         "0.4",
					"Layer.Act.Gbar.L":         "0.1",
				}},
			{Sel: ".InhibLay", Desc: "generic params for all layers: lower gain, slower, soft clamp",
				Params: params.Params{
					"Layer.Act.XX1.Thr": "0.4", // essential for getting active early
				}},
			{Sel: ".Inhib", Desc: "inhibitory projections",
				Params: params.Params{
					"Prjn.WtInit.Dist": "Uniform",
					"Prjn.WtInit.Mean": "0.5",
					"Prjn.WtInit.Var":  "0",
					"Prjn.WtInit.Sym":  "false",
				}},
		},
	}},
	{Name: "Untrained", Desc: "simulates untrained weights -- lower variance", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: ".Excite", Desc: "excitatory connections",
				Params: params.Params{
					"Prjn.WtInit.Dist": "Uniform",
					"Prjn.WtInit.Mean": "0.25",
					"Prjn.WtInit.Var":  "0.2",
				}},
		},
	}},
	{Name: "Trained", Desc: "simulates trained weights -- higher variance", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: ".Excite", Desc: "excitatory connections",
				Params: params.Params{
					"Prjn.WtInit.Dist": "Gaussian",
					"Prjn.WtInit.Mean": "0.25",
					"Prjn.WtInit.Var":  "0.7",
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
	BidirNet   bool    `desc:"if true, use the bidirectionally-connected network -- otherwise use the simpler feedforward network"`
	TrainedWts bool    `desc:"simulate trained weights by having higher variance and Gaussian distributed weight values -- otherwise lower variance, uniform"`
	InputPct   float32 `def:"20" min:"5" max:"50" step:"1" desc:"percent of active units in input layer (literally number of active units, because input has 100 units total)"`
	FFFBInhib  bool    `def:"false" desc:"use feedforward, feedback (FFFB) computed inhibition instead of unit-level inhibition"`

	HiddenGbarI       float32 `def:"0.4" min:"0" step:"0.05" desc:"inhibitory conductance strength for inhibition into Hidden layer"`
	InhibGbarI        float32 `def:"0.75" min:"0" step:"0.05" desc:"inhibitory conductance strength for inhibition into Inhib layer (self-inhibition -- tricky!)"`
	FFinhibWtScale    float32 `def:"1" min:"0" step:"0.1" desc:"feedforward (FF) inhibition relative strength: for FF projections into Inhib neurons"`
	FBinhibWtScale    float32 `def:"1" min:"0" step:"0.1" desc:"feedback (FB) inhibition relative strength: for projections into Inhib neurons"`
	HiddenGTau        float32 `def:"40" min:"1" step:"1" desc:"time constant (tau) for updating G conductances into Hidden neurons -- much slower than std default of 1.4"`
	InhibGTau         float32 `def:"20" min:"1" step:"1" desc:"time constant (tau) for updating G conductances into Inhib neurons -- much slower than std default of 1.4, but 2x faster than Hidden"`
	FmInhibWtScaleAbs float32 `def:"1" desc:"absolute weight scaling of projections from inhibition onto hidden and inhib layers -- this must be set to 0 to turn off the connection-based inhibition when using the FFFBInhib computed inbhition"`

	NetFF      *leabra.Network   `view:"no-inline" desc:"the feedforward network -- click to view / edit parameters for layers, prjns, etc"`
	NetBidir   *leabra.Network   `view:"no-inline" desc:"the bidirectional network -- click to view / edit parameters for layers, prjns, etc"`
	TstCycLog  *etable.Table     `view:"no-inline" desc:"testing trial-level log data -- click to see record of network's response to each input"`
	Params     params.Sets       `view:"no-inline" desc:"full collection of param sets -- not really interesting for this model"`
	Time       leabra.Time       `desc:"leabra timing parameters and state"`
	ViewUpdt   leabra.TimeScales `desc:"at what time scale to update the display during testing?  Change to AlphaCyc to make display updating go faster"`
	TstRecLays []string          `desc:"names of layers to record activations etc of during testing"`
	Pats       *etable.Table     `view:"no-inline" desc:"the input patterns to use -- randomly generated"`

	// internal state - view:"-"
	Win          *gi.Window                  `view:"-" desc:"main GUI window"`
	NetViewFF    *netview.NetView            `view:"-" desc:"the network viewer"`
	NetViewBidir *netview.NetView            `view:"-" desc:"the network viewer"`
	ToolBar      *gi.ToolBar                 `view:"-" desc:"the master toolbar"`
	TstCycPlot   *eplot.Plot2D               `view:"-" desc:"the test-trial plot"`
	ValsTsrs     map[string]*etensor.Float32 `view:"-" desc:"for holding layer values"`
	IsRunning    bool                        `view:"-" desc:"true if sim is running"`
	StopNow      bool                        `view:"-" desc:"flag to stop running"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.NetFF = &leabra.Network{}
	ss.NetBidir = &leabra.Network{}
	ss.TstCycLog = &etable.Table{}
	ss.Params = ParamSets
	ss.ViewUpdt = leabra.Cycle
	ss.TstRecLays = []string{"Hidden", "Inhib"}
	ss.Pats = &etable.Table{}
	ss.Defaults()
}

// Defaults sets default params
func (ss *Sim) Defaults() {
	ss.TrainedWts = false
	ss.InputPct = 20
	ss.FFFBInhib = false
	ss.HiddenGbarI = 0.4
	ss.InhibGbarI = 0.75
	ss.FFinhibWtScale = 1
	ss.FBinhibWtScale = 1
	ss.HiddenGTau = 40
	ss.InhibGTau = 20
	ss.FmInhibWtScaleAbs = 1
	ss.Time.CycPerQtr = 50
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigPats()
	ss.ConfigNetFF(ss.NetFF)
	ss.ConfigNetBidir(ss.NetBidir)
	ss.ConfigTstCycLog(ss.TstCycLog)
}

func (ss *Sim) ConfigNetFF(net *leabra.Network) {
	net.InitName(net, "InhibFF")
	inp := net.AddLayer2D("Input", 10, 10, emer.Input)
	hid := net.AddLayer2D("Hidden", 10, 10, emer.Hidden)
	inh := net.AddLayer2D("Inhib", 10, 2, emer.Hidden)
	inh.SetClass("InhibLay")

	pj := net.ConnectLayers(inp, hid, prjn.NewFull(), emer.Forward)
	pj.SetClass("Excite")
	net.ConnectLayers(inp, inh, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(hid, inh, prjn.NewFull(), emer.Back)
	net.ConnectLayers(inh, hid, prjn.NewFull(), emer.Inhib)
	net.ConnectLayers(inh, inh, prjn.NewFull(), emer.Inhib)

	inh.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Hidden", YAlign: relpos.Front, Space: 1})

	net.Defaults()
	ss.SetParams("Network", false) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	ss.InitWts(net)
}

func (ss *Sim) ConfigNetBidir(net *leabra.Network) {
	net.InitName(net, "InhibBidir")
	inp := net.AddLayer2D("Input", 10, 10, emer.Input)
	hid := net.AddLayer2D("Hidden", 10, 10, emer.Hidden)
	inh := net.AddLayer2D("Inhib", 10, 2, emer.Hidden)
	inh.SetClass("InhibLay")
	hid2 := net.AddLayer2D("Hidden2", 10, 10, emer.Hidden)
	inh2 := net.AddLayer2D("Inhib2", 10, 2, emer.Hidden)
	inh2.SetClass("InhibLay")

	pj := net.ConnectLayers(inp, hid, prjn.NewFull(), emer.Forward)
	pj.SetClass("Excite")
	net.ConnectLayers(inp, inh, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(hid2, inh, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(hid, inh, prjn.NewFull(), emer.Back)
	net.ConnectLayers(inh, hid, prjn.NewFull(), emer.Inhib)
	net.ConnectLayers(inh, inh, prjn.NewFull(), emer.Inhib)

	pj = net.ConnectLayers(hid, hid2, prjn.NewFull(), emer.Forward)
	pj.SetClass("Excite")
	pj = net.ConnectLayers(hid2, hid, prjn.NewFull(), emer.Back)
	pj.SetClass("Excite")
	net.ConnectLayers(hid, inh2, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(hid2, inh2, prjn.NewFull(), emer.Back)
	net.ConnectLayers(inh2, hid2, prjn.NewFull(), emer.Inhib)
	net.ConnectLayers(inh2, inh2, prjn.NewFull(), emer.Inhib)

	inh.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Hidden", YAlign: relpos.Front, Space: 1})
	hid2.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Hidden", YAlign: relpos.Front, XAlign: relpos.Middle})
	inh2.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Hidden2", YAlign: relpos.Front, Space: 1})

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
}

func (ss *Sim) ConfigPats() {
	dt := ss.Pats
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")
	dt.SetFromSchema(etable.Schema{
		{"Name", etensor.STRING, nil, nil},
		{"Input", etensor.FLOAT32, []int{10, 10}, []string{"Y", "X"}},
	}, 1)
	patgen.PermutedBinaryRows(dt.Cols[1], int(ss.InputPct), 1, 0)
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Time.Reset()
	ss.StopNow = false
	ss.SetParams("", false) // all sheets
	ss.InitWts(ss.NetFF)
	ss.InitWts(ss.NetBidir)
	ss.UpdateView()
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters() string {
	return fmt.Sprintf("Cycle:\t%d\t\t\t", ss.Time.Cycle)
}

func (ss *Sim) UpdateView() {
	var nv *netview.NetView
	if ss.BidirNet {
		nv = ss.NetViewBidir
	} else {
		nv = ss.NetViewFF
	}
	if nv != nil && nv.IsVisible() {
		nv.Record(ss.Counters())
		// note: essential to use Go version of update when called from another goroutine
		nv.GoUpdate() // note: using counters is significantly slower..
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// Net returns the current active network
func (ss *Sim) Net() *leabra.Network {
	if ss.BidirNet {
		return ss.NetBidir
	} else {
		return ss.NetFF
	}
}

// AlphaCyc runs one alpha-cycle (100 msec, 4 quarters)			 of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// Handles netview updating within scope of AlphaCycle
func (ss *Sim) AlphaCyc() {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdt := ss.ViewUpdt

	nt := ss.Net()

	// note: this has no learning calls

	nt.AlphaCycInit()
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			nt.Cycle(&ss.Time)
			ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
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
		nt.QuarterFinal(&ss.Time)
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
func (ss *Sim) ApplyInputs() {
	nt := ss.Net()
	nt.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	ly := nt.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	pat := ss.Pats.CellTensor("Input", 0)
	ly.ApplyExt(pat)
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

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial() {
	nt := ss.Net()
	nt.InitActs()
	ss.SetParams("", false) // all sheets
	ss.ApplyInputs()
	ss.AlphaCyc()
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
	nt := ss.Net()
	err := ss.SetParamsSet("Base", sheet, setMsg)
	if ss.TrainedWts {
		ss.SetParamsSet("Trained", sheet, setMsg)
	} else {
		ss.SetParamsSet("Untrained", sheet, setMsg)
	}
	ffinhsc := ss.FFinhibWtScale
	if nt == ss.NetBidir {
		ffinhsc *= 0.5 // 2 inhib prjns so .5 ea
	}
	hid := nt.LayerByName("Hidden").(leabra.LeabraLayer).AsLeabra()
	hid.Act.Gbar.I = ss.HiddenGbarI
	hid.Act.Dt.GTau = ss.HiddenGTau
	hid.Act.Update()
	inh := nt.LayerByName("Inhib").(leabra.LeabraLayer).AsLeabra()
	inh.Act.Gbar.I = ss.InhibGbarI
	inh.Act.Dt.GTau = ss.InhibGTau
	inh.Act.Update()
	ff := inh.RcvPrjns.SendName("Input").(leabra.LeabraPrjn).AsLeabra()
	ff.WtScale.Rel = ffinhsc
	fb := inh.RcvPrjns.SendName("Hidden").(leabra.LeabraPrjn).AsLeabra()
	fb.WtScale.Rel = ss.FBinhibWtScale
	hid.Inhib.Layer.On = ss.FFFBInhib
	inh.Inhib.Layer.On = ss.FFFBInhib
	fi := hid.RcvPrjns.SendName("Inhib").(leabra.LeabraPrjn).AsLeabra()
	fi.WtScale.Abs = ss.FmInhibWtScaleAbs
	fi = inh.RcvPrjns.SendName("Inhib").(leabra.LeabraPrjn).AsLeabra()
	fi.WtScale.Abs = ss.FmInhibWtScaleAbs
	if nt == ss.NetBidir {
		hid = nt.LayerByName("Hidden2").(leabra.LeabraLayer).AsLeabra()
		hid.Act.Gbar.I = ss.HiddenGbarI
		hid.Act.Dt.GTau = ss.HiddenGTau
		hid.Act.Update()
		inh = nt.LayerByName("Inhib2").(leabra.LeabraLayer).AsLeabra()
		inh.Act.Gbar.I = ss.InhibGbarI
		inh.Act.Dt.GTau = ss.InhibGTau
		inh.Act.Update()
		hid.Inhib.Layer.On = ss.FFFBInhib
		inh.Inhib.Layer.On = ss.FFFBInhib
		fi = hid.RcvPrjns.SendName("Inhib2").(leabra.LeabraPrjn).AsLeabra()
		fi.WtScale.Abs = ss.FmInhibWtScaleAbs
		fi = inh.RcvPrjns.SendName("Inhib2").(leabra.LeabraPrjn).AsLeabra()
		fi.WtScale.Abs = ss.FmInhibWtScaleAbs
		ff = inh.RcvPrjns.SendName("Hidden").(leabra.LeabraPrjn).AsLeabra()
		ff.WtScale.Rel = ffinhsc
		fb = inh.RcvPrjns.SendName("Hidden2").(leabra.LeabraPrjn).AsLeabra()
		fb.WtScale.Rel = ss.FBinhibWtScale
		inh = nt.LayerByName("Inhib").(leabra.LeabraLayer).AsLeabra()
		ff = inh.RcvPrjns.SendName("Hidden2").(leabra.LeabraPrjn).AsLeabra()
		ff.WtScale.Rel = ffinhsc
	}
	return err
}

// SetParamsSet sets the params for given params.Set name.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParamsSet(setNm string, sheet string, setMsg bool) error {
	nt := ss.Net()
	pset, err := ss.Params.SetByNameTry(setNm)
	if err != nil {
		return err
	}
	if sheet == "" || sheet == "Network" {
		netp, ok := pset.Sheets["Network"]
		if ok {
			nt.ApplyParams(netp, setMsg)
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

// LogTstCyc adds data from current cycle to the TstCycLog table.
// log always contains number of testing items
func (ss *Sim) LogTstCyc(dt *etable.Table, cyc int) {
	nt := ss.Net()
	if dt.Rows <= cyc {
		dt.SetNumRows(cyc + 1)
	}
	row := cyc

	// ly := nt.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	dt.SetCellFloat("Cycle", row, float64(cyc))

	for _, lnm := range ss.TstRecLays {
		ly := nt.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(lnm+"ActAvg", row, float64(ly.Pools[0].Inhib.Act.Avg))
	}

	// note: essential to use Go version of update when called from another goroutine
	if cyc%10 == 0 {
		ss.TstCycPlot.GoUpdate()
	}
}

func (ss *Sim) ConfigTstCycLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstCycLog")
	dt.SetMetaData("desc", "Record of testing per cycle")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	ncy := 200 // max cycles
	sch := etable.Schema{
		{"Cycle", etensor.INT64, nil, nil},
	}
	for _, lnm := range ss.TstRecLays {
		sch = append(sch, etable.Column{lnm + "ActAvg", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, ncy)
}

func (ss *Sim) ConfigTstCycPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Inhib Test Cycle Plot"
	plt.Params.XAxisCol = "Cycle"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Cycle", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

	for _, lnm := range ss.TstRecLays {
		plt.SetColParams(lnm+"ActAvg", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	}
	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("inhib")
	gi.SetAppAbout(`This simulation explores how inhibitory interneurons can dynamically
control overall activity levels within the network, by providing both
feedforward and feedback inhibition to excitatory pyramidal neurons.
  See <a href="https://github.com/CompCogNeuro/sims/ch3/inhib/README.md">README.md on GitHub</a>.</p>`)

	win := gi.NewMainWindow("inhib", "Inhibition", width, height)
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

	nv := tv.AddNewTab(netview.KiT_NetView, "FF Net").(*netview.NetView)
	nv.Var = "Act"
	nv.Params.MaxRecs = 200
	nv.SetNet(ss.NetFF)
	ss.NetViewFF = nv
	nv.ViewDefaults()

	nv = tv.AddNewTab(netview.KiT_NetView, "Bidir Net").(*netview.NetView)
	nv.Var = "Act"
	nv.Params.MaxRecs = 200
	nv.SetNet(ss.NetBidir)
	ss.NetViewBidir = nv

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
			ss.TestTrial() // show every update
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Config Pats", Icon: "update", Tooltip: "Generates a new input pattern based on current InputPct amount.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.ConfigPats()
			vp.SetNeedsFullRender()
		}
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
			gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch3/inhib/README.md")
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
	},
}

func mainrun() {
	TheSim.New()
	TheSim.Config()

	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}
