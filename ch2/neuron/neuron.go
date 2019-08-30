// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
neuron: This simulation illustrates the basic properties of neural spiking and
rate-code activation, reflecting a balance of excitatory and inhibitory
influences (including leak and synaptic inhibition).
*/
package main

import (
	"fmt"
	"log"
	"strconv"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
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
					"Layer.Inhib.Layer.On":  "false",
					"Layer.Act.XX1.Gain":    "30",
					"Layer.Act.XX1.NVar":    "0.01",
					"Layer.Act.Noise.Dist":  "Gaussian",
					"Layer.Act.Noise.Var":   "0",
					"Layer.Act.Noise.Type":  "GeNoise",
					"Layer.Act.Noise.Fixed": "false",
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
	Spike        bool            `desc:"use discrete spiking equations -- otherwise use Noisy X-over-X-plus-1 rate code activation function"`
	GbarE        float32         `def:"0.3" desc:"excitatory conductance multiplier -- determines overall value of Ge which drives neuron to be more excited -- pushes up over threshold to fire if strong enough"`
	GbarL        float32         `def:"0.3" desc:"leak conductance -- determines overall value of Gl which drives neuron to be less excited (inhibited) -- pushes back to resting membrane potential"`
	Noise        float32         `min:"0" step:"0.01" desc:"the variance parameter for Gaussian noise added to unit activations on every cycle"`
	NCycles      int             `def:"200" desc:"total number of cycles to run"`
	OnCycle      int             `def:"10" desc:"when does excitatory input into neuron come on?"`
	OffCycle     int             `def:"160" desc:"when does excitatory input into neuron go off?"`
	UpdtInterval int             `def:"10"  desc:"how often to update display (in cycles)"`
	Net          *leabra.Network `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	SpikeParams  SpikeActParams  `view:"no-inline" desc:"parameters for spiking funcion"`
	SpikeNeuron  SpikeNeuron     `desc:"state parameters for spiking neuron"`
	TstCycLog    *etable.Table   `view:"no-inline" desc:"testing trial-level log data -- click to see record of network's response to each input"`
	Params       params.Sets     `view:"no-inline" desc:"full collection of param sets -- not really interesting for this model"`

	Cycle int `interactive:"-" desc:"current cycle of updating"`

	// internal state - view:"-"
	Win        *gi.Window       `view:"-" desc:"main GUI window"`
	NetView    *netview.NetView `view:"-" desc:"the network viewer"`
	ToolBar    *gi.ToolBar      `view:"-" desc:"the master toolbar"`
	TstCycPlot *eplot.Plot2D    `view:"-" desc:"the test-trial plot"`
	IsRunning  bool             `view:"-" desc:"true if sim is running"`
	StopNow    bool             `view:"-" desc:"flag to stop running"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &leabra.Network{}
	ss.TstCycLog = &etable.Table{}
	ss.Params = ParamSets
	ss.Defaults()
	ss.SpikeParams.Defaults()
}

// Defaults sets default params
func (ss *Sim) Defaults() {
	ss.UpdtInterval = 10
	ss.Cycle = 0
	ss.Spike = true
	ss.GbarE = 0.3
	ss.GbarL = 0.3
	ss.Noise = 0
	ss.NCycles = 200
	ss.OnCycle = 10
	ss.OffCycle = 160
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigNet(ss.Net)
	ss.ConfigTstCycLog(ss.TstCycLog)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.InitName(net, "Neuron")
	net.AddLayer2D("Neuron", 1, 1, emer.Hidden)

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

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Cycle = 0
	ss.InitWts(ss.Net)
	ss.SpikeNeuron.InitAct()
	ss.StopNow = false
	ss.SetParams("", false) // all sheets
	ss.UpdateView()
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters() string {
	return fmt.Sprintf("Cycle:\t%d\t\t\t", ss.Cycle)
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

// RunCycles updates neuron over specified number of cycles
func (ss *Sim) RunCycles() {
	ss.StopNow = false
	ss.Net.InitActs()
	ss.SetParams("", false)
	inputOn := false
	for cyc := 0; cyc < ss.NCycles; cyc++ {
		ss.Cycle = cyc
		switch cyc {
		case ss.OnCycle:
			inputOn = true
		case ss.OffCycle:
			inputOn = false
		}
		if ss.Spike {
			ss.SpikeUpdt(ss.Net, inputOn)
		} else {
			ss.RateUpdt(ss.Net, inputOn)
		}
		ss.LogTstCyc(ss.TstCycLog, ss.Cycle)
		if ss.Cycle%ss.UpdtInterval == 0 {
			ss.UpdateView()
		}
		if ss.StopNow {
			break
		}
	}
	ss.UpdateView()
}

// RateUpdt updates the neuron in rate-code mode
// this just calls the relevant activation code directly, bypassing most other stuff.
func (ss *Sim) RateUpdt(nt *leabra.Network, inputOn bool) {
	ly := ss.Net.LayerByName("Neuron").(*leabra.Layer)
	nrn := &(ly.Neurons[0])
	if inputOn {
		nrn.Ge = 1
	} else {
		nrn.Ge = 0
	}
	nrn.Gi = 0
	ly.Act.VmFmG(nrn)
	ly.Act.ActFmG(nrn)
	nrn.Ge = nrn.Ge * ly.Act.Gbar.E // display effective Ge
}

// SpikeUpdt updates the neuron in spiking mode
// which is just computed directly as spiking is not yet implemented in main codebase
func (ss *Sim) SpikeUpdt(nt *leabra.Network, inputOn bool) {
	ly := ss.Net.LayerByName("Neuron").(*leabra.Layer)
	nrn := &(ly.Neurons[0])
	if inputOn {
		nrn.Ge = 1
	} else {
		nrn.Ge = 0
	}
	nrn.Gi = 0
	ss.SpikeParams.SpikeVmFmG(nrn, &ss.SpikeNeuron)
	ss.SpikeParams.SpikeActFmVm(nrn, &ss.SpikeNeuron)
	nrn.Ge = nrn.Ge * ly.Act.Gbar.E // display effective Ge
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.StopNow = true
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
	ly := ss.Net.LayerByName("Neuron").(*leabra.Layer)
	ly.Act.Gbar.E = float32(ss.GbarE)
	ly.Act.Gbar.L = float32(ss.GbarL)
	ly.Act.Noise.Var = float64(ss.Noise)
	ly.Act.Update()
	ss.SpikeParams.ActParams = ly.Act // keep sync'd
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

//////////////////////////////////////////////
//  TstCycLog

// LogTstCyc adds data from current cycle to the TstCycLog table.
// log always contains number of testing items
func (ss *Sim) LogTstCyc(dt *etable.Table, cyc int) {
	if dt.Rows <= cyc {
		dt.SetNumRows(cyc + 1)
	}
	row := cyc

	ly := ss.Net.LayerByName("Neuron").(*leabra.Layer)
	nrn := &(ly.Neurons[0])

	dt.SetCellFloat("Cycle", row, float64(cyc))
	dt.SetCellFloat("Ge", row, float64(nrn.Ge))
	dt.SetCellFloat("Act", row, float64(nrn.Act))
	dt.SetCellFloat("ActEq", row, float64(nrn.ActAvg))
	dt.SetCellFloat("Inet", row, float64(nrn.Inet))
	dt.SetCellFloat("Vm", row, float64(nrn.Vm))

	// note: essential to use Go version of update when called from another goroutine
	if cyc%ss.UpdtInterval == 0 {
		ss.TstCycPlot.GoUpdate()
	}
}

func (ss *Sim) ConfigTstCycLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstCycLog")
	dt.SetMetaData("desc", "Record of testing per cycle")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.NCycles // max cycles
	sch := etable.Schema{
		{"Cycle", etensor.INT64, nil, nil},
		{"Ge", etensor.FLOAT64, nil, nil},
		{"Act", etensor.FLOAT64, nil, nil},
		{"ActEq", etensor.FLOAT64, nil, nil},
		{"Inet", etensor.FLOAT64, nil, nil},
		{"Vm", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTstCycPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Neuron Test Cycle Plot"
	plt.Params.XAxisCol = "Cycle"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Cycle", false, true, 0, false, 0)
	plt.SetColParams("Ge", true, true, 0, true, 1)
	plt.SetColParams("Act", true, true, 0, true, 1)
	plt.SetColParams("ActEq", true, true, 0, true, 1)
	plt.SetColParams("Inet", true, true, 0, true, 1)
	plt.SetColParams("Vm", true, true, 0, true, 1)
	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	// nv.Scene().Camera.Pose.Pos.Set(0, 1.5, 3.0) // more "head on" than default which is more "top down"
	// nv.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("neuron")
	gi.SetAppAbout(`This simulation illustrates the basic properties of neural spiking and
rate-code activation, reflecting a balance of excitatory and inhibitory
influences (including leak and synaptic inhibition).
See <a href="https://github.com/CompCogNeuro/sims/ch2/neuron/README.md">README.md on GitHub</a>.</p>`)

	win := gi.NewWindow2D("neuron", "Neuron", width, height, true)
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = gi.X
	split.SetStretchMaxWidth()
	split.SetStretchMaxHeight()

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

	tbar.AddAction(gi.ActOpts{Label: "Run Cycles", Icon: "step-fwd", Tooltip: "Runs neuron updating over NCycles.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.RunCycles()
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Defaults", Icon: "reset", Tooltip: "Restore initial default parameters.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Defaults()
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch2/neuron/README.md")
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
