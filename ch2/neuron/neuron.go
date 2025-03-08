// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// neuron: This simulation illustrates the basic properties of neural spiking and
// rate-code activation, reflecting a balance of excitatory and inhibitory
// influences (including leak and synaptic inhibition).
package main

//go:generate core generate -add-types

import (
	"embed"
	"fmt"
	"log"
	"reflect"
	"time"

	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32/minmax"
	"cogentcore.org/core/system"
	"cogentcore.org/core/tree"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/estats"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/netview"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/leabra/v2/leabra"
	"github.com/emer/leabra/v2/spike"
)

//go:embed README.md
var readme embed.FS

func main() {
	sim := &Sim{}
	sim.New()
	sim.ConfigAll()
	sim.RunGUI()
}

// ParamSets is the default set of parameters
var ParamSets = params.Sets{
	"Base": {
		{Sel: "Layer", Desc: "generic params for all layers: lower gain, slower, soft clamp",
			Params: params.Params{
				"Layer.Inhib.Layer.On":  "false",
				"Layer.Act.XX1.Gain":    "30",
				"Layer.Act.XX1.NVar":    "0.01",
				"Layer.Act.Init.Vm":     "0.3",
				"Layer.Act.Noise.Dist":  "Gaussian",
				"Layer.Act.Noise.Var":   "0",
				"Layer.Act.Noise.Type":  "GeNoise",
				"Layer.Act.Noise.Fixed": "false",
			}},
	},
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {

	// use discrete spiking equations -- otherwise use Noisy X-over-X-plus-1 rate code activation function
	Spike bool

	// excitatory conductance multiplier -- determines overall value of Ge which drives neuron to be more excited -- pushes up over threshold to fire if strong enough
	GbarE float32 `min:"0" step:"0.01" def:"0.3"`

	// leak conductance -- determines overall value of Gl which drives neuron to be less excited (inhibited) -- pushes back to resting membrane potential
	GbarL float32 `min:"0" step:"0.01" def:"0.3"`

	// excitatory reversal (driving) potential -- determines where excitation pushes Vm up to
	ErevE float32 `min:"0" max:"1" step:"0.01" def:"1"`

	// leak reversal (driving) potential -- determines where excitation pulls Vm down to
	ErevL float32 `min:"0" max:"1" step:"0.01" def:"0.3"`

	// the variance parameter for Gaussian noise added to unit activations on every cycle
	Noise float32 `min:"0" step:"0.01"`

	// apply sodium-gated potassium adaptation mechanisms that cause the neuron to reduce spiking over time
	KNaAdapt bool

	// total number of cycles to run
	NCycles int `min:"10" def:"200"`

	// when does excitatory input into neuron come on?
	OnCycle int `min:"0" def:"10"`

	// when does excitatory input into neuron go off?
	OffCycle int `min:"0" def:"160"`

	// how often to update display (in cycles)
	UpdateInterval int `min:"1" def:"10"`

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *leabra.Network `display:"-"`

	SpikeParams spike.ActParams `view:"no-inline"`
	// testing trial-level log data -- click to see record of network's response to each input

	// leabra timing parameters and state
	Context leabra.Context `display:"-"`

	// contains computed statistic values
	Stats estats.Stats `display:"-"`

	// logging
	Logs elog.Logs `display:"-"`

	// all parameter management
	Params emer.NetParams `display:"-"`

	// netview update parameters
	ViewUpdate netview.ViewUpdate `display:"add-fields"`

	// manages all the gui elements
	GUI egui.GUI `display:"-"`

	// map of values for detailed debugging / testing
	ValMap map[string]float32 `display:"-"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = leabra.NewNetwork("Neuron")
	ss.Defaults()
	ss.Stats.Init()
	ss.ValMap = make(map[string]float32)
}

func (ss *Sim) Defaults() {
	ss.SpikeParams.Defaults()
	ss.Params.Config(ParamSets, "", "", ss.Net)
	ss.UpdateInterval = 10
	ss.Spike = true
	ss.GbarE = 0.3
	ss.GbarL = 0.3
	ss.ErevE = 1
	ss.ErevL = 0.3
	ss.Noise = 0
	ss.KNaAdapt = true
	ss.NCycles = 200
	ss.OnCycle = 10
	ss.OffCycle = 160
}

/////////////////////////////////////////////////////////////////////////////
// 		Configs

// ConfigAll configures all the elements using the standard functions
func (ss *Sim) ConfigAll() {
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.AddLayer2D("Neuron", 1, 1, leabra.SuperLayer)
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.Defaults()
	ss.SetParams("Network", false) // only set Network params
	ss.InitWeights(net)
}

// InitWeights loads the saved weights
func (ss *Sim) InitWeights(net *leabra.Network) {
	net.InitWeights()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Context.Reset()
	ss.InitWeights(ss.Net)
	ss.GUI.StopNow = false
	ss.SetParams("", false) // all sheets
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters() string {
	return fmt.Sprintf("Cycle:\t%d\t\t\t", ss.Context.Cycle)
}

func (ss *Sim) UpdateView() {
	ss.GUI.GoUpdatePlot(etime.Test, etime.Cycle)
	ss.GUI.ViewUpdate.Text = ss.Counters()
	ss.GUI.ViewUpdate.UpdateCycle(int(ss.Context.Cycle))
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// RunCycles updates neuron over specified number of cycles
func (ss *Sim) RunCycles(updt bool) {
	ctx := &ss.Context
	ss.Init()
	ss.GUI.StopNow = false
	ss.Net.InitActs()
	ctx.AlphaCycStart()
	ss.SetParams("", false)
	ly := ss.Net.LayerByName("Neuron")
	nrn := &(ly.Neurons[0])
	inputOn := false
	for cyc := 0; cyc < ss.NCycles; cyc++ {
		switch cyc {
		case ss.OnCycle:
			inputOn = true
		case ss.OffCycle:
			inputOn = false
		}
		nrn.Noise = float32(ly.Act.Noise.Gen())
		if inputOn {
			nrn.Ge = 1
		} else {
			nrn.Ge = 0
		}
		nrn.Ge += nrn.Noise // GeNoise
		nrn.Gi = 0
		if ss.Spike {
			ss.SpikeUpdate(ss.Net, inputOn)
		} else {
			ss.RateUpdate(ss.Net, inputOn)
		}
		ctx.Cycle = cyc
		ss.Logs.Log(etime.Test, etime.Cycle)
		ss.RecordValues(cyc)
		if updt && cyc%ss.UpdateInterval == 0 {
			ss.UpdateView()
		}
		ss.Context.CycleInc()
		if ss.GUI.StopNow {
			break
		}
	}
	if updt {
		ss.UpdateView()
	}
}

// RateUpdate updates the neuron in rate-code mode
// this just calls the relevant activation code directly, bypassing most other stuff.
func (ss *Sim) RateUpdate(nt *leabra.Network, inputOn bool) {
	ly := ss.Net.LayerByName("Neuron")
	nrn := &(ly.Neurons[0])
	ly.Act.VmFromG(nrn)
	ly.Act.ActFromG(nrn)
	nrn.Ge = nrn.Ge * ly.Act.Gbar.E // display effective Ge
}

// SpikeUpdate updates the neuron in spiking mode
// which is just computed directly as spiking is not yet implemented in main codebase
func (ss *Sim) SpikeUpdate(nt *leabra.Network, inputOn bool) {
	ly := ss.Net.LayerByName("Neuron")
	nrn := &(ly.Neurons[0])
	ss.SpikeParams.SpikeVmFromG(nrn)
	ss.SpikeParams.SpikeActFromVm(nrn)
	nrn.Ge = nrn.Ge * ly.Act.Gbar.E // display effective Ge
}

func (ss *Sim) RecordValues(cyc int) {
	var vals []float32
	ly := ss.Net.LayerByName("Neuron")
	key := fmt.Sprintf("cyc: %03d", cyc)
	for _, vnm := range leabra.NeuronVars {
		ly.UnitValues(&vals, vnm, 0)
		vkey := key + fmt.Sprintf("\t%s", vnm)
		ss.ValMap[vkey] = vals[0]
	}
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.GUI.StopNow = true
}

// SpikeVsRate runs comparison between spiking vs. rate-code
func (ss *Sim) SpikeVsRate() {
	row := 0
	nsamp := 100
	// ss.KNaAdapt = false
	tcl := ss.Logs.Table(etime.Test, etime.Cycle)
	svr := ss.Logs.MiscTable("SpikeVsRate")
	svp := ss.GUI.Plots[etime.ScopeKey("SpikeVsRate")]
	for gbarE := 0.1; gbarE <= 0.7; gbarE += 0.025 {
		ss.GbarE = float32(gbarE)
		spike := float64(0)
		ss.Noise = 0.1 // RunCycles calls SetParams to set this
		ss.Spike = true
		for ns := 0; ns < nsamp; ns++ {
			tcl.Rows = 0
			ss.RunCycles(false)
			if ss.GUI.StopNow {
				break
			}
			act := tcl.Float("Act", 159)
			spike += act
		}
		rate := float64(0)
		ss.Spike = false
		// ss.Noise = 0 // doesn't make much diff
		for ns := 0; ns < nsamp; ns++ {
			tcl.Rows = 0
			ss.RunCycles(false)
			if ss.GUI.StopNow {
				break
			}
			act := tcl.Float("Act", 159)
			rate += act
			if core.TheApp.Platform() == system.Web {
				time.Sleep(time.Millisecond) // critical to prevent hanging!
			}
		}
		if ss.GUI.StopNow {
			break
		}
		spike /= float64(nsamp)
		rate /= float64(nsamp)
		svr.AddRows(1)
		svr.SetFloat("GBarE", row, gbarE)
		svr.SetFloat("Spike", row, spike)
		svr.SetFloat("Rate", row, rate)
		svp.GoUpdatePlot()
		row++
	}
	ss.Defaults()
	svp.GoUpdatePlot()
	ss.GUI.IsRunning = false
	ss.GUI.UpdateWindow()
}

/////////////////////////////////////////////////////////////////////////
//   Params setting

// SetParams sets the params for "Base" and then current ParamSet.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParams(sheet string, setMsg bool) {
	ss.Params.SetAll()
	ly := ss.Net.LayerByName("Neuron")
	ly.Act.Gbar.E = float32(ss.GbarE)
	ly.Act.Gbar.L = float32(ss.GbarL)
	ly.Act.Erev.E = float32(ss.ErevE)
	ly.Act.Erev.L = float32(ss.ErevL)
	ly.Act.Noise.Var = float64(ss.Noise)
	ly.Act.KNa.On = ss.KNaAdapt
	ly.Act.Update()
	ss.SpikeParams.ActParams = ly.Act // keep sync'd
	ss.SpikeParams.KNa.On = ss.KNaAdapt
	ly.UpdateParams()
}

func (ss *Sim) ConfigLogs() {
	ss.ConfigLogItems()
	ss.Logs.CreateTables()

	ss.Logs.PlotItems("Ge", "Inet", "Vm", "Act", "Spike", "Gk")

	ss.Logs.SetContext(&ss.Stats, ss.Net)
	ss.Logs.ResetLog(etime.Test, etime.Cycle)

	svr := ss.Logs.MiscTable("SpikeVsRate")
	svr.AddFloat64Column("GBarE")
	svr.AddFloat64Column("Spike")
	svr.AddFloat64Column("Rate")
	svr.SetMetaData("Rate:On", "+")
}

func (ss *Sim) ConfigLogItems() {
	ly := ss.Net.LayerByName("Neuron")
	lg := &ss.Logs

	lg.AddItem(&elog.Item{
		Name:   "Cycle",
		Type:   reflect.Int,
		FixMax: false,
		Range:  minmax.F32{Max: 1},
		Write: elog.WriteMap{
			etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
				ctx.SetInt(int(ss.Context.Cycle))
			}}})

	vars := []string{"Ge", "Inet", "Vm", "Act", "Spike", "Gk", "ISI", "AvgISI"}

	for _, vnm := range vars {
		lg.AddItem(&elog.Item{
			Name:   vnm,
			Type:   reflect.Float64,
			FixMax: false,
			Range:  minmax.F32{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					vl := ly.UnitValue(vnm, []int{0, 0}, 0)
					ctx.SetFloat32(vl)
				}}})
	}

}

func (ss *Sim) ResetTestCyclePlot() {
	ss.Logs.ResetLog(etime.Test, etime.Cycle)
	ss.GUI.UpdatePlot(etime.Test, etime.Cycle)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		GUI

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	// nv.ViewDefaults()
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "Neuron"
	ss.GUI.MakeBody(ss, "neuron", title, `This simulation illustrates the basic properties of neural spiking and rate-code activation, reflecting a balance of excitatory and inhibitory influences (including leak and synaptic inhibition). See <a href="https://github.com/CompCogNeuro/sims/blob/main/ch2/neuron/README.md">README.md on GitHub</a>.</p>`, readme)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
	nv.Var = "Act"
	nv.Options.Raster.Max = 100
	nv.SetNet(ss.Net)
	ss.ConfigNetView(nv) // add labels etc
	ss.ViewUpdate.Config(nv, etime.AlphaCycle, etime.AlphaCycle)
	ss.GUI.ViewUpdate = &ss.ViewUpdate

	ss.GUI.AddPlots(title, &ss.Logs)

	svr := "SpikeVsRate"
	dt := ss.Logs.MiscTable(svr)
	plt := ss.GUI.AddMiscPlotTab(svr + " Plot")
	plt.Options.Title = svr
	plt.Options.XAxis = "GBarE"
	plt.SetTable(dt)

	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Init", Icon: icons.Update,
		Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Init()
			ss.GUI.UpdateWindow()
		},
	})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Stop", Icon: icons.Stop,
		Tooltip: "Stops running.",
		Active:  egui.ActiveRunning,
		Func: func() {
			ss.Stop()
			ss.GUI.UpdateWindow()
		},
	})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Run cycles", Icon: icons.PlayArrow,
		Tooltip: "Runs neuron updating over NCycles.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				go func() {
					ss.GUI.IsRunning = true
					ss.RunCycles(true)
					ss.GUI.IsRunning = false
					ss.GUI.UpdateWindow()
				}()
			}
		},
	})
	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Reset plot", Icon: icons.Update,
		Tooltip: "Reset TestCyclePlot.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.ResetTestCyclePlot()
			ss.GUI.UpdateWindow()
		},
	})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Spike Vs Rate", Icon: icons.PlayArrow,
		Tooltip: "Generate a plot of actual spiking rate vs computed NXX1 rate code.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.GUI.IsRunning = true
			go ss.SpikeVsRate()
			ss.GUI.UpdateWindow()
		},
	})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Defaults", Icon: icons.Update,
		Tooltip: "Restore initial default parameters.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Defaults()
			ss.Init()
			ss.GUI.SimForm.Update()
			ss.GUI.UpdateWindow()
		},
	})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "README",
		Icon:    icons.FileMarkdown,
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.TheApp.OpenURL("https://github.com/CompCogNeuro/sims/blob/main/ch2/neuron/README.md")
		},
	})
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}
