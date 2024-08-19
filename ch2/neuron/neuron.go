// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
neuron: This simulation illustrates the basic properties of neural spiking and
rate-code activation, reflecting a balance of excitatory and inhibitory
influences (including leak and synaptic inhibition).
*/
package main

//go:generate core generate -add-types

import (
	"fmt"
	"log"
	"reflect"

	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32/minmax"
	"cogentcore.org/core/tree"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/estats"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/netview"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/leabra/v2/leabra"
	"github.com/emer/leabra/v2/spike"
)

func main() {
	sim := &Sim{}
	sim.New()
	sim.ConfigAll()
	sim.RunGUI()
}

// ParamSets is the default set of parameters
var ParamSets = params.Sets{
	"Base": {
		{Sel: "Path", Desc: "no learning",
			Params: params.Params{
				"Path.Learn.Learn": "false",
			}},
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
	Net *leabra.Network `display:"no-inline"`

	SpikeParams spike.ActParams `view:"no-inline"`
	// testing trial-level log data -- click to see record of network's response to each input

	// leabra timing parameters and state
	Context leabra.Context

	// contains computed statistic values
	Stats estats.Stats

	// logging
	Logs elog.Logs `display:"no-inline"`

	// all parameter management
	Params emer.NetParams `display:"inline"`

	// current cycle of updating
	Cycle int `edit:"-"`

	// netview update parameters
	ViewUpdate netview.ViewUpdate `display:"inline"`

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
	ss.Params.Config(ParamSets, "", "", ss.Net)
	ss.UpdateInterval = 10
	ss.Cycle = 0
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
	in := net.AddLayer2D("Input", 1, 1, leabra.InputLayer)
	hid := net.AddLayer2D("Neuron", 1, 1, leabra.SuperLayer)

	net.ConnectLayers(in, hid, paths.NewFull(), leabra.ForwardPath)

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
	ss.GUI.UpdatePlot(etime.Test, etime.Cycle)
	ss.GUI.ViewUpdate.Text = ss.Counters()
	ss.GUI.ViewUpdate.UpdateCycle(int(ss.Context.Cycle))
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// RunCycles updates neuron over specified number of cycles
func (ss *Sim) RunCycles() {
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
		ss.Logs.LogRow(etime.Test, etime.Cycle, cyc)
		ss.RecordValues(cyc)
		if cyc%ss.UpdateInterval == 0 {
			ss.UpdateView()
		}
		ss.Context.CycleInc()
		if ss.GUI.StopNow {
			break
		}
	}
	ss.UpdateView()
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
}

func (ss *Sim) ConfigLogs() {
	ss.ConfigLogItems()
	ss.Logs.CreateTables()

	ss.Logs.PlotItems("Vm", "Spike")

	ss.Logs.SetContext(&ss.Stats, ss.Net)
	ss.Logs.ResetLog(etime.Test, etime.Cycle)
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
		cvnm := vnm // closure
		lg.AddItem(&elog.Item{
			Name:   cvnm,
			Type:   reflect.Float64,
			FixMax: false,
			Range:  minmax.F32{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					vl := ly.UnitValue(cvnm, []int{0, 0}, 0)
					ctx.SetFloat32(vl)
				}}})
	}

}

func (ss *Sim) ResetTstCycPlot() {
	ss.Logs.ResetLog(etime.Test, etime.Cycle)
	ss.GUI.UpdatePlot(etime.Test, etime.Cycle)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	// nv.ViewDefaults()
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "Neuron"
	ss.GUI.MakeBody(ss, "neuron", title, `This simulation illustrates the basic properties of neural spiking and rate-code activation, reflecting a balance of excitatory and inhibitory influences (including leak and synaptic inhibition). See <a href="https://github.com/emer/leabra/blob/main/examples/neuron/README.md">README.md on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("NetView")
	nv.Var = "Act"
	nv.SetNet(ss.Net)
	ss.ConfigNetView(nv) // add labels etc
	ss.ViewUpdate.Config(nv, etime.AlphaCycle, etime.AlphaCycle)
	ss.GUI.ViewUpdate = &ss.ViewUpdate

	ss.GUI.AddPlots(title, &ss.Logs)
	// key := etime.Scope(etime.Test, etime.Cycle)
	// plt := ss.GUI.NewPlot(key, ss.GUI.Tabs.NewTab("TstCycPlot"))
	// plt.SetTable(ss.Logs.Table(etime.Test, etime.Cycle))
	// egui.ConfigPlotFromLog("Neuron", plt, &ss.Logs, key)
	// ss.TstCycPlot = plt

	ss.GUI.Body.AddAppBar(func(p *tree.Plan) {
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
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Run Cycles", Icon: icons.PlayArrow,
			Tooltip: "Runs neuron updating over NCycles.",
			Active:  egui.ActiveStopped,
			Func: func() {
				if !ss.GUI.IsRunning {
					go func() {
						ss.GUI.IsRunning = true
						ss.RunCycles()
						ss.GUI.IsRunning = false
						ss.GUI.UpdateWindow()
					}()
				}
			},
		})
		tree.Add(p, func(w *core.Separator) {})
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Reset Plot", Icon: icons.Update,
			Tooltip: "Reset TstCycPlot.",
			Active:  egui.ActiveStopped,
			Func: func() {
				ss.ResetTstCycPlot()
				ss.GUI.UpdateWindow()
			},
		})

		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Defaults", Icon: icons.Update,
			Tooltip: "Restore initial default parameters.",
			Active:  egui.ActiveStopped,
			Func: func() {
				ss.Defaults()
				ss.Init()
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
	})
	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}
